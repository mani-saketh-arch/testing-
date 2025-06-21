"""
SafeIndy Assistant - Analytics Service
Usage tracking, metrics collection, and dashboard data
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..config import settings
from ..database import (
    get_db_context, 
    UserInteraction, 
    Document, 
    DocumentImage,
    SystemLog,
    log_user_interaction
)
from ..utils import (
    timer, 
    log_error, 
    cache,
    calculate_distance,
    is_in_indianapolis,
    format_coordinates,
    format_timestamp
)

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Analytics and metrics collection service"""
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes cache for dashboard data
        logger.info("✅ Analytics service initialized")
    
    @timer
    async def track_user_interaction(
        self,
        session_id: str,
        user_query: str,
        response_data: Dict[str, Any],
        request_info: Dict[str, Any]
    ) -> bool:
        """Track user interaction for analytics"""
        
        try:
            with get_db_context() as db:
                log_user_interaction(
                    db=db,
                    session_id=session_id,
                    user_query=user_query,
                    query_type=response_data.get("intent", "unknown"),
                    intent_confidence=response_data.get("confidence", 0.0),
                    latitude=request_info.get("latitude"),
                    longitude=request_info.get("longitude"),
                    location_accuracy=request_info.get("location_accuracy"),
                    response_time_ms=response_data.get("response_time_ms", 0),
                    emergency_detected=response_data.get("emergency", False),
                    platform=request_info.get("platform", "web"),
                    user_agent=request_info.get("user_agent"),
                    response_length=len(response_data.get("response", "")),
                    sources_used=json.dumps(response_data.get("sources", []))
                )
            
            logger.debug(f"Tracked interaction for session {session_id}")
            return True
        
        except Exception as e:
            log_error(e, "Tracking user interaction", {"session_id": session_id})
            return False
    
    @timer
    async def get_dashboard_overview(self, time_period: str = "24h") -> Dict[str, Any]:
        """Get dashboard overview statistics"""
        
        cache_key = f"dashboard_overview_{time_period}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Calculate time range
            time_ranges = {
                "1h": timedelta(hours=1),
                "24h": timedelta(hours=24),
                "7d": timedelta(days=7),
                "30d": timedelta(days=30),
                "90d": timedelta(days=90)
            }
            
            time_delta = time_ranges.get(time_period, timedelta(hours=24))
            start_time = datetime.utcnow() - time_delta
            
            with get_db_context() as db:
                # Basic metrics
                total_interactions = db.query(UserInteraction).filter(
                    UserInteraction.timestamp >= start_time
                ).count()
                
                unique_sessions = db.query(func.count(func.distinct(UserInteraction.session_id))).filter(
                    UserInteraction.timestamp >= start_time
                ).scalar() or 0
                
                emergency_requests = db.query(UserInteraction).filter(
                    and_(
                        UserInteraction.timestamp >= start_time,
                        UserInteraction.emergency_detected == True
                    )
                ).count()
                
                # Average response time
                avg_response_time = db.query(func.avg(UserInteraction.response_time_ms)).filter(
                    and_(
                        UserInteraction.timestamp >= start_time,
                        UserInteraction.response_time_ms.isnot(None)
                    )
                ).scalar() or 0
                
                # Top query types
                query_types = db.query(
                    UserInteraction.query_type,
                    func.count(UserInteraction.id).label('count')
                ).filter(
                    UserInteraction.timestamp >= start_time
                ).group_by(UserInteraction.query_type).all()
                
                # Platform distribution
                platforms = db.query(
                    UserInteraction.platform,
                    func.count(UserInteraction.id).label('count')
                ).filter(
                    UserInteraction.timestamp >= start_time
                ).group_by(UserInteraction.platform).all()
                
                # Document statistics
                total_documents = db.query(Document).count()
                processed_documents = db.query(Document).filter(
                    Document.status == "completed"
                ).count()
                
                # Recent activity (last 24 hours broken down by hour)
                hourly_activity = []
                for i in range(24):
                    hour_start = datetime.utcnow() - timedelta(hours=i+1)
                    hour_end = datetime.utcnow() - timedelta(hours=i)
                    
                    hour_count = db.query(UserInteraction).filter(
                        and_(
                            UserInteraction.timestamp >= hour_start,
                            UserInteraction.timestamp < hour_end
                        )
                    ).count()
                    
                    hourly_activity.append({
                        "hour": hour_start.strftime("%H:00"),
                        "count": hour_count
                    })
                
                hourly_activity.reverse()  # Show chronologically
                
                overview = {
                    "time_period": time_period,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": {
                        "total_interactions": total_interactions,
                        "unique_sessions": unique_sessions,
                        "emergency_requests": emergency_requests,
                        "avg_response_time_ms": round(avg_response_time, 2) if avg_response_time else 0,
                        "total_documents": total_documents,
                        "processed_documents": processed_documents
                    },
                    "query_types": [{"type": qt[0], "count": qt[1]} for qt in query_types],
                    "platforms": [{"platform": p[0], "count": p[1]} for p in platforms],
                    "hourly_activity": hourly_activity
                }
                
                # Cache for 5 minutes
                cache.set(cache_key, overview, self.cache_ttl)
                return overview
        
        except Exception as e:
            log_error(e, f"Getting dashboard overview for {time_period}")
            return {
                "error": str(e),
                "metrics": {},
                "query_types": [],
                "platforms": [],
                "hourly_activity": []
            }
    
    @timer
    async def get_geographic_data(self, time_period: str = "7d") -> Dict[str, Any]:
        """Get geographic distribution of users for mapping"""
        
        cache_key = f"geographic_data_{time_period}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Calculate time range
            time_ranges = {
                "24h": timedelta(hours=24),
                "7d": timedelta(days=7),
                "30d": timedelta(days=30)
            }
            
            time_delta = time_ranges.get(time_period, timedelta(days=7))
            start_time = datetime.utcnow() - time_delta
            
            with get_db_context() as db:
                # Get interactions with location data
                interactions = db.query(UserInteraction).filter(
                    and_(
                        UserInteraction.timestamp >= start_time,
                        UserInteraction.latitude.isnot(None),
                        UserInteraction.longitude.isnot(None)
                    )
                ).all()
                
                # Process geographic data
                location_points = []
                emergency_points = []
                neighborhoods = defaultdict(int)
                
                for interaction in interactions:
                    lat, lon = interaction.latitude, interaction.longitude
                    
                    # Basic location point
                    point = {
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": interaction.timestamp.isoformat(),
                        "query_type": interaction.query_type,
                        "emergency": interaction.emergency_detected
                    }
                    
                    location_points.append(point)
                    
                    # Emergency-specific tracking
                    if interaction.emergency_detected:
                        emergency_points.append({
                            "latitude": lat,
                            "longitude": lon,
                            "timestamp": interaction.timestamp.isoformat(),
                            "query_type": interaction.query_type
                        })
                    
                    # Neighborhood clustering (simplified)
                    neighborhood = self._get_neighborhood_from_coords(lat, lon)
                    neighborhoods[neighborhood] += 1
                
                # Calculate heat map data (grid-based)
                heat_map_data = self._calculate_heat_map(location_points)
                
                # Get Indianapolis boundary info
                indy_center = {
                    "latitude": settings.INDY_LATITUDE,
                    "longitude": settings.INDY_LONGITUDE
                }
                
                geographic_data = {
                    "time_period": time_period,
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_points": len(location_points),
                    "emergency_points_count": len(emergency_points),
                    "indianapolis_center": indy_center,
                    "location_points": location_points[-100:],  # Last 100 points for performance
                    "emergency_points": emergency_points,
                    "heat_map_data": heat_map_data,
                    "neighborhoods": dict(neighborhoods),
                    "statistics": {
                        "points_in_indianapolis": sum(1 for p in location_points 
                                                     if is_in_indianapolis(p["latitude"], p["longitude"])),
                        "average_latitude": sum(p["latitude"] for p in location_points) / len(location_points) if location_points else 0,
                        "average_longitude": sum(p["longitude"] for p in location_points) / len(location_points) if location_points else 0
                    }
                }
                
                # Cache for 10 minutes (geographic data changes less frequently)
                cache.set(cache_key, geographic_data, 600)
                return geographic_data
        
        except Exception as e:
            log_error(e, f"Getting geographic data for {time_period}")
            return {
                "error": str(e),
                "location_points": [],
                "emergency_points": [],
                "heat_map_data": [],
                "neighborhoods": {}
            }
    
    def _get_neighborhood_from_coords(self, latitude: float, longitude: float) -> str:
        """Get neighborhood name from coordinates (simplified mapping)"""
        
        # Indianapolis neighborhood boundaries (approximate)
        neighborhoods = {
            "Downtown": (39.7684, -86.1581, 2),  # lat, lon, radius_miles
            "Broad Ripple": (39.8403, -86.1378, 2),
            "Fountain Square": (39.7370, -86.1478, 1.5),
            "Mass Ave": (39.7717, -86.1478, 1),
            "Irvington": (39.7745, -86.1117, 2),
            "Butler-Tarkington": (39.8100, -86.1700, 2),
        }
        
        for neighborhood, (n_lat, n_lon, radius) in neighborhoods.items():
            distance = calculate_distance(latitude, longitude, n_lat, n_lon)
            if distance <= radius:
                return neighborhood
        
        # Check if it's within Indianapolis boundaries
        if is_in_indianapolis(latitude, longitude):
            return "Other Indianapolis"
        else:
            return "Outside Indianapolis"
    
    def _calculate_heat_map(self, location_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate heat map data for geographic visualization"""
        
        if not location_points:
            return []
        
        # Create grid-based heat map
        # Indianapolis approximate bounds
        lat_min, lat_max = 39.6, 39.9
        lon_min, lon_max = -86.3, -85.9
        
        grid_size = 20  # 20x20 grid
        lat_step = (lat_max - lat_min) / grid_size
        lon_step = (lon_max - lon_min) / grid_size
        
        grid = defaultdict(int)
        
        for point in location_points:
            lat, lon = point["latitude"], point["longitude"]
            
            # Calculate grid cell
            lat_idx = int((lat - lat_min) / lat_step)
            lon_idx = int((lon - lon_min) / lon_step)
            
            # Ensure within bounds
            if 0 <= lat_idx < grid_size and 0 <= lon_idx < grid_size:
                grid[(lat_idx, lon_idx)] += 1
        
        # Convert to heat map format
        heat_map_data = []
        for (lat_idx, lon_idx), count in grid.items():
            center_lat = lat_min + (lat_idx + 0.5) * lat_step
            center_lon = lon_min + (lon_idx + 0.5) * lon_step
            
            heat_map_data.append({
                "latitude": center_lat,
                "longitude": center_lon,
                "intensity": count,
                "radius": min(count * 100, 1000)  # Scale radius by count
            })
        
        return heat_map_data
    
    @timer
    async def get_usage_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get usage trends over time"""
        
        cache_key = f"usage_trends_{days}d"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            with get_db_context() as db:
                # Daily usage trends
                daily_stats = []
                for i in range(days):
                    day_start = start_date + timedelta(days=i)
                    day_end = day_start + timedelta(days=1)
                    
                    day_interactions = db.query(UserInteraction).filter(
                        and_(
                            UserInteraction.timestamp >= day_start,
                            UserInteraction.timestamp < day_end
                        )
                    ).count()
                    
                    day_emergencies = db.query(UserInteraction).filter(
                        and_(
                            UserInteraction.timestamp >= day_start,
                            UserInteraction.timestamp < day_end,
                            UserInteraction.emergency_detected == True
                        )
                    ).count()
                    
                    day_sessions = db.query(func.count(func.distinct(UserInteraction.session_id))).filter(
                        and_(
                            UserInteraction.timestamp >= day_start,
                            UserInteraction.timestamp < day_end
                        )
                    ).scalar() or 0
                    
                    daily_stats.append({
                        "date": day_start.strftime("%Y-%m-%d"),
                        "interactions": day_interactions,
                        "emergencies": day_emergencies,
                        "unique_sessions": day_sessions
                    })
                
                # Query type trends
                query_type_trends = {}
                query_types = db.query(func.distinct(UserInteraction.query_type)).filter(
                    UserInteraction.timestamp >= start_date
                ).all()
                
                for (query_type,) in query_types:
                    if not query_type:
                        continue
                    
                    type_daily_counts = []
                    for i in range(days):
                        day_start = start_date + timedelta(days=i)
                        day_end = day_start + timedelta(days=1)
                        
                        count = db.query(UserInteraction).filter(
                            and_(
                                UserInteraction.timestamp >= day_start,
                                UserInteraction.timestamp < day_end,
                                UserInteraction.query_type == query_type
                            )
                        ).count()
                        
                        type_daily_counts.append({
                            "date": day_start.strftime("%Y-%m-%d"),
                            "count": count
                        })
                    
                    query_type_trends[query_type] = type_daily_counts
                
                trends_data = {
                    "time_period": f"{days} days",
                    "timestamp": datetime.utcnow().isoformat(),
                    "daily_stats": daily_stats,
                    "query_type_trends": query_type_trends,
                    "summary": {
                        "total_interactions": sum(day["interactions"] for day in daily_stats),
                        "total_emergencies": sum(day["emergencies"] for day in daily_stats),
                        "total_unique_sessions": len(set(day["unique_sessions"] for day in daily_stats if day["unique_sessions"] > 0)),
                        "avg_daily_interactions": round(sum(day["interactions"] for day in daily_stats) / days, 1),
                        "peak_day": max(daily_stats, key=lambda x: x["interactions"])["date"] if daily_stats else None
                    }
                }
                
                # Cache for 1 hour
                cache.set(cache_key, trends_data, 3600)
                return trends_data
        
        except Exception as e:
            log_error(e, f"Getting usage trends for {days} days")
            return {
                "error": str(e),
                "daily_stats": [],
                "query_type_trends": {},
                "summary": {}
            }
    
    @timer
    async def get_emergency_analytics(self, time_period: str = "30d") -> Dict[str, Any]:
        """Get detailed emergency request analytics"""
        
        cache_key = f"emergency_analytics_{time_period}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            time_ranges = {
                "24h": timedelta(hours=24),
                "7d": timedelta(days=7),
                "30d": timedelta(days=30),
                "90d": timedelta(days=90)
            }
            
            time_delta = time_ranges.get(time_period, timedelta(days=30))
            start_time = datetime.utcnow() - time_delta
            
            with get_db_context() as db:
                # Get all emergency interactions
                emergency_interactions = db.query(UserInteraction).filter(
                    and_(
                        UserInteraction.timestamp >= start_time,
                        UserInteraction.emergency_detected == True
                    )
                ).all()
                
                # Emergency type analysis (based on query_type)
                emergency_types = Counter(interaction.query_type for interaction in emergency_interactions)
                
                # Response time analysis
                response_times = [interaction.response_time_ms for interaction in emergency_interactions 
                                if interaction.response_time_ms is not None]
                
                # Geographic distribution
                emergency_locations = []
                for interaction in emergency_interactions:
                    if interaction.latitude and interaction.longitude:
                        emergency_locations.append({
                            "latitude": interaction.latitude,
                            "longitude": interaction.longitude,
                            "timestamp": interaction.timestamp.isoformat(),
                            "query_type": interaction.query_type,
                            "response_time_ms": interaction.response_time_ms
                        })
                
                # Time distribution (hourly pattern)
                hourly_pattern = defaultdict(int)
                for interaction in emergency_interactions:
                    hour = interaction.timestamp.hour
                    hourly_pattern[hour] += 1
                
                # Calculate statistics
                analytics = {
                    "time_period": time_period,
                    "timestamp": datetime.utcnow().isoformat(),
                    "summary": {
                        "total_emergencies": len(emergency_interactions),
                        "avg_response_time_ms": round(sum(response_times) / len(response_times), 2) if response_times else 0,
                        "median_response_time_ms": sorted(response_times)[len(response_times)//2] if response_times else 0,
                        "fastest_response_ms": min(response_times) if response_times else 0,
                        "slowest_response_ms": max(response_times) if response_times else 0,
                        "locations_captured": len(emergency_locations),
                        "location_capture_rate": round(len(emergency_locations) / len(emergency_interactions) * 100, 1) if emergency_interactions else 0
                    },
                    "emergency_types": [{"type": et[0], "count": et[1]} for et in emergency_types.most_common()],
                    "hourly_pattern": [{"hour": hour, "count": count} for hour, count in sorted(hourly_pattern.items())],
                    "emergency_locations": emergency_locations,
                    "response_time_distribution": {
                        "under_1s": sum(1 for rt in response_times if rt < 1000),
                        "1s_to_3s": sum(1 for rt in response_times if 1000 <= rt < 3000),
                        "3s_to_5s": sum(1 for rt in response_times if 3000 <= rt < 5000),
                        "over_5s": sum(1 for rt in response_times if rt >= 5000)
                    } if response_times else {}
                }
                
                # Cache for 15 minutes
                cache.set(cache_key, analytics, 900)
                return analytics
        
        except Exception as e:
            log_error(e, f"Getting emergency analytics for {time_period}")
            return {
                "error": str(e),
                "summary": {},
                "emergency_types": [],
                "hourly_pattern": [],
                "emergency_locations": []
            }
    
    @timer
    async def get_document_analytics(self) -> Dict[str, Any]:
        """Get document processing and usage analytics"""
        
        cache_key = "document_analytics"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            with get_db_context() as db:
                # Document statistics
                total_documents = db.query(Document).count()
                
                # Status distribution
                status_counts = db.query(
                    Document.status,
                    func.count(Document.id).label('count')
                ).group_by(Document.status).all()
                
                # Processing time analysis
                processing_times = db.query(
                    func.extract('epoch', Document.processed_at - Document.uploaded_at).label('processing_seconds')
                ).filter(
                    and_(
                        Document.processed_at.isnot(None),
                        Document.uploaded_at.isnot(None)
                    )
                ).all()
                
                processing_seconds = [pt[0] for pt in processing_times if pt[0] is not None]
                
                # File size distribution
                file_sizes = db.query(Document.file_size).filter(
                    Document.file_size.isnot(None)
                ).all()
                
                sizes = [fs[0] for fs in file_sizes if fs[0] is not None]
                
                # Image extraction statistics
                total_images = db.query(DocumentImage).count()
                images_per_doc = db.query(
                    func.count(DocumentImage.id).label('image_count')
                ).group_by(DocumentImage.document_id).all()
                
                analytics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "document_stats": {
                        "total_documents": total_documents,
                        "total_images_extracted": total_images,
                        "avg_images_per_document": round(sum(ipc[0] for ipc in images_per_doc) / len(images_per_doc), 1) if images_per_doc else 0
                    },
                    "status_distribution": [{"status": sc[0], "count": sc[1]} for sc in status_counts],
                    "processing_performance": {
                        "avg_processing_time_seconds": round(sum(processing_seconds) / len(processing_seconds), 2) if processing_seconds else 0,
                        "median_processing_time_seconds": sorted(processing_seconds)[len(processing_seconds)//2] if processing_seconds else 0,
                        "fastest_processing_seconds": min(processing_seconds) if processing_seconds else 0,
                        "slowest_processing_seconds": max(processing_seconds) if processing_seconds else 0
                    },
                    "file_size_stats": {
                        "avg_file_size_mb": round(sum(sizes) / len(sizes) / 1024 / 1024, 2) if sizes else 0,
                        "total_storage_mb": round(sum(sizes) / 1024 / 1024, 2) if sizes else 0,
                        "largest_file_mb": round(max(sizes) / 1024 / 1024, 2) if sizes else 0,
                        "smallest_file_mb": round(min(sizes) / 1024 / 1024, 2) if sizes else 0
                    }
                }
                
                # Cache for 30 minutes
                cache.set(cache_key, analytics, 1800)
                return analytics
        
        except Exception as e:
            log_error(e, "Getting document analytics")
            return {
                "error": str(e),
                "document_stats": {},
                "status_distribution": [],
                "processing_performance": {},
                "file_size_stats": {}
            }
    
    async def export_analytics_data(
        self, 
        data_type: str, 
        time_period: str = "30d",
        format_type: str = "json"
    ) -> Optional[Dict[str, Any]]:
        """Export analytics data for external analysis"""
        
        try:
            if data_type == "overview":
                data = await self.get_dashboard_overview(time_period)
            elif data_type == "geographic":
                data = await self.get_geographic_data(time_period)
            elif data_type == "trends":
                days = int(time_period.replace("d", "")) if time_period.endswith("d") else 30
                data = await self.get_usage_trends(days)
            elif data_type == "emergency":
                data = await self.get_emergency_analytics(time_period)
            elif data_type == "documents":
                data = await self.get_document_analytics()
            else:
                return None
            
            # Add export metadata
            export_data = {
                "export_info": {
                    "data_type": data_type,
                    "time_period": time_period,
                    "format": format_type,
                    "exported_at": datetime.utcnow().isoformat(),
                    "system": "SafeIndy Assistant"
                },
                "data": data
            }
            
            return export_data
        
        except Exception as e:
            log_error(e, f"Exporting analytics data: {data_type}")
            return None


# Global analytics service instance
analytics_service = AnalyticsService()