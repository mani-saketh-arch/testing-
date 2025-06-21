"""
SafeIndy Assistant - LLM Service
AWS Claude Sonnet 4 + Groq integration for AI processing
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import logging
import boto3
from groq import Groq
from botocore.exceptions import ClientError, BotoCoreError

from ..config import settings, LLM_CONFIG, EMERGENCY_KEYWORDS, EMERGENCY_CONFIDENCE_THRESHOLD
from ..utils import timer, log_error, cache

logger = logging.getLogger(__name__)


class LLMService:
    """LLM service with AWS Claude primary and Groq backup"""
    
    def __init__(self):
        self.claude_client = None
        self.groq_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS Bedrock and Groq clients"""
        # Initialize AWS Bedrock client for Claude
        try:
            self.claude_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            logger.info("✅ AWS Claude Sonnet 4 client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS Claude client: {e}")
            self.claude_client = None
        
        # Initialize Groq client
        try:
            self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
            logger.info("✅ Groq backup client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq client: {e}")
            self.groq_client = None
    
    @timer
    async def generate_response(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        emergency_mode: bool = False,
        use_groq: bool = False
    ) -> Dict[str, Any]:
        """Generate AI response with intelligent model selection"""
        
        start_time = time.time()
        
        # Choose model based on emergency mode and availability
        if emergency_mode or use_groq:
            response = await self._generate_groq_response(user_message, context)
        else:
            # Try Claude first, fallback to Groq
            response = await self._generate_claude_response(user_message, context)
            if not response.get("success") and self.groq_client:
                logger.warning("Claude failed, falling back to Groq")
                response = await self._generate_groq_response(user_message, context)
        
        # Add timing information
        response["response_time_ms"] = int((time.time() - start_time) * 1000)
        
        return response
    
    @timer
    async def _generate_claude_response(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using AWS Claude Sonnet 4"""
        
        if not self.claude_client:
            return {
                "success": False,
                "error": "Claude client not available",
                "model_used": "none"
            }
        
        try:
            # Build system prompt
            system_prompt = self._build_system_prompt(context)
            
            # Prepare Claude request
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": LLM_CONFIG["aws_claude"]["max_tokens"],
                "temperature": LLM_CONFIG["aws_claude"]["temperature"],
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
            
            # Make API call
            response = self.claude_client.invoke_model(
                modelId=settings.AWS_BEDROCK_MODEL_ID,
                body=json.dumps(request_body),
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            if response_body.get('content') and len(response_body['content']) > 0:
                ai_response = response_body['content'][0]['text']
                
                return {
                    "success": True,
                    "response": ai_response,
                    "model_used": "aws_claude_sonnet_4",
                    "tokens_used": response_body.get('usage', {}).get('output_tokens', 0)
                }
            else:
                return {
                    "success": False,
                    "error": "Empty response from Claude",
                    "model_used": "aws_claude_sonnet_4"
                }
        
        except ClientError as e:
            error_msg = f"AWS Claude API error: {e}"
            log_error(e, "Claude API call", {"user_message": user_message[:100]})
            return {
                "success": False,
                "error": error_msg,
                "model_used": "aws_claude_sonnet_4"
            }
        
        except Exception as e:
            error_msg = f"Claude service error: {e}"
            log_error(e, "Claude service", {"user_message": user_message[:100]})
            return {
                "success": False,
                "error": error_msg,
                "model_used": "aws_claude_sonnet_4"
            }
    
    @timer
    async def _generate_groq_response(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using Groq (backup/emergency)"""
        
        if not self.groq_client:
            return {
                "success": False,
                "error": "Groq client not available",
                "model_used": "none"
            }
        
        try:
            # Build system prompt
            system_prompt = self._build_system_prompt(context)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Make API call
            response = self.groq_client.chat.completions.create(
                model=LLM_CONFIG["groq_backup"]["model"],
                messages=messages,
                max_tokens=LLM_CONFIG["groq_backup"]["max_tokens"],
                temperature=LLM_CONFIG["groq_backup"]["temperature"],
                timeout=LLM_CONFIG["groq_backup"]["timeout"]
            )
            
            if response.choices and len(response.choices) > 0:
                ai_response = response.choices[0].message.content
                
                return {
                    "success": True,
                    "response": ai_response,
                    "model_used": "groq_llama",
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }
            else:
                return {
                    "success": False,
                    "error": "Empty response from Groq",
                    "model_used": "groq_llama"
                }
        
        except Exception as e:
            error_msg = f"Groq service error: {e}"
            log_error(e, "Groq service", {"user_message": user_message[:100]})
            return {
                "success": False,
                "error": error_msg,
                "model_used": "groq_llama"
            }
    
    def _build_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt for AI models"""
        
        base_prompt = f"""You are SafeIndy Assistant, an AI-powered emergency response and civic assistance system for Indianapolis, Indiana.

CORE MISSION:
- Provide immediate emergency response guidance
- Offer civic assistance and city service information
- Support Indianapolis residents with safety and community resources

EMERGENCY PROTOCOLS:
- If someone describes an active emergency, guide them to call 911 immediately
- Provide clear, concise emergency instructions
- Capture location information when possible
- Stay calm and provide reassuring guidance

INDIANAPOLIS CONTEXT:
- Location: Indianapolis, Marion County, Indiana
- Emergency: 911
- Non-emergency: {settings.INDY_NON_EMERGENCY}
- Coordinates: {settings.INDY_LATITUDE}, {settings.INDY_LONGITUDE}

RESPONSE GUIDELINES:
- Be concise but comprehensive
- Prioritize safety and accuracy
- Use clear, accessible language
- Provide specific Indianapolis resources when relevant
- Include sources when citing information

CAPABILITIES:
- Emergency detection and response
- City services information
- Weather and safety alerts
- Community resource discovery
- Real-time Indianapolis data access"""

        # Add context-specific information
        if context:
            if context.get("emergency_detected"):
                base_prompt += "\n\n🚨 EMERGENCY MODE: Prioritize immediate safety guidance and emergency contacts."
            
            if context.get("user_location"):
                lat, lon = context["user_location"]
                base_prompt += f"\n\nUSER LOCATION: {lat}, {lon} (Use for location-specific guidance)"
            
            if context.get("document_context"):
                base_prompt += f"\n\nDOCUMENT CONTEXT: {context['document_context']}"
            
            if context.get("search_results"):
                base_prompt += f"\n\nCURRENT INFORMATION: {context['search_results'][:500]}"
        
        return base_prompt
    
    async def classify_intent(self, user_message: str) -> Dict[str, Any]:
        """Classify user intent and detect emergencies"""
        
        # Quick emergency keyword detection first
        emergency_check = self._quick_emergency_detection(user_message)
        
        if emergency_check["is_emergency"] and emergency_check["confidence"] > EMERGENCY_CONFIDENCE_THRESHOLD:
            return {
                "intent": "emergency",
                "confidence": emergency_check["confidence"],
                "emergency": True,
                "emergency_type": emergency_check["categories"],
                "keywords": emergency_check["keywords"]
            }
        
        # Use AI for more nuanced classification
        classification_prompt = f"""Classify the following user message into one of these categories:

CATEGORIES:
- emergency: Active emergency requiring immediate 911 response
- urgent: Non-emergency urgent situation (police non-emergency, utilities)
- information: General information request about Indianapolis
- city_services: Request for city services or 311
- weather: Weather-related inquiry
- location: Location or directions request
- general: General conversation

USER MESSAGE: "{user_message}"

Respond with JSON only:
{{
    "intent": "category_name",
    "confidence": 0.0-1.0,
    "emergency": true/false,
    "reasoning": "brief explanation"
}}"""
        
        try:
            # Use Groq for fast classification
            response = await self._generate_groq_response(
                classification_prompt,
                context={"classification_mode": True}
            )
            
            if response.get("success"):
                # Parse JSON response
                import re
                json_match = re.search(r'\{.*\}', response["response"], re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group())
                    
                    # Merge with emergency detection
                    classification.update({
                        "emergency_keywords": emergency_check["keywords"],
                        "model_used": response["model_used"]
                    })
                    
                    return classification
        
        except Exception as e:
            log_error(e, "Intent classification")
        
        # Fallback classification
        return {
            "intent": "general",
            "confidence": 0.5,
            "emergency": emergency_check["is_emergency"],
            "emergency_keywords": emergency_check["keywords"],
            "model_used": "fallback"
        }
    
    def _quick_emergency_detection(self, text: str) -> Dict[str, Any]:
        """Quick emergency keyword detection"""
        text_lower = text.lower()
        
        emergency_patterns = {
            "fire": ["fire", "smoke", "burning", "flames"],
            "medical": ["heart attack", "overdose", "unconscious", "bleeding", "choking", 
                      "can't breathe", "chest pain", "stroke", "seizure", "collapsed"],
            "violence": ["shooting", "stabbing", "assault", "robbery", "break in", 
                        "domestic violence", "attacked", "weapon", "gun"],
            "accident": ["accident", "crash", "collision", "car accident", "hit and run"],
            "utility": ["gas leak", "power line down", "water main break", "explosion"],
            "immediate": ["help", "emergency", "urgent", "911", "call police", "ambulance", "now"]
        }
        
        detected_categories = []
        matched_keywords = []
        total_matches = 0
        
        for category, keywords in emergency_patterns.items():
            category_matches = 0
            for keyword in keywords:
                if keyword in text_lower:
                    if category not in detected_categories:
                        detected_categories.append(category)
                    matched_keywords.append(keyword)
                    category_matches += 1
                    total_matches += 1
        
        # Calculate confidence
        # Base confidence on number of categories + keyword density
        confidence = 0.0
        if detected_categories:
            category_score = min(len(detected_categories) * 0.25, 0.8)
            keyword_density = min(total_matches / max(len(text_lower.split()), 1), 0.5)
            confidence = min(category_score + keyword_density, 1.0)
        
        return {
            "is_emergency": len(detected_categories) > 0,
            "confidence": confidence,
            "categories": detected_categories,
            "keywords": matched_keywords,
            "total_matches": total_matches
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of LLM services"""
        health = {
            "aws_claude": "unknown",
            "groq": "unknown",
            "overall": "unknown"
        }
        
        # Test Claude
        if self.claude_client:
            try:
                test_response = await self._generate_claude_response(
                    "Hello, this is a health check. Please respond with 'OK'.",
                    context={"health_check": True}
                )
                health["aws_claude"] = "healthy" if test_response.get("success") else "error"
            except Exception as e:
                health["aws_claude"] = f"error: {str(e)[:50]}"
        else:
            health["aws_claude"] = "not_configured"
        
        # Test Groq
        if self.groq_client:
            try:
                test_response = await self._generate_groq_response(
                    "Hello, this is a health check. Please respond with 'OK'.",
                    context={"health_check": True}
                )
                health["groq"] = "healthy" if test_response.get("success") else "error"
            except Exception as e:
                health["groq"] = f"error: {str(e)[:50]}"
        else:
            health["groq"] = "not_configured"
        
        # Overall health
        if health["aws_claude"] == "healthy" or health["groq"] == "healthy":
            health["overall"] = "healthy"
        elif "healthy" in [health["aws_claude"], health["groq"]]:
            health["overall"] = "degraded"
        else:
            health["overall"] = "error"
        
        return health


# Global LLM service instance
llm_service = LLMService()