from typing import Dict, List, Any, Optional
from mcp.server.fastmcp import Context
import logging
import json
import os
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

# Module-level declarations for functions to be registered
async def get_conversation_history(platform: str, user_id: str, limit: int = 10):
        """
        Get conversation history for a user.
        
        Args:
            platform: The platform identifier (e.g., "web", "telegram")
            user_id: The user identifier
            limit: Maximum number of messages to include
            
        Returns:
            The conversation history formatted for UI display
        """
        try:
            # Get conversation history from memory manager
            history = await agent.message_memory.get_conversation_history(
                platform=platform,
                user_id=user_id,
                limit=limit
            )
            
            # Format for UI display
            formatted_history = []
            for msg in history:
                formatted_history.append({
                    "role": "assistant" if msg.get('is_response') else "user",
                    "content": msg.get('text', ''),
                    "timestamp": msg.get('timestamp', '')
                })
            
            return {
                "success": True,
                "history": formatted_history,
                "count": len(formatted_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "history": []
            }
    
async def add_memory_conversation_to_context(context: Any, platform: str, user_id: str, limit: int = 10):
        """
        Add conversation history to an MCP context.
        
        Args:
            context: The MCP Context object
            platform: The platform identifier
            user_id: The user identifier
            limit: Maximum number of messages to include
            
        Returns:
            Success status
        """
        try:
            if not isinstance(context, Context):
                return {
                    "success": False,
                    "error": "Invalid context object"
                }
            
            # Get conversation history
            history = await agent.message_memory.get_conversation_history(
                platform=platform,
                user_id=user_id,
                limit=limit
            )
            
            # Sort by timestamp to ensure correct order
            history.sort(key=lambda x: x.get('timestamp', 0))
            
            # Add to context
            message_count = 0
            for msg in history:
                if msg.get('is_response'):
                    context.add_assistant_message(msg.get('text', ''))
                else:
                    context.add_user_message(msg.get('text', ''))
                message_count += 1
            
            return {
                "success": True,
                "messages_added": message_count
            }
            
        except Exception as e:
            logger.error(f"Error adding conversation to context: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "messages_added": 0
            }
    
async def clear_user_memory(platform: str, user_id: str):
        """
        Clear a user's conversation history.
        
        Args:
            platform: The platform identifier
            user_id: The user identifier
            
        Returns:
            Success status
        """
        try:
            # Get Redis keys for this user
            patterns = [
                f"*:{platform}:{user_id}:*",  # Standard message patterns
                f"*:short_term:{platform}:{user_id}"  # Short-term cache
            ]
            
            deleted_count = 0
            for pattern in patterns:
                keys = await agent.message_memory.redis.keys(pattern)
                for key in keys:
                    await agent.message_memory.redis.delete(key)
                    deleted_count += 1
            
            # Clear short-term cache
            cache_key = f"{platform}:{user_id}"
            if cache_key in agent.message_memory._short_term_cache:
                del agent.message_memory._short_term_cache[cache_key]
            
            return {
                "success": True,
                "deleted_keys": deleted_count,
                "platform": platform,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error clearing user memory: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }