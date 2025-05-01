import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

def register_memory_resources(mcp, agent):
    """Register memory resources with the MCP server"""
    
    @mcp.resource("memory/stats")
    async def get_memory_stats():
        """Get memory usage statistics"""
        try:
            stats = await agent.message_memory.get_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}", exc_info=True)
            return {"error": str(e)}
    
    @mcp.resource("memory/user/{user_id}")
    async def get_user_memory(user_id: str):
        """Get memory information for a specific user"""
        try:
            # Get platforms where this user has activity
            platform_patterns = ["web", "telegram"]
            user_data = {}
            
            for platform in platform_patterns:
                history = await agent.message_memory.get_conversation_history(
                    platform=platform,
                    user_id=user_id,
                    limit=50  # Higher limit for admin purposes
                )
                
                if history:
                    user_data[platform] = {
                        "message_count": len(history),
                        "last_active": max(msg.get('timestamp', '0') for msg in history)
                    }
            
            if not user_data:
                return {"user_id": user_id, "found": False}
            
            return {
                "user_id": user_id,
                "found": True,
                "platforms": user_data
            }
            
        except Exception as e:
            logger.error(f"Error getting user memory: {e}", exc_info=True)
            return {"error": str(e)}