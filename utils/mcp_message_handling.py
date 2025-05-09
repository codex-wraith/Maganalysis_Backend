import logging
import asyncio
from typing import Dict, Any, List, Optional
import re
from utils.message_handling import MessageProcessor

logger = logging.getLogger(__name__)

# Module-level declarations for functions to be registered
async def stream_large_response(content: str, platform: str, max_chunk_size: int = 4096, delay: float = 0.5):
        """
        Stream a large response in chunks.
        REPLACEMENT for message streaming functionality.
        
        Args:
            content: The content to stream
            platform: The platform to stream to (e.g., "telegram", "web")
            max_chunk_size: Maximum chunk size
            delay: Delay between chunks
            
        Returns:
            A success message
        """
        from main import get_app
        app = get_app()
        
        try:
            # Clean the message
            content = MessageProcessor.clean_message(content)
            
            # Split into chunks
            chunks = MCPMessageProcessor.split_message(content, max_chunk_size)
            
            if platform == "telegram":
                # Get the Telegram bot from the app state
                bot = app.state.social_media_handler.telegram_bot
                chat_id = app.state.social_media_handler.current_chat_id
                
                if bot and chat_id:
                    # Send chunks with delay
                    for i, chunk in enumerate(chunks):
                        # Add continuation marker if needed
                        if i < len(chunks) - 1:
                            chunk += " ⏩"
                            
                        await bot.send_message(chat_id=chat_id, text=chunk)
                        
                        # Add delay between chunks
                        if i < len(chunks) - 1:
                            await asyncio.sleep(delay)
                            
                    return {"success": True, "chunks_sent": len(chunks)}
                else:
                    return {"success": False, "error": "No active Telegram chat"}
            else:
                # For web platform, we can't stream directly
                # Return the chunked content for streaming by the web client
                return {
                    "success": True,
                    "chunks": chunks,
                    "delay": delay
                }
        except Exception as e:
            logger.error(f"Error streaming large response: {e}")
            return {"success": False, "error": str(e)}
    
async def format_for_platform(content: str, platform: str = "web"):
        """
        Format content for a specific platform.
        
        Args:
            content: The content to format
            platform: The target platform (web, telegram)
            
        Returns:
            Formatted content
        """
        try:
            if platform == "telegram":
                # Telegram-specific formatting
                formatted = MCPMessageProcessor.format_for_telegram(content)
                return {"formatted_content": formatted, "platform": platform}
            else:
                # Default web formatting
                formatted = MCPMessageProcessor.clean_message(content)
                return {"formatted_content": formatted, "platform": platform}
        except Exception as e:
            logger.error(f"Error formatting content: {e}")
            return {"error": str(e), "original_content": content}

# Create a wrapper around the MessageProcessor for MCP
class MCPMessageProcessor:
    @staticmethod
    def clean_message(text: str) -> str:
        """Clean message text by removing control characters and normalizing newlines."""
        return MessageProcessor.clean_message(text)
        
    @staticmethod
    def split_message(text: str, max_chunk_size: int = 4096) -> List[str]:
        """Split a message into chunks of max_chunk_size."""
        processor = MessageProcessor(max_chunk_size=max_chunk_size)
        return processor.split_message(text)
        
    @staticmethod
    def format_for_telegram(text: str) -> str:
        """Format text specifically for Telegram."""
        cleaned = MessageProcessor.clean_message(text)
        
        # Apply additional Telegram-specific formatting if needed
        # For example, ensure proper markdown formatting
        # This assumes MessageProcessor.clean_message handles most of the necessary cleaning
        
        return cleaned