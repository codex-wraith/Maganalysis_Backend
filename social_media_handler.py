import logging
import asyncio
import os
from mcp_server import mcp
from mcp.server.fastmcp import Context
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SocialMediaHandler:
    """
    MCP-integrated social media handler.
    This is a complete replacement of the previous implementation.
    """
    
    def __init__(self, agent):
        """Initialize the social media handler with the agent reference"""
        self.agent = agent
        self.telegram_bot = None
        self.current_chat_id = None
        logger.info("Initialized MCP-integrated SocialMediaHandler")
    
    async def initialize_apis(self):
        """Initialize social media APIs"""
        await self._initialize_telegram()
    
    async def _initialize_telegram(self):
        """Initialize Telegram bot"""
        try:
            # Check if Telegram token exists
            telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
            if not telegram_token:
                logger.warning("TELEGRAM_BOT_TOKEN not found, Telegram integration disabled")
                return
                
            # Import the telegram library
            from telegram import Update, Bot
            from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler
            
            # Initialize the bot
            self.telegram_bot = Bot(token=telegram_token)
            
            # Create application
            application = Application.builder().token(telegram_token).build()
            
            # Add handlers
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_telegram_message))
            application.add_handler(CommandHandler("start", self.handle_start_command))
            
            # Start polling
            self._telegram_application = application
            logger.info("Telegram bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram: {e}")
            self.telegram_bot = None
    
    async def start_telegram_polling(self):
        """Start Telegram polling if a bot is configured"""
        if hasattr(self, '_telegram_application'):
            await self._telegram_application.initialize()
            await self._telegram_application.start_polling()
            logger.info("Telegram polling started")
    
    async def stop_telegram(self):
        """Stop Telegram polling"""
        if hasattr(self, '_telegram_application'):
            await self._telegram_application.stop()
            logger.info("Telegram polling stopped")
    
    async def handle_start_command(self, update, context):
        """Handle /start command in Telegram"""
        user_id = str(update.message.from_user.id)
        chat_id = str(update.message.chat_id)
        
        # Set current chat for potential streaming responses
        self.current_chat_id = chat_id
        
        # Send greeting message
        await context.bot.send_message(
            chat_id=chat_id,
            text="Hello! I'm Cipher, your AI market analyst. How can I help you analyze assets today?"
        )
    
    async def handle_telegram_message(self, update, context):
        """
        Handle incoming Telegram messages using only MCP functionality.
        All previous implementation code is replaced.
        """
        try:
            message = update.message
            chat_id = str(message.chat_id)
            user_id = str(message.from_user.id)
            
            # Set current chat context for potential streaming responses
            self.current_chat_id = chat_id
            
            # Skip messages without text
            if not message.text:
                return
                
            # Get chat type (private or group)
            chat_type = message.chat.type
            is_private = chat_type == "private"
            chat_title = message.chat.title if not is_private else None
            
            # Check if bot should respond (always in private, only when mentioned in groups)
            should_respond = is_private
            if not is_private:
                # In groups, only respond when mentioned
                should_respond = message.text.startswith(f"@{context.bot.username}") or \
                                "@" + context.bot.username in message.text
                
            if not should_respond:
                return
                
            # Remove bot mention from text in groups
            text = message.text
            if not is_private and should_respond:
                text = text.replace(f"@{context.bot.username}", "").strip()
                
            # Use the agent's respond method
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            
            response = await self.agent.respond(
                text, 
                platform="telegram",
                user_id=user_id,
                context={
                    'chat_id': chat_id,
                    'chat_type': chat_type,
                    'chat_title': chat_title
                }
            )
            
            # Split response if needed for Telegram
            from utils.mcp_message_handling import MCPMessageProcessor
            chunks = MCPMessageProcessor.split_message(response)
            
            # Send response
            for i, chunk in enumerate(chunks):
                # Add continuation marker if needed
                if i < len(chunks) - 1:
                    chunk += " ⏩"
                    
                await context.bot.send_message(chat_id=chat_id, text=chunk)
                
                # Add delay between chunks
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error handling Telegram message: {e}", exc_info=True)
            # Try to send an error message
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="I encountered an error processing your request. Please try again later."
                )
            except:
                pass