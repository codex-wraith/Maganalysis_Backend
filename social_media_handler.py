from typing import TYPE_CHECKING
import asyncio
import logging
import os
from datetime import datetime, UTC
from telegram.ext import ApplicationBuilder, MessageHandler, filters, CommandHandler

if TYPE_CHECKING:
    from aiagent import CipherAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class SocialMediaHandler:
    def __init__(self, agent: "CipherAgent"):
        self.agent = agent
        self.telegram_bot = None
        self.telegram_running = False
        self._telegram_task = None

    async def initialize_apis(self):
        """Initialize social media API clients"""
        try:
            # Telegram setup
            telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
            if telegram_bot_token:
                try:
                    logger.info("Building Telegram bot application...")
                    application = ApplicationBuilder().token(telegram_bot_token).build()
                    
                    # Store the application first
                    self.telegram_bot = application
                    
                    # Initialize handlers (which includes webhook deletion)
                    await self.initialize_telegram()
                    
                    # Verify bot is running
                    me = await application.bot.get_me()
                    logger.info(f"Telegram Bot @{me.username} initialized successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize Telegram bot: {e}", exc_info=True)
                    self.telegram_bot = None
                    self.telegram_running = False
            else:
                logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
                
        except Exception as e:
            logger.error(f"Error initializing APIs: {e}")
            raise
        
    async def initialize_telegram(self):
        """Initialize Telegram bot handlers (only call this once)"""
        if not self.telegram_bot:
            logger.error("Cannot initialize handlers: Telegram bot not initialized")
            return

        try:
            # Remove webhook using bot instance
            await self.telegram_bot.bot.delete_webhook()
            
            # Get and store bot info
            bot = await self.telegram_bot.bot.get_me()
            self.bot_username = bot.username
            logger.info(f"Initialized bot with username @{self.bot_username}")
            
            # Command handlers
            async def start_command(update, context):
                logger.info("Received /start command")
                await context.bot.send_message(
                    chat_id=update.effective_chat.id, 
                    text="System online and ready to engage. I'm your creative director - here to make things happen. What's on your mind? ✨"
                )

            # Message handler
            async def message_handler(update, context):
                try:
                    logger.info(f"Received message: {update.message.text}")
                    chat_id = str(update.effective_chat.id)
                    user_id = str(update.message.from_user.id)
                    
                    # Get group information
                    chat_type = update.effective_chat.type
                    chat_title = update.effective_chat.title if chat_type != 'private' else None
                    
                    # Get bot info
                    bot = await context.bot.get_me()
                    bot_username = bot.username
                    
                    # Check if we should respond
                    should_respond = True
                    if chat_type != 'private':
                        # In groups, only respond if bot is mentioned
                        mentioned = False
                        # Check for mentions
                        if update.message.entities:
                            for entity in update.message.entities:
                                if entity.type == 'mention':
                                    mention_text = update.message.text[entity.offset:entity.offset + entity.length]
                                    if mention_text.lower() == f"@{bot_username.lower()}":
                                        mentioned = True
                                        break
                        # Also check for direct bot commands
                        elif update.message.text.startswith('/'):
                            mentioned = True
                        
                        should_respond = mentioned
                    
                    if not should_respond:
                        return
                    
                    # Get or create group context
                    group_context = await self.agent.message_memory.get_group_context(chat_id)
                    
                    # Update group context if needed
                    if chat_type != 'private':
                        group_context.update({
                            'chat_id': chat_id,
                            'chat_title': chat_title,
                            'chat_type': chat_type,
                            'last_active': datetime.now().isoformat()
                        })
                        await self.agent.message_memory.set_group_context(chat_id, group_context)
                    
                    # Show typing indicator
                    await context.bot.send_chat_action(
                        chat_id=chat_id, 
                        action="typing"
                    )
                    
                    # Process message with context
                    response = await self.agent.respond(
                        input_text=update.message.text,
                        platform="telegram",
                        user_id=user_id,  # This will be a positive number
                        context={
                            'chat_id': chat_id,  # This will match user_id
                            'chat_type': chat_type,  # Will be 'private'
                            'chat_title': chat_title,  # Will be None
                            'group_context': group_context,  # Will be empty for private chats
                            'bot': context.bot  # Pass the bot instance for direct message sending
                        }
                    )
                    
                    # Check if response length exceeds Telegram's limit (4096 characters)
                    if response and len(response) > 4000:
                        # Split message into multiple chunks of 4000 characters
                        chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                        
                        for i, chunk in enumerate(chunks):
                            # Add counter for multi-part messages
                            if len(chunks) > 1:
                                prefix = f"[Part {i+1}/{len(chunks)}]\n\n" if i > 0 else ""
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=prefix + chunk
                                )
                            else:
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=chunk
                                )
                            
                            # Small delay between chunks to avoid rate limiting
                            if i < len(chunks) - 1:
                                await asyncio.sleep(0.3)
                    else:
                        # Send as a single message if within limits
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=response
                        )
                    
                except Exception as e:
                    logger.error(f"Message handling error: {e}", exc_info=True)
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="An error occurred processing your message. Please try again."
                    )

            # Register handlers
            logger.info("Registering Telegram handlers...")
            self.telegram_bot.add_handler(CommandHandler("start", start_command))
            self.telegram_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
            
            logger.info("Telegram handlers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Telegram handlers: {e}", exc_info=True)
            self.telegram_bot = None
            
    async def start_telegram_polling(self):
        """Start Telegram polling (in the existing event loop)."""
        if not self.telegram_bot:
            logger.error("Cannot start polling: Telegram bot not initialized")
            return
            
        if self.telegram_running:
            logger.info("Telegram polling already running")
            return
        
        try:
            # Initialize the application if not done
            if not self.telegram_bot._initialized:
                await self.telegram_bot.initialize()
            
            # Start the application if not running
            if not self.telegram_bot._running:
                await self.telegram_bot.start()
            
            # Start polling in the current event loop
            self._telegram_task = asyncio.create_task(
                self.telegram_bot.updater.start_polling()
            )
            
            self.telegram_running = True
            logger.info("Telegram polling started successfully")
            
        except Exception as e:
            logger.error(f"Error in Telegram polling: {e}", exc_info=True)
            self.telegram_running = False

    async def stop_telegram(self):
        """Stop Telegram bot gracefully"""
        if self.telegram_bot and self.telegram_running:
            try:
                self.telegram_running = False

                # Cancel any existing polling task (if you had one).
                if self._telegram_task:
                    self._telegram_task.cancel()
                    try:
                        await self._telegram_task
                    except asyncio.CancelledError:
                        pass

                await self.telegram_bot.stop()

                logger.info("Telegram bot stopped gracefully")
            except Exception as e:
                logger.error(f"Error stopping Telegram bot: {e}")
        

    async def monitor_telegram(self):
        """Monitor Telegram bot status and reconnect if needed."""
        try:
            if not self.telegram_bot:
                return
                
            # Check if polling is still active
            if not self.telegram_running:
                logger.warning("Telegram polling stopped, attempting restart...")
                await self.start_telegram_polling()
                
            # Check bot health
            me = await self.telegram_bot.bot.get_me()
            logger.debug(f"Telegram bot status: @{me.username} is active")
                
        except Exception as e:
            logger.error(f"Telegram monitoring error: {e}")
            self.telegram_running = False
            await asyncio.sleep(60)  # Wait before retry