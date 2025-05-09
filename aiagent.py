import os
import json
import logging
import asyncio
import re
import pandas as pd
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional, Tuple, Union

# Import MCP components
from mcp_server import mcp
# Removed import of Context which is no longer needed
from aisettings import AISettings
from memory.message_memory import MessageMemoryManager
from prompts.prompt_manager import PromptManager
from market.market_manager import MarketManager
from market.indicators import TechnicalIndicators
from market.price_levels import PriceLevelAnalyzer, LevelType
from utils.message_handling import MessageProcessor

# Import the tools directly from their module files
from market.mcp_tools import get_technical_indicators, get_news_sentiment, get_raw_market_data
from market.mcp_price_levels import get_price_levels
from market.mcp_multi_timeframe import analyze_multi_timeframe

logger = logging.getLogger(__name__)

class CipherAgent:
    """
    MCP-based implementation of the Maganalysis AI agent.
    This is a complete replacement of the previous implementation,
    designed for production use from day one.
    """
    
    def __init__(self, settings: AISettings = None):
        """Initialize the CipherAgent with MCP integration"""
        # Load settings
        self.settings = settings or AISettings()
        
        # Load character configuration
        self.character = self._load_character_config()
        
        # Initialize components
        self.prompt_manager = PromptManager()
        self.message_memory = MessageMemoryManager(self.settings, self.character)
        self.market_manager = None  # Will be set from main.py
        
        # Format and cache the base system prompt
        self.base_system_prompt = self._format_base_prompt()
        
        # Define timeframe hierarchy for market analysis
        self.TIMEFRAME_HIERARCHY = {
            "1min": ["5min", "15min", "60min", "daily"],
            "5min": ["15min", "30min", "60min"],  
            "15min": ["30min", "60min"],
            "30min": ["60min", 'daily'],
            "60min": ["daily"],
            "daily": ["weekly", "monthly"],
            "weekly": ["monthly"],
            "monthly": []
        }
        
        # Initialize caching for expensive operations
        self._mtf_cache = {}  # Cache for multi-timeframe analysis
        self._price_level_cache = {}  # Cache for price level analysis
        self._market_data_cache = {}  # Cache for market data
        
        # Initialize API clients with better error handling
        try:
            from anthropic import AsyncAnthropic
            self.anthropic_client = AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
            
            # Initialize Tavily client with error handling if API key exists
            tavily_api_key = os.environ.get("TAVILY_API_KEY")
            if tavily_api_key:
                from tavily import AsyncTavilyClient
                self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
            else:
                self.tavily_client = None
                logger.warning("Tavily API key not found, search functionality will be limited")
                
        except Exception as e:
            logger.error(f"Error initializing API clients: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize CipherAgent API clients: {e}")
        
        # Register tools and resources with the MCP server
        self._register_all_mcp_components()
        
        logger.info("CipherAgent initialization complete with MCP integration")
    
    def _register_all_mcp_components(self):
        """Register all tools, resources, and prompts with MCP"""
        # Register tools
        self._register_market_tools()
        self._register_memory_tools()
        self._register_messaging_tools()
        
        # Register resources
        self._register_market_resources()
        self._register_memory_resources()
    
    def _register_market_tools(self):
        """Register market analysis tools with MCP"""
        from main import app
        http = app.state.http
        market_manager = app.state.market_manager
        
        # Import and register market tools
        from market.mcp_tools import register_market_tools
        register_market_tools(mcp, app, market_manager, http, self)
        
        # Register multi-timeframe tools
        from market.mcp_multi_timeframe import register_multi_timeframe_tools
        register_multi_timeframe_tools(mcp, app, market_manager, http, self)
        
        # Register price level tools
        from market.mcp_price_levels import register_price_level_tools
        register_price_level_tools(mcp, app, market_manager, http, self)
    
    def _register_memory_tools(self):
        """Register memory management tools with MCP"""
        from memory.mcp_tools import register_memory_tools
        register_memory_tools(mcp, self)
    
    def _register_messaging_tools(self):
        """Register message processing tools with MCP"""
        from utils.mcp_message_handling import register_messaging_tools
        register_messaging_tools(mcp, self)
    
    def _register_market_resources(self):
        """Register market data resources with MCP"""
        # MCP resources are registered directly in mcp_server.py
        pass
    
    def _register_memory_resources(self):
        """Register memory resources with MCP"""
        from memory.mcp_resources import register_memory_resources
        register_memory_resources(mcp, self)
    
    async def _format_conversation_history(self, conversation_history: list) -> str:
        """Format conversation history with proper breaks and timestamps."""
        if not conversation_history:
            return ""
        historical_context = "\nRecent Messages:\n"
        last_timestamp = None
        for msg in conversation_history:
            if msg.get('is_response') and "Top Movers:" in msg.get('text', ''):
                continue
            if isinstance(msg, dict):
                current_timestamp = msg.get('timestamp')
                if isinstance(current_timestamp, str):
                    try:
                        current_timestamp = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00')).timestamp()
                    except:
                        current_timestamp = 0
                if last_timestamp and (float(current_timestamp) - float(last_timestamp) > 300):
                    historical_context += "\n--- New Session ---\n\n"
                last_timestamp = current_timestamp
                if msg.get('is_response'):
                    historical_context += f"Assistant: {msg['text']}\n\n"
                else:
                    historical_context += f"User: {msg['text']}\n\n"
        return historical_context

    async def respond(self, text, platform=None, user_id=None, context=None):
        """
        Main method to process user input and generate a response.
        
        This implementation uses MCP exclusively - no fallback to old implementation.
        
        Args:
            text: The user's message
            platform: The platform the message is from (e.g., "telegram", "web")
            user_id: The unique identifier for the user
            context: Additional context for the request
            
        Returns:
            The AI response
        """
        try:
            # Process user message for storage
            message_data = {
                "text": text,
                "platform": platform or "web",
                "user_id": user_id or "anonymous",
                "timestamp": datetime.now(UTC).isoformat(),
                "context": context or {}
            }
            
            # Store message in memory
            await self.message_memory.add(message_data)
            
            # Add to short-term cache for quick retrieval
            await self.message_memory.add_to_short_term(
                platform=platform or "web",
                user_id=user_id or "anonymous",
                message=message_data
            )
            
            # Check if user is admin
            is_admin = await self._is_admin_user(platform, user_id)
            
            # Handle platform restrictions
            if platform == "telegram" and not is_admin and self.settings.TELEGRAM_RESTRICT_TO_ADMIN:
                restricted_message = "I'm sorry, I'm currently in maintenance mode and only responding to admin users."
                await self.message_memory.add_response(platform, user_id, restricted_message)
                return restricted_message
            
            # --- Prepare System Prompt ---
            system_prompts_content = []  # List to hold all system instruction strings

            # Add base system prompt content
            system_prompts_content.append(self.base_system_prompt)

            # Add platform-specific context content
            platform_key = platform or "web"
            try:
                platform_context = self.prompt_manager.get_system_prompt(platform_key)
                if platform_context:
                    system_prompts_content.append(platform_context)
                else:
                    logger.warning(f"No system prompt template found for platform key: {platform_key}")
            except Exception as e:
                logger.warning(f"Could not get platform context for {platform_key} using PromptManager: {e}", exc_info=True)
            
            # Get conversation history (needed for both system prompt and messages list)
            conversation_history = await self.message_memory.get_conversation_history(
                platform=platform or "web", 
                user_id=user_id or "anonymous",
                limit=15  # Include more history for better context
            )
            
            # Add conversation break content (as system instruction)
            if conversation_history:
                last_message_time = conversation_history[-1].get('timestamp')
                if last_message_time:
                    # Parse timestamp
                    if isinstance(last_message_time, str):
                        try:
                            last_dt = datetime.fromisoformat(last_message_time.replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            last_dt = None
                    else:
                        last_dt = None
                        
                    # Add session break if it's been more than 3 hours
                    if last_dt and (datetime.now(UTC) - last_dt).total_seconds() > 10800:
                        try:
                            # Use prompt_manager to get conversation break prompt
                            session_break = self.prompt_manager.get_conversation_break("new_session")
                            if session_break:
                                system_prompts_content.append(session_break)
                        except Exception as e:
                            logger.warning(f"Could not get conversation break prompt: {e}", exc_info=True)
            
            # Add formatted conversation history (as system instruction)
            formatted_history = await self._format_conversation_history(conversation_history)
            if formatted_history:
                system_prompts_content.append(formatted_history)

            # --- START: Dynamic Data Injection for Specific Commands ---
            lower_text = text.lower()

            # 1. Handle Search Queries (Tavily)
            search_query = ""
            search_type = None
            search_depth = "basic"  # Default search depth
            time_filter = "week"    # Default time filter
            max_results = 5         # Default max results
            search_context_label = "Search Results"

            if "news:" in lower_text:
                search_query = lower_text.split("news:", 1)[1].strip()
                search_type = "news"
                time_filter = "day"
                max_results = 8
                search_context_label = "News Search Results"
            elif "deepsearch:" in lower_text:
                search_query = lower_text.split("deepsearch:", 1)[1].strip()
                search_type = "general"
                search_depth = "advanced"
                time_filter = "day"
                search_context_label = "Deep Research Results"
            elif "search:" in lower_text:
                search_query = lower_text.split("search:", 1)[1].strip()
                search_type = "general"
                search_context_label = "Search Results"

            if search_query and self.tavily_client:
                logger.debug(f"Triggering Tavily {search_type or 'general'} query: '{search_query}' (depth: {search_depth}, timeframe: {time_filter})")
                try:
                    search_params = {
                        "query": search_query,
                        "search_depth": search_depth,
                        "max_results": max_results,
                        "include_answer": True, # Get summarized answer
                    }

                    # Handle search type and time filter
                    if search_type == "news":
                        search_params["topic"] = "news"

                    # Add time filtering based on the time_filter value
                    # Convert logical time filters to specific date ranges
                    from datetime import datetime, timedelta, UTC

                    now = datetime.now(UTC)

                    if time_filter == "day":
                        # Last 24 hours
                        yesterday = now - timedelta(days=1)
                        search_params["start_published_date"] = yesterday.strftime("%Y-%m-%dT%H:%M:%SZ")
                    elif time_filter == "week":
                        # Last 7 days
                        last_week = now - timedelta(days=7)
                        search_params["start_published_date"] = last_week.strftime("%Y-%m-%dT%H:%M:%SZ")
                    elif time_filter == "month":
                        # Last 30 days
                        last_month = now - timedelta(days=30)
                        search_params["start_published_date"] = last_month.strftime("%Y-%m-%dT%H:%M:%SZ")

                    # Set end date to now
                    search_params["end_published_date"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")

                    logger.debug(f"Using Tavily search parameters: {search_params}")
                    search_response = await self.tavily_client.search(**search_params)

                    search_context_data = ""
                    if search_response and "answer" in search_response:
                        search_context_data += search_response["answer"] + "\n\n"
                    if search_response and "results" in search_response and search_response["results"]:
                        search_context_data += f"{search_context_label.replace('Results', 'Sources')}:\n"
                        for idx, result in enumerate(search_response["results"], 1):
                            title = result.get("title", "Untitled")
                            url = result.get("url", "No URL")
                            snippet = result.get("snippet", "No snippet available")
                            # Similar date formatting as in non-MCP:
                            DATE_PATTERN = r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
                            date_match = re.search(DATE_PATTERN, f"{title} {snippet}", re.IGNORECASE)
                            date_str = f" ({date_match.group(0)})" if date_match else ""
                            recency_marker = ""
                            if date_match and date_match.group(0) and self.is_recent_date(date_match.group(0), max_days=3):
                                recency_marker = "🆕 "
                            search_context_data += f"{idx}. {recency_marker}{title}{date_str}\n   URL: {url}\n   Summary: {snippet}\n\n"

                    if search_context_data:
                        system_prompts_content.append(f"\n\n{search_context_label}:\n{search_context_data}")
                    else:
                        system_prompts_content.append(f"\n\nNo relevant {search_type or 'general'} results found for '{search_query}'.")
                except Exception as e:
                    logger.error(f"Error during Tavily {search_type or 'general'} search: {e}")
                    system_prompts_content.append(f"\n\nError performing {search_type or 'general'} search: {str(e)}")

            # 2. Handle "intraday" command
            elif "intraday" in lower_text:
                ticker = self._extract_ticker(text)
                interval = self._extract_interval(text) or "5min"

                formatted_data_str = f"Could not retrieve intraday data for {ticker}."
                if ticker:
                    try:
                        is_crypto_req = "crypto" in lower_text or self._determine_asset_type(ticker) == "crypto"

                        if is_crypto_req:
                            raw_intraday_data = await self.market_manager.get_crypto_intraday(symbol=ticker.replace("CRYPTO:", ""), market="USD", interval=interval)
                            pm_format_config = self.prompt_manager.get_intraday_formatting(is_crypto=True, ticker=ticker, interval=interval)

                            if raw_intraday_data and "Time Series Crypto" in str(raw_intraday_data):
                                # Use proper formatting with prompt manager configuration
                                time_series_key = next((k for k in raw_intraday_data.keys() if k.startswith("Time Series") or k == "data"), None)

                                if time_series_key and time_series_key in raw_intraday_data:
                                    time_series_data = raw_intraday_data[time_series_key]
                                    # Create a clean, well-structured format using prompt manager config
                                    formatted_data_str = ""

                                    # Use intro from prompt manager with proper ticker and interval
                                    intro = pm_format_config.get("intro", "").format(
                                        ticker=ticker.upper(),
                                        interval=interval
                                    )
                                    formatted_data_str += f"{intro}\n\n"

                                    # Add header with proper interval
                                    header = pm_format_config.get("header", "").format(
                                        interval=interval
                                    )
                                    formatted_data_str += f"{header}\n\n"

                                    # Add intro to the data display
                                    vertical_intro = pm_format_config.get("vertical_intro", "")
                                    formatted_data_str += f"{vertical_intro}\n\n"

                                    # Get most recent entries (up to 5)
                                    recent_times = sorted(time_series_data.keys(), reverse=True)[:5]

                                    for timestamp in recent_times:
                                        entry = time_series_data[timestamp]
                                        # Map keys based on common Alpha Vantage response format
                                        open_price = entry.get("1. open", entry.get("open", "N/A"))
                                        high_price = entry.get("2. high", entry.get("high", "N/A"))
                                        low_price = entry.get("3. low", entry.get("low", "N/A"))
                                        close_price = entry.get("4. close", entry.get("close", "N/A"))
                                        volume = entry.get("5. volume", entry.get("volume", "N/A"))

                                        # Format timestamp for better readability
                                        try:
                                            # Try to parse and reformat the timestamp
                                            from datetime import datetime
                                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                            display_time = dt.strftime("%Y-%m-%d %H:%M")
                                        except:
                                            # If parsing fails, use the original timestamp
                                            display_time = timestamp

                                        # Format using prompt manager template with better timestamp display
                                        formatted_entry = pm_format_config.get("vertical_record_format", "").format(
                                            time=display_time,
                                            open=self._format_price(open_price),
                                            high=self._format_price(high_price),
                                            low=self._format_price(low_price),
                                            close=self._format_price(close_price),
                                            volume=self._format_volume(volume)
                                        )
                                        formatted_data_str += f"{formatted_entry}\n\n"

                                    # Add footer to complete the presentation
                                    footer = pm_format_config.get("footer", "")
                                    formatted_data_str += footer
                                else:
                                    formatted_data_str = f"Crypto Intraday Data for {ticker} ({interval}) could not be formatted."
                            else:
                                formatted_data_str = f"No crypto intraday data found for {ticker}."
                        else: # Stock
                            raw_intraday_data = await self.market_manager.get_intraday_data(symbol=ticker, interval=interval, outputsize="compact")
                            pm_format_config = self.prompt_manager.get_intraday_formatting(is_crypto=False, ticker=ticker, interval=interval)

                            if raw_intraday_data and any(k for k in raw_intraday_data.keys() if k.startswith("Time Series")):
                                # Use proper formatting with prompt manager configuration
                                time_series_key = next((k for k in raw_intraday_data.keys() if k.startswith("Time Series")), None)

                                if time_series_key and time_series_key in raw_intraday_data:
                                    time_series_data = raw_intraday_data[time_series_key]
                                    # Create a clean, well-structured format using prompt manager config
                                    formatted_data_str = ""

                                    # Use intro from prompt manager with proper ticker and interval
                                    intro = pm_format_config.get("intro", "").format(
                                        ticker=ticker.upper(),
                                        interval=interval
                                    )
                                    formatted_data_str += f"{intro}\n\n"

                                    # Add header with proper interval
                                    header = pm_format_config.get("header", "").format(
                                        interval=interval
                                    )
                                    formatted_data_str += f"{header}\n\n"

                                    # Add intro to the data display
                                    vertical_intro = pm_format_config.get("vertical_intro", "")
                                    formatted_data_str += f"{vertical_intro}\n\n"

                                    # Get most recent entries (up to 5)
                                    recent_times = sorted(time_series_data.keys(), reverse=True)[:5]

                                    for timestamp in recent_times:
                                        entry = time_series_data[timestamp]
                                        # Map keys based on common Alpha Vantage response format
                                        open_price = entry.get("1. open", entry.get("open", "N/A"))
                                        high_price = entry.get("2. high", entry.get("high", "N/A"))
                                        low_price = entry.get("3. low", entry.get("low", "N/A"))
                                        close_price = entry.get("4. close", entry.get("close", "N/A"))
                                        volume = entry.get("5. volume", entry.get("volume", "N/A"))

                                        # Format timestamp for better readability
                                        try:
                                            # Try to parse and reformat the timestamp
                                            from datetime import datetime
                                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                            display_time = dt.strftime("%Y-%m-%d %H:%M")
                                        except:
                                            # If parsing fails, use the original timestamp
                                            display_time = timestamp

                                        # Format using prompt manager template with better timestamp display
                                        formatted_entry = pm_format_config.get("vertical_record_format", "").format(
                                            time=display_time,
                                            open=self._format_price(open_price),
                                            high=self._format_price(high_price),
                                            low=self._format_price(low_price),
                                            close=self._format_price(close_price),
                                            volume=self._format_volume(volume)
                                        )
                                        formatted_data_str += f"{formatted_entry}\n\n"

                                    # Add footer to complete the presentation
                                    footer = pm_format_config.get("footer", "")
                                    formatted_data_str += footer
                                else:
                                    formatted_data_str = f"Stock Intraday Data for {ticker} ({interval}) could not be formatted."
                            else:
                                formatted_data_str = f"No stock intraday data found for {ticker}."

                        system_prompts_content.append(f"\n\n```\n{formatted_data_str}\n```")
                    except Exception as e:
                        logger.error(f"Error fetching/formatting intraday data for {ticker}: {e}")
                        system_prompts_content.append(f"\n\nError retrieving intraday data for {ticker}.")
                else: # No ticker found for intraday
                    system_prompts_content.append("\n\nPlease provide a valid ticker symbol for intraday analysis.")

            # 3. Handle "top movers" command
            elif "top movers" in lower_text:
                try:
                    top_data = await self.market_manager.get_top_gainers_losers()
                    top_format_config = self.prompt_manager.get_top_movers_formatting()

                    if top_data.get('gainers') or top_data.get('most_actively_traded'):
                        display_count = 15
                        top_gainers = top_data.get("gainers", [])[:display_count]
                        top_active = top_data.get("most_actively_traded", [])[:display_count]

                        gainers_formatted_lines = [
                            top_format_config["line_format"].format(
                                emoji=top_format_config["gainers_emoji"],
                                ticker=str(item.get('ticker', 'N/A')).strip(),
                                price=self._format_price(item.get('price', 'N/A')),
                                change_percentage=self._sanitize_numeric_field(item.get('change_percentage', 'N/A'))
                            ) for item in top_gainers
                        ]
                        gainers_formatted = "\n".join(gainers_formatted_lines) or top_format_config["empty_message"]

                        active_formatted_lines = [
                            top_format_config["line_format"].format(
                                emoji=top_format_config["active_emoji"],
                                ticker=str(item.get('ticker', 'N/A')).strip(),
                                price=self._format_price(item.get('price', 'N/A')),
                                change_percentage=self._sanitize_numeric_field(item.get('change_percentage', 'N/A'))
                            ) for item in top_active
                        ]
                        active_formatted = "\n".join(active_formatted_lines) or top_format_config["empty_message"]

                        stock_context = (
                            f"{top_format_config.get('instructions', '')}\n\n"
                            f"IMPORTANT: DO NOT USE ANY MARKDOWN FORMATTING CHARACTERS (NO #, ##, *, _).\n\n"
                            f"{top_format_config.get('wrapper_start', '')}\n"
                            f"{top_format_config.get('title', 'THE MARKET WATCH - DAILY MOVERS')}\n\n"
                            f"{top_format_config.get('gainers_header', 'Top Gainers:')}\n"
                            f"{gainers_formatted}\n\n"
                            f"{top_format_config.get('active_header', 'Most Actively Traded:')}\n"
                            f"{active_formatted}\n"
                            f"{top_format_config.get('wrapper_end', '')}\n\n"
                            f"Format exactly like this example (no markdown):\n{top_format_config.get('example_format', '')}"
                        )
                        system_prompts_content.append(f"\n\n{stock_context}")
                    else:
                        system_prompts_content.append(f"\n\n{top_format_config.get('empty_message', 'No top movers data available.')}")
                except Exception as e:
                    logger.error(f"Error fetching/formatting top movers: {e}")
                    system_prompts_content.append("\n\nError retrieving top movers data.")

            # 4. Handle "current price" command
            elif "current price" in lower_text:
                # Pattern to find "current price of SYMBOL" or "current price SYMBOL"
                match = re.search(r"current price\s*(?:from|of)?\s*([A-Za-z0-9\-]+)", lower_text)
                from_currency_symbol = ""
                if match:
                    from_currency_symbol = self._extract_ticker(match.group(1))
                else: # Fallback if "of" or "from" is not used
                    tokens = text.split()
                    if "price" in tokens:
                        price_idx = tokens.index("price")
                        if price_idx + 1 < len(tokens):
                            from_currency_symbol = self._extract_ticker(tokens[price_idx+1])

                if from_currency_symbol:
                    try:
                        exchange_data = await self.market_manager.get_exchange_rate(from_currency=from_currency_symbol, to_currency="USD")
                        rate_info = exchange_data.get("Realtime Currency Exchange Rate", {})
                        from_name = rate_info.get("2. From_Currency Name", from_currency_symbol)
                        exchange_rate_val = rate_info.get("5. Exchange Rate", "N/A")
                        formatted_rate = self._format_price(exchange_rate_val) if exchange_rate_val != "N/A" else "N/A"
                        price_message = f"Current Exchange Rate: {from_name} is {formatted_rate} USD."
                        system_prompts_content.append(f"\n\n{price_message}")
                    except Exception as e:
                        logger.error(f"Error fetching current price for {from_currency_symbol}: {e}")
                        system_prompts_content.append(f"\n\nError retrieving current price for {from_currency_symbol}.")
                else:
                    system_prompts_content.append("\n\nPlease specify a currency symbol to get its current price (e.g., 'current price BTC').")

            # If this is an analysis request but not handled by any of the special commands above,
            # let it be routed to the dedicated analyze_asset method instead of adding market analysis here
            # --- END: Dynamic Data Injection ---

            # Combine all system instructions into a single string
            final_system_prompt = "\n\n".join(system_prompts_content)
            
            # --- Prepare Messages List (User/Assistant roles ONLY) ---
            messages_for_llm = []
            for msg in conversation_history:
                role = "assistant" if msg.get('is_response') else "user"
                content = msg.get('text', '')
                # Skip system messages that might have been stored incorrectly before
                if content and role in ["user", "assistant"]:
                    messages_for_llm.append({"role": role, "content": content})
            
            # Add the current message
            messages_for_llm.append({"role": "user", "content": text})
            
            # Determine model to use
            model_to_use = self.settings.SOCIAL_MODEL
            
            # Allow forcing specific model from context
            if context and context.get("force_claude"):
                model_to_use = self.settings.SOCIAL_MODEL
            
            # Set max tokens with context override if provided
            max_tokens = context.get("max_tokens", self.settings.MAX_TOKENS)
            
            # Generate response using the initialized Anthropic client with correct structure
            try:
                response = await self.anthropic_client.messages.create(
                    messages=messages_for_llm,    # ONLY user/assistant messages
                    system=final_system_prompt,   # Pass system instructions here
                    model=model_to_use,
                    max_tokens=max_tokens,
                    temperature=self.settings.TEMPERATURE
                )
            except Exception as api_error:
                # Log the specific API error
                logger.error(f"Anthropic API call failed: {api_error}", exc_info=True)
                # Re-raise to be caught by the outer try-except
                raise api_error
            
            # Extract and process the response text based on Anthropic's response structure
            response_text = ""
            if hasattr(response, 'content') and response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        response_text += block.text
            else:
                logger.warning("Received empty content from Anthropic API")
                response_text = "Sorry, I encountered an issue generating the response."
            
            # Clean and format response
            cleaned_response = MessageProcessor.clean_message(response_text)
            
            # Store in memory
            await self.message_memory.add_response(
                platform=platform or "web",
                user_id=user_id or "anonymous",
                response=cleaned_response
            )
            
            return cleaned_response
            
        except Exception as e:
            error_id = f"err-{os.urandom(4).hex()}"
            
            # Check if it's a specific Anthropic API error about unexpected role
            if isinstance(e, Exception) and "Unexpected role \"system\"" in str(e):
                logger.error(f"Error ID {error_id} in respond (Anthropic Message Format Error): {e}", exc_info=True)
                error_response = f"I encountered an issue with the message format (Error ID: {error_id}). Please try again."
            else:
                logger.error(f"Error ID {error_id} in respond: {e}", exc_info=True)
                error_response = f"I encountered a technical issue (Error ID: {error_id}). Please try again in a moment."
            
            # Store error in memory
            await self.message_memory.add_response(
                platform=platform or "web",
                user_id=user_id or "anonymous",
                response=error_response
            )
            
            return error_response
            
    async def enhance_with_sentiment(self, symbol: str, signal_base: str, reasoning: str):
        """
        Enhances a trade signal with news sentiment data.
        
        Parameters:
            symbol (str): The ticker symbol to analyze
            signal_base (str): The initial technical signal
            reasoning (str): The initial reasoning for the signal
            
        Returns:
            tuple: (updated_signal, updated_reasoning, sentiment_data)
            where sentiment_data contains sentiment_score, sentiment_label, news_count,
            and most_relevant_article (if found)
        """
        sentiment_score = 0
        sentiment_label = "NEUTRAL"
        news_count = 0
        most_relevant_article = None
        article_highlight = ""
        
        try:
            # Use direct function call to get news sentiment
            news_data = await get_news_sentiment(symbol=symbol)
            
            # Process sentiment data
            sentiment_score = news_data.get("sentiment", {}).get("score", 50)
            sentiment_label = news_data.get("sentiment", {}).get("label", "NEUTRAL")
            news_count = news_data.get("news_count", 0)
            most_relevant_article = news_data.get("most_relevant_article")
            
            # Format article highlight if available
            if most_relevant_article:
                article_title = most_relevant_article.get("title", "")
                article_summary = most_relevant_article.get("summary", "")
                published_time = most_relevant_article.get("published", "")
                
                # Format time if available
                formatted_time = ""
                if published_time and len(published_time) >= 8:
                    try:
                        # Add recency check
                        is_recent = self.is_recent_date(published_time, max_days=1)
                        is_old = not self.is_recent_date(published_time, max_days=90)
                        
                        if is_recent:
                            formatted_time = "🆕 Recently published"
                        elif is_old:
                            formatted_time = "📅 Older article"
                        else:
                            formatted_time = "Published recently"
                    except Exception as e:
                        logger.warning(f"Error formatting publication time: {e}")
                
                # Create article highlight
                article_highlight = f"📑 NEWS: \"{article_title}\" ({formatted_time})\n" + \
                                    f"Summary: {article_summary}\n"
            
            # Add sentiment to the reasoning
            updated_reasoning = reasoning
            updated_signal = signal_base
            
            # For strong buy/strong sell signals, news sentiment can confirm the signal
            if sentiment_score >= 65 and signal_base in ["BUY", "ACCUMULATE", "HOLD/LONG"]:
                updated_reasoning += f", confirmed by bullish news sentiment ({sentiment_score:.2f})"
            elif sentiment_score <= 35 and signal_base in ["SELL", "REDUCE", "AVOID/SHORT"]:
                updated_reasoning += f", confirmed by bearish news sentiment ({sentiment_score:.2f})"
                
            # For neutral technical signals, news sentiment can provide direction
            elif sentiment_score >= 65 and signal_base == "NEUTRAL":
                updated_signal = "SPECULATIVE BUY"
                updated_reasoning += f", supported by bullish news sentiment ({sentiment_score:.2f})"
            elif sentiment_score <= 35 and signal_base == "NEUTRAL":
                updated_signal = "SPECULATIVE SELL"
                updated_reasoning += f", supported by bearish news sentiment ({sentiment_score:.2f})"
                
            # For conflicting signals (technical vs news), add caution
            elif sentiment_score >= 65 and signal_base in ["SELL", "REDUCE", "AVOID/SHORT"]:
                updated_reasoning += f", but conflicting with bullish news sentiment ({sentiment_score:.2f}) - proceed with caution"
            elif sentiment_score <= 35 and signal_base in ["BUY", "ACCUMULATE", "HOLD/LONG"]:
                updated_reasoning += f", but conflicting with bearish news sentiment ({sentiment_score:.2f}) - proceed with caution"
            
            # Return the updated signal and reasoning with article data
            return updated_signal, updated_reasoning, {
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "news_count": news_count,
                "most_relevant_article": most_relevant_article,
                "article_highlight": article_highlight
            }
        except Exception as e:
            logger.error(f"Error processing sentiment data for {symbol}: {str(e)}")
            
        # Return original signal and reasoning if no sentiment data
        return signal_base, reasoning, {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "news_count": news_count,
            "most_relevant_article": most_relevant_article,
            "article_highlight": article_highlight
        }
    
    async def analyze_asset(self, symbol, asset_type="stock", market="USD", interval="60min", for_user_display=False, platform="web"):
        """
        Analyze an asset and return a comprehensive analysis.
        Implementation using direct data gathering and formatting rather than MCP tools.

        Args:
            symbol: The asset symbol (e.g., AAPL, BTC)
            asset_type: Either "stock" or "crypto"
            market: The market to use for crypto (default "USD")
            interval: The timeframe to analyze
            for_user_display: Whether to format the output for user display
            platform: Target platform for formatting ("web", "telegram")

        Returns:
            A string containing the analysis
        """
        try:
            # --- Direct Data Gathering ---

            # 1. Analyze primary timeframe
            primary_timeframe_data = await self._analyze_timeframe(
                symbol=symbol,
                asset_type=asset_type,
                market=market,
                interval=interval
            )

            # Get current price and other metadata from primary timeframe
            current_price = primary_timeframe_data.get("current_price")
            price_change_pct = primary_timeframe_data.get("price_change_pct", 0)
            change_direction = primary_timeframe_data.get("change_direction", "→")
            latest_atr = primary_timeframe_data.get("indicators", {}).get("atr", {}).get("value")

            # 2. Get multi-timeframe levels and insights
            mtf_data = await self._analyze_multi_timeframe_levels(
                symbol=symbol,
                asset_type=asset_type,
                current_price=current_price,
                main_interval=interval,
                latest_atr=latest_atr
            )

            # 3. Get news sentiment data
            # Initialize variables for sentiment
            signal_base = mtf_data.get("signal", "NEUTRAL") if mtf_data else "NEUTRAL"
            reasoning = mtf_data.get("reasoning", "Insufficient data") if mtf_data else "Insufficient data"

            # Enhance with sentiment analysis
            signal, reasoning, sentiment_data = await self.enhance_with_sentiment(
                symbol=symbol,
                signal_base=signal_base,
                reasoning=reasoning
            )

            # --- Construct technical_indicators_text ---

            # Extract indicators from primary timeframe data
            indicators = primary_timeframe_data.get("indicators", {})
            trend_data = primary_timeframe_data.get("trend", {})

            # Calculate support/resistance proximity
            support_levels = primary_timeframe_data.get("support_levels", [])
            resistance_levels = primary_timeframe_data.get("resistance_levels", [])

            # Get indicator values
            latest_rsi = indicators.get("rsi", {}).get("value")
            rsi_interpretation = "Neutral"
            if latest_rsi is not None:
                if latest_rsi >= 70:
                    rsi_interpretation = "Overbought"
                elif latest_rsi <= 30:
                    rsi_interpretation = "Oversold"
                elif latest_rsi >= 60:
                    rsi_interpretation = "Bullish momentum"
                elif latest_rsi <= 40:
                    rsi_interpretation = "Bearish momentum"

            latest_sma = indicators.get("sma", {}).get("value")
            sma_period = indicators.get("sma", {}).get("period", 50)

            latest_ema = indicators.get("ema", {}).get("value")
            ema_period = indicators.get("ema", {}).get("period", 20)

            # Price vs Moving Averages
            price_vs_ma = "above"
            if latest_sma is not None and current_price < latest_sma:
                price_vs_ma = "below"

            # MACD
            macd_value = indicators.get("macd", {}).get("value")
            signal_value = indicators.get("macd", {}).get("signal")
            macd_trend = "neutral"
            macd_interpretation = "Neutral momentum"
            if macd_value is not None and signal_value is not None:
                if macd_value > signal_value:
                    macd_trend = "bullish"
                    if macd_value > 0:
                        macd_interpretation = "Strong bullish momentum"
                    else:
                        macd_interpretation = "Weakening bearish momentum"
                else:
                    macd_trend = "bearish"
                    if macd_value < 0:
                        macd_interpretation = "Strong bearish momentum"
                    else:
                        macd_interpretation = "Weakening bullish momentum"

            # Bollinger Bands
            bbands_upper = indicators.get("bbands", {}).get("upper")
            bbands_lower = indicators.get("bbands", {}).get("lower")
            bbands_status = "within range"
            if bbands_upper is not None and bbands_lower is not None:
                if current_price > bbands_upper:
                    bbands_status = "above upper band (potentially overbought)"
                elif current_price < bbands_lower:
                    bbands_status = "below lower band (potentially oversold)"

            # ATR
            latest_atr = indicators.get("atr", {}).get("value")
            atr_pct = None
            atr_interpretation = "Average volatility"
            if latest_atr is not None and current_price:
                atr_pct = (latest_atr / current_price) * 100
                if atr_pct > 3:
                    atr_interpretation = "High volatility"
                elif atr_pct < 1:
                    atr_interpretation = "Low volatility"

            # Stochastic
            k_value = indicators.get("stochastic", {}).get("k")
            d_value = indicators.get("stochastic", {}).get("d")
            stoch_interpretation = "Neutral momentum"
            if k_value is not None and d_value is not None:
                if k_value > 80 and d_value > 80:
                    stoch_interpretation = "Strongly overbought"
                elif k_value < 20 and d_value < 20:
                    stoch_interpretation = "Strongly oversold"
                elif k_value > d_value:
                    stoch_interpretation = "Bullish momentum building"
                elif k_value < d_value:
                    stoch_interpretation = "Bearish momentum building"

            # ADX for trend strength
            latest_adx = indicators.get("adx", {}).get("value")
            trend_strength = "moderate"
            if latest_adx is not None:
                if latest_adx > 25:
                    trend_strength = "strong"
                elif latest_adx < 15:
                    trend_strength = "weak"

            # Trend direction
            trend = trend_data.get("direction", "neutral").upper()

            # Volume analysis
            vol_comment = primary_timeframe_data.get("volume_comment", "Volume is at average levels")

            # VWAP
            vwap_value = indicators.get("vwap", {}).get("value")
            is_extended = interval in ["daily", "weekly", "monthly"]

            # Determine signal based on indicator consensus
            indicator_signals = []
            if trend_data.get("direction") == "uptrend":
                indicator_signals.append("BUY")
            elif trend_data.get("direction") == "downtrend":
                indicator_signals.append("SELL")

            if latest_rsi is not None:
                if latest_rsi > 70:
                    indicator_signals.append("OVERBOUGHT")
                elif latest_rsi < 30:
                    indicator_signals.append("OVERSOLD")

            if macd_trend == "bullish":
                indicator_signals.append("BUY")
            elif macd_trend == "bearish":
                indicator_signals.append("SELL")

            # Determine signal based on indicator consensus
            # Start with the multi-timeframe signal as a base
            signal = mtf_data.get("signal", "NEUTRAL") if mtf_data else "NEUTRAL"

            # Count buy/sell indicators
            buy_count = 0
            sell_count = 0

            # Check various indicators
            if trend_data.get("direction") == "uptrend":
                buy_count += 1
            elif trend_data.get("direction") == "downtrend":
                sell_count += 1

            if latest_rsi is not None:
                if latest_rsi > 70:
                    sell_count += 1  # Overbought
                elif latest_rsi < 30:
                    buy_count += 1   # Oversold
                elif latest_rsi > 60:
                    buy_count += 0.5  # Bullish momentum but not extreme
                elif latest_rsi < 40:
                    sell_count += 0.5  # Bearish momentum but not extreme

            if macd_trend == "bullish":
                buy_count += 1
            elif macd_trend == "bearish":
                sell_count += 1

            if bbands_status and "above" in bbands_status:
                sell_count += 0.5
            elif bbands_status and "below" in bbands_status:
                buy_count += 0.5

            if k_value is not None and d_value is not None:
                if k_value > 80 and d_value > 80:
                    sell_count += 0.5
                elif k_value < 20 and d_value < 20:
                    buy_count += 0.5
                elif k_value > d_value:
                    buy_count += 0.3
                elif k_value < d_value:
                    sell_count += 0.3

            # Final decision based on counts and strength
            if buy_count > sell_count + 1:
                if buy_count > 2.5:
                    signal = "STRONG BUY"
                else:
                    signal = "BUY"
            elif sell_count > buy_count + 1:
                if sell_count > 2.5:
                    signal = "STRONG SELL"
                else:
                    signal = "SELL"
            elif buy_count > sell_count:
                signal = "ACCUMULATE"
            elif sell_count > buy_count:
                signal = "REDUCE"
            else:
                signal = "NEUTRAL"

            # Adjust signal with sentiment if needed (keep the existing signal from enhance_with_sentiment)
            logger.debug(f"Calculated signal from indicators: {signal}")

            # The signal variable is already updated through enhance_with_sentiment earlier
            # This indicator-based calculation is a fallback/additional input
            logger.debug(f"Final signal (considering tech indicators and sentiment): {signal}")

            # Format the multi-timeframe summary
            mtf_summary = ""
            if mtf_data:
                mtf_summary = mtf_data.get("summary", "")
            mtf_detail = ""

            # Build technical indicators text
            tech_indicators_list = [
                f"- Current Price: ${current_price:.2f if current_price is not None else 'N/A'} ({change_direction or '-'} {price_change_pct:.2f if price_change_pct is not None else 'N/A'}%)",
                f"- Trend: {trend} with {trend_strength.upper()} momentum (ADX: {latest_adx:.2f if latest_adx is not None else 'N/A'})",
                f"- Moving Averages: Price trading {price_vs_ma} {sma_period}-day SMA (${latest_sma:.2f if latest_sma is not None else 'N/A'})",
                f"- EMA {ema_period}-day: ${latest_ema:.2f if latest_ema is not None else 'N/A'} trending {macd_trend.lower()}"
            ]

            if not is_extended and vwap_value is not None:
                tech_indicators_list.append(f"- VWAP: ${vwap_value:.2f} - Critical volume-weighted price level")

            tech_indicators_list.extend([
                f"- RSI: {latest_rsi:.2f if latest_rsi is not None else 'N/A'} ({rsi_interpretation})",
                f"- MACD: {macd_value:.2f if macd_value is not None else 'N/A'} vs Signal {signal_value:.2f if signal_value is not None else 'N/A'} - {macd_interpretation}",
                f"- Bollinger Bands: Price is {bbands_status}",
                f"- ATR: ${latest_atr:.2f if latest_atr is not None else 'N/A'} ({atr_pct:.2f if atr_pct is not None else 'N/A'}% of price) - {atr_interpretation}",
                f"- Stochastic: {k_value:.2f if k_value is not None else 'N/A'}/{d_value:.2f if d_value is not None else 'N/A'} - {stoch_interpretation}",
                f"- Volume Analysis: {vol_comment}",
                f"- Technical Signal: {signal}"
            ])

            technical_indicators_text = "\n".join(tech_indicators_list)

            if mtf_summary:
                technical_indicators_text += f"\n\n{mtf_summary}"
            if mtf_detail:
                technical_indicators_text += f"\n\n{mtf_detail}"

            # Prepare template data with all necessary fields
            # Ensure all variables used by market_analysis_template in prompts.json are included
            interval_display = interval.upper()

            # Format variables for template
            formatted_price = f"{current_price:,.2f}" if current_price is not None else "N/A"
            formatted_change_pct = f"{abs(price_change_pct):.2f}" if price_change_pct is not None else "N/A"

            # Construct price vs moving average text
            price_vs_ma_text = f"Price trading {price_vs_ma} the {sma_period}-day SMA"

            # Create template data dictionary with every field used in the template
            template_data = {
                "ASSET_TYPE": asset_type.upper(),
                "SYMBOL": symbol.upper(),
                "TIMEFRAME": interval_display,
                "PRICE": formatted_price,
                "CHANGE_DIRECTION": change_direction,
                "CHANGE_PCT": formatted_change_pct,
                "SENTIMENT_LABEL": sentiment_data.get("sentiment_label", "NEUTRAL"),
                "ARTICLE_HIGHLIGHT": sentiment_data.get("article_highlight", "No recent news articles found."),
                "TREND": trend,
                "TREND_STRENGTH": trend_strength.upper(),
                "PRICE_VS_MA": price_vs_ma_text,
                "SMA_PERIOD": str(sma_period),
                "SMA_VALUE": self._format_price(latest_sma) if latest_sma is not None else "N/A",
                "EMA_PERIOD": str(ema_period),
                "EMA_VALUE": self._format_price(latest_ema) if latest_ema is not None else "N/A",
                "RSI_VALUE": f"{latest_rsi:.2f}" if latest_rsi is not None else "N/A",
                "RSI_INTERPRETATION": rsi_interpretation,
                "MACD_VALUE": f"{macd_value:.2f}" if macd_value is not None else "N/A",
                "MACD_SIGNAL": f"{signal_value:.2f}" if signal_value is not None else "N/A",
                "MACD_INTERPRETATION": macd_interpretation,
                "BBANDS_STATUS": bbands_status,
                "ATR_VALUE": self._format_price(latest_atr) if latest_atr is not None else "N/A",
                "ATR_PCT": f"{atr_pct:.2f}" if atr_pct is not None else "N/A",
                "ATR_INTERPRETATION": atr_interpretation,
                "STOCH_K": f"{k_value:.2f}" if k_value is not None else "N/A",
                "STOCH_D": f"{d_value:.2f}" if d_value is not None else "N/A",
                "STOCH_INTERPRETATION": stoch_interpretation,
                "VOLUME_COMMENT": vol_comment,
                "SIGNAL": signal,
                # TRUMP_ANALYSIS will be filled after LLM call
            }

            # --- Simplify LLM system prompt ---

            # Just use base system prompt with simple instruction for plain text
            system_prompt_for_analysis = self.base_system_prompt + "\n\nIMPORTANT: Write in plain text only without markdown formatting."

            # --- Create comprehensive user message for LLM ---

            # Get the market analysis template sections
            market_analysis_template_parts = self.prompt_manager.get_template_section("market_analysis_template", {})

            # Prepare prompt parts for the LLM
            prompt_parts_for_llm = {
                "technical_data": technical_indicators_text,
                "current_price_info": f"CURRENT PRICE: ${template_data['PRICE']}",
                "formatting_guidance": market_analysis_template_parts.get("formatting_guidance", ""),
                "analysis_guidance": market_analysis_template_parts.get("analysis_guidance", ""),
                "timeframe_expectations": market_analysis_template_parts.get("timeframe_expectations", ""),
                "conclusion_instruction": f"6. Ends with a clear conclusion about the {signal} recommendation"
            }

            # Handle news sentiment data
            has_news = sentiment_data.get("article_highlight") and sentiment_data.get("news_count", 0) > 0
            if has_news:
                prompt_parts_for_llm["news_sentiment_info"] = f"NEWS SENTIMENT:\n- News Sentiment: {template_data['SENTIMENT_LABEL']}\n{template_data['ARTICLE_HIGHLIGHT']}"
                prompt_parts_for_llm["response_format_instruction"] = market_analysis_template_parts.get("response_format_with_news", "")
            else:
                prompt_parts_for_llm["response_format_instruction"] = market_analysis_template_parts.get("response_format_without_news", "")

            # Assemble the final prompt string for user message with more detailed instructions
            llm_user_prompt = f"Analyze {symbol} {asset_type.upper()} on the {interval} timeframe:\n\n"

            # Add technical indicators section first (most important)
            llm_user_prompt += f"TECHNICAL INDICATORS:\n{prompt_parts_for_llm['technical_data']}\n\n"

            # Add news sentiment if available
            if has_news:
                llm_user_prompt += f"{prompt_parts_for_llm['news_sentiment_info']}\n\n"

            # Add current price information for emphasis
            llm_user_prompt += f"{prompt_parts_for_llm['current_price_info']}\n\n"

            # Add formatting and analysis guidelines with clearer structure
            llm_user_prompt += (
                f"FORMATTING INSTRUCTIONS:\n{prompt_parts_for_llm['formatting_guidance']}\n\n"
                f"ANALYSIS APPROACH:\n{prompt_parts_for_llm['analysis_guidance']}\n\n"
                f"TARGET PRICE GUIDELINES:\n{prompt_parts_for_llm['timeframe_expectations']}\n\n"
                f"CONCLUSION REQUIREMENT:\n{prompt_parts_for_llm['conclusion_instruction']}\n\n"
                f"FINAL RESPONSE FORMAT:\n{prompt_parts_for_llm['response_format_instruction']}\n\n"
                f"Remember to focus on the most impactful insights and present a cohesive analysis that leads to a clear {signal} recommendation."
            )

            # --- LLM Call ---
            messages_for_llm = [{"role": "user", "content": llm_user_prompt}]

            # Generate analysis using Anthropic client
            try:
                response = await self.anthropic_client.messages.create(
                    messages=messages_for_llm,
                    system=system_prompt_for_analysis,
                    model=self.settings.SOCIAL_MODEL,
                    max_tokens=4000,
                    temperature=0
                )
            except Exception as api_error:
                logger.error(f"Anthropic API call failed in analyze_asset: {api_error}", exc_info=True)
                raise api_error

            # Extract the analysis text from response
            trump_analysis_text = ""
            if hasattr(response, 'content') and response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        trump_analysis_text += block.text
            else:
                logger.warning("Received empty content from Anthropic API for analysis")
                trump_analysis_text = "Sorry, I encountered an issue generating the analysis."

            # Clean up any markdown formatting the LLM might have added
            trump_analysis_text = trump_analysis_text.strip().replace("#", "").replace("*", "")

            # --- Final Output Assembly ---

            # Add the LLM-generated analysis to the template data
            template_data["TRUMP_ANALYSIS"] = trump_analysis_text

            # Use PromptManager to get the fully formatted analysis
            final_formatted_analysis = self.prompt_manager.get_market_analysis_prompt(**template_data)

            # If for_user_display is false, just return the raw trump_analysis_text
            if not for_user_display:
                return trump_analysis_text

            # Process the formatted analysis through MessageProcessor
            # MessageProcessor already imported at the top level

            # Clean the message for standard formatting
            cleaned_analysis = MessageProcessor.clean_message(final_formatted_analysis)

            # Platform-specific formatting if needed
            if platform == "telegram":
                from utils.mcp_message_handling import MCPMessageProcessor
                cleaned_analysis = MCPMessageProcessor.format_for_telegram(cleaned_analysis)

            return cleaned_analysis

        except Exception as e:
            error_id = f"err-{os.urandom(4).hex()}"

            # Check if it's a specific Anthropic API error about unexpected role
            if isinstance(e, Exception) and "Unexpected role \"system\"" in str(e):
                logger.error(f"Error ID {error_id} in analyze_asset (Anthropic Message Format Error): {e}", exc_info=True)
                return f"I encountered an issue with the message format while analyzing {symbol} (Error ID: {error_id}). Please try again."
            else:
                logger.error(f"Error ID {error_id} in analyze_asset: {e}", exc_info=True)
                return f"I encountered an error analyzing {symbol} (Error ID: {error_id}). Please try again later."
    
    # Helper methods
    
    def _load_character_config(self) -> Dict:
        """Load and validate character configuration"""
        try:
            with open('scripts/character.json', 'r') as f:
                config = json.load(f)
            self._validate_character_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading character config: {e}", exc_info=True)
            # Return minimal default config if file can't be loaded
            return {
                "name": "Cipher",
                "bio": "AI market analyst",
                "personality": ["Analytical", "Precise", "Helpful"],
                "formatting": {"Style": "Professional"},
                "chat_style": {"Tone": "Confident"},
                "settings": {
                    "admin": {
                        "telegram_admin_id": os.environ.get("TELEGRAM_ADMIN_ID", ""),
                        "admin_name": "System Admin",
                        "admin_commands": ["analyze", "search", "top movers"]
                    }
                }
            }
            
    def _validate_character_config(self, config: Dict) -> None:
        """Validates the character config has all required fields"""
        try:
            required_keys = ["name", "bio", "personality", "formatting", "chat_style", "settings"]
            for key in required_keys:
                if key not in config:
                    logger.error(f"Character config missing required field: {key}")
                    raise ValueError(f"Character config missing required field: {key}")
            
            # Validate admin settings
            if "admin" not in config.get("settings", {}):
                logger.error("Character config missing admin settings")
                raise ValueError("Character config missing admin settings")
                
            admin_required = ["telegram_admin_id", "admin_name", "admin_commands"]
            for key in admin_required:
                if key not in config.get("settings", {}).get("admin", {}):
                    logger.error(f"Character config missing admin setting: {key}")
                    raise ValueError(f"Character config missing admin setting: {key}")
                    
            # Validate data types for key fields
            if not isinstance(config.get("personality", []), list):
                logger.error(f"Character config 'personality' must be a list")
                raise ValueError("Character config 'personality' must be a list")
                
            if not isinstance(config.get("formatting", {}), dict):
                logger.error(f"Character config 'formatting' must be a dictionary")
                raise ValueError("Character config 'formatting' must be a dictionary")
                
            if not isinstance(config.get("chat_style", {}), dict):
                logger.error(f"Character config 'chat_style' must be a dictionary")
                raise ValueError("Character config 'chat_style' must be a dictionary")
                
            logger.info("Character config validation successful")
            
        except Exception as e:
            logger.error(f"Character config validation failed: {str(e)}")
            raise
    
    def _format_base_prompt(self) -> str:
        """Format base system prompt with character info"""
        try:
            # Get current date
            date_info = self._get_current_datetime_info()
            
            # Format the base prompt
            base_prompt = self.prompt_manager.get_system_prompt('base', 
                name=self.character.get('name', "Cipher"),
                bio=self.character.get('bio', "AI market analyst"),
                personality="\n".join(f"- {trait}" for trait in self.character.get('personality', [])),
                formatting="\n".join(f"- {k}: {v}" for k, v in self.character.get('formatting', {}).items()),
                chat_style="\n".join(f"- {k}: {v}" for k, v in self.character.get('chat_style', {}).items())
            )
            
            # Add current date
            return f"{base_prompt}\n\nCurrent date: {date_info['date_verbose']}"
        except Exception as e:
            logger.error(f"Error formatting base prompt: {e}", exc_info=True)
            # Return minimal prompt if formatting fails
            return "You are Cipher, an AI market analyst. Be helpful, analytical and precise. Today's date is " + \
                   datetime.now(UTC).strftime("%B %d, %Y")
    
    def _get_current_datetime_info(self) -> Dict[str, str]:
        """
        Get current date and time information in various formats.
        Returns a dictionary with formatted date/time information.
        """
        # Get current time in UTC
        now_utc = datetime.now(UTC)
        
        try:
            # Try to import pytz for timezone support
            import pytz
            
            # Get current time in Eastern Time (US market timezone) and UTC
            eastern = pytz.timezone('US/Eastern')
            now_eastern = now_utc.astimezone(eastern)
            
            # Format times in various useful formats
            date_info = {
                "date": now_eastern.strftime("%Y-%m-%d"),
                "date_verbose": now_eastern.strftime("%A, %B %d, %Y"),
                "time_eastern": now_eastern.strftime("%I:%M %p ET"),
                "time_utc": now_utc.strftime("%H:%M UTC"),
                "date_time": now_eastern.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "iso_format": now_utc.isoformat(),
                "day_of_week": now_eastern.strftime("%A"),
                "unix_timestamp": str(int(now_utc.timestamp())),
                "market_status": "CLOSED" if now_eastern.weekday() >= 5 or now_eastern.hour < 9 or now_eastern.hour >= 16 else "OPEN"
            }
            
            return date_info
        except ImportError:
            # Fallback if pytz is not available
            return {
                "date": now_utc.strftime("%Y-%m-%d"),
                "time": now_utc.strftime("%H:%M:%S"),
                "date_verbose": now_utc.strftime("%B %d, %Y"),
                "time_verbose": now_utc.strftime("%I:%M %p UTC"),
                "day_of_week": now_utc.strftime("%A"),
                "timezone": "UTC"
            }
    
    async def _is_admin_user(self, platform, user_id) -> bool:
        """Check if a user is an admin"""
        if not user_id:
            return False
        
        admin_id = self.character.get('settings', {}).get('admin', {}).get('telegram_admin_id', '')
        return str(user_id) == str(admin_id)
    
    def _is_analysis_request(self, text: str) -> bool:
        """Detect if a message is requesting market analysis"""
        analysis_keywords = [
            "analyze", "analysis", "chart", "price of", "technical", "support", "resistance",
            "trend", "bullish", "bearish", "stock", "crypto", "bitcoin", "eth", "btc",
            "trading", "predict", "forecast", "market", "indicator"
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in analysis_keywords)
    
    def _extract_ticker(self, text: str) -> Optional[str]:
        """Extract ticker symbol from text"""
        # Implementation from original method 
        # (This preserves the existing extraction logic)
        # re already imported at the top level
        
        # Common crypto symbols to check for
        common_cryptos = ["BTC", "ETH", "XRP", "LTC", "ADA", "DOGE", "USDT", "USDC", 
                          "DOT", "SOL", "AVAX", "MATIC", "BNB", "LINK"]
        
        # Look for symbols with $ prefix (common for stocks)
        dollar_matches = re.findall(r'\$([A-Za-z]{1,5})', text)
        if dollar_matches:
            return dollar_matches[0].upper()
        
        # Look for crypto symbols
        for crypto in common_cryptos:
            # Check for the symbol surrounded by spaces or punctuation
            pattern = r'(?<![A-Za-z])' + re.escape(crypto) + r'(?![A-Za-z])'
            if re.search(pattern, text, re.IGNORECASE):
                return crypto
        
        # Look for stock patterns
        # Typical stock symbols are 1-5 letters
        words = re.findall(r'\b[A-Za-z]{1,5}\b', text)
        for word in words:
            # Skip common words
            if word.upper() in ["A", "I", "THE", "FOR", "AND", "OR", "OF", "IN", "ON", "AT"]:
                continue
            # If it's all uppercase, likely a ticker
            if word.isupper() and len(word) >= 2:
                return word
            # Or if it's mentioned with "stock" or "price"
            if f"{word} stock" in text.lower() or f"{word} price" in text.lower() or f"price of {word}" in text.lower():
                return word.upper()
        
        # No ticker found
        return None
    
    def _extract_interval(self, text: str) -> Optional[str]:
        """Extract interval/timeframe from text"""
        # Implementation from original method 
        # (This preserves the existing extraction logic)
        text_lower = text.lower()
        
        # Map of phrases to intervals
        interval_mapping = {
            "1min": ["1min", "1 min", "1 minute", "1-minute", "1m chart", "1 min chart", "one minute"],
            "5min": ["5min", "5 min", "5 minute", "5-minute", "5m chart", "5 min chart", "five minute"],
            "15min": ["15min", "15 min", "15 minute", "15-minute", "15m chart", "15 min chart", "fifteen minute"],
            "30min": ["30min", "30 min", "30 minute", "30-minute", "30m chart", "30 min chart", "thirty minute"],
            "60min": ["60min", "60 min", "1 hour", "1h", "hourly", "hour chart", "1 hour chart", "60 minute", "60m"],
            "daily": ["daily", "day chart", "day", "1d", "1 day", "1 day chart", "daily chart"],
            "weekly": ["weekly", "week chart", "week", "1w", "1 week", "1 week chart", "weekly chart"],
            "monthly": ["monthly", "month chart", "month", "1mo", "1 month", "1 month chart", "monthly chart"]
        }
        
        # Check for each interval phrase
        for interval, phrases in interval_mapping.items():
            if any(phrase in text_lower for phrase in phrases):
                return interval
        
        # Default to daily if no match
        return None
    
    def _determine_asset_type(self, symbol: str) -> str:
        """Determine if a symbol is for crypto or stock"""
        # Common crypto symbols
        common_cryptos = ["BTC", "ETH", "XRP", "LTC", "ADA", "DOGE", "USDT", "USDC", 
                          "DOT", "SOL", "AVAX", "MATIC", "BNB", "LINK"]
        
        if symbol.upper() in common_cryptos:
            return "crypto"
        else:
            return "stock"
            
    async def _fetch_real_time_price(self, symbol: str, asset_type: str, fallback_price=None):
        """
        Fetch real-time price for a symbol using MCP tools.
        
        Parameters:
            symbol: The asset symbol
            asset_type: Either "stock" or "crypto"
            fallback_price: Fallback price to use if real-time price can't be fetched
            
        Returns:
            tuple: (price, timestamp)
        """
        try:
            # Get technical data which includes current price using direct function call
            tech_data = await get_technical_indicators(symbol=symbol, asset_type=asset_type, timeframe="60min")
            
            # Extract current price and format timestamp
            current_price = tech_data.get("price_data", {}).get("current", fallback_price)
            current_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            
            return current_price, current_time
        except Exception as e:
            logger.error(f"Error fetching real-time price for {symbol}: {e}")
            return fallback_price, datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            
    async def _analyze_timeframe(self, symbol, asset_type, market, interval):
        """
        Analyze a single timeframe for a given asset.
        Directly uses MarketManager methods and TechnicalIndicators calculations
        without relying on MCP tools.

        Parameters:
            symbol: The asset symbol to analyze
            asset_type: Either "stock" or "crypto"
            market: The market for crypto assets (e.g., "USD")
            interval: The timeframe to analyze

        Returns:
            dict: Analysis results for the timeframe
        """
        try:
            # Get raw market data directly from MarketManager based on asset type and timeframe
            raw_market_data = None
            processed_df = None
            time_series_key = None

            # 1. Fetch appropriate market data directly from MarketManager
            if asset_type.lower() == "crypto":
                if interval in ["daily", "weekly", "monthly"]:
                    # Use appropriate crypto timeframe method
                    if interval == "daily":
                        raw_market_data = await self.market_manager.get_crypto_daily(
                            symbol=symbol, market=market
                        )
                    elif interval == "weekly":
                        raw_market_data = await self.market_manager.get_crypto_weekly(
                            symbol=symbol, market=market
                        )
                    elif interval == "monthly":
                        raw_market_data = await self.market_manager.get_crypto_monthly(
                            symbol=symbol, market=market
                        )
                else:
                    # Intraday data for crypto
                    raw_market_data = await self.market_manager.get_crypto_intraday(
                        symbol=symbol, market=market, interval=interval
                    )
            else:  # Stock data
                if interval in ["daily", "weekly", "monthly"]:
                    # Use appropriate stock timeframe method
                    if interval == "daily":
                        raw_market_data = await self.market_manager.get_time_series_daily(
                            symbol=symbol
                        )
                    elif interval == "weekly":
                        raw_market_data = await self.market_manager.get_time_series_weekly(
                            symbol=symbol
                        )
                    elif interval == "monthly":
                        raw_market_data = await self.market_manager.get_time_series_monthly(
                            symbol=symbol
                        )
                else:
                    # Intraday data for stocks
                    raw_market_data = await self.market_manager.get_intraday_data(
                        symbol=symbol, interval=interval, outputsize="compact"
                    )

            # 2. Extract time series key and process data
            if raw_market_data:
                # Find the time series key based on common Alpha Vantage response format
                for key in raw_market_data.keys():
                    if key.startswith("Time Series") or key.startswith("Digital Currency") or key == "data":
                        time_series_key = key
                        break

                if time_series_key and time_series_key in raw_market_data:
                    time_series_data = raw_market_data[time_series_key]

                    # Process the time series data into a dataframe for calculations
                    import pandas as pd

                    # Convert to DataFrame (simple version)
                    df = pd.DataFrame.from_dict(time_series_data, orient='index')

                    # Rename columns to standardized format if needed
                    col_mapping = {
                        '1. open': 'open',
                        '2. high': 'high',
                        '3. low': 'low',
                        '4. close': 'close',
                        '5. volume': 'volume',
                        '1a. open (USD)': 'open',
                        '2a. high (USD)': 'high',
                        '3a. low (USD)': 'low',
                        '4a. close (USD)': 'close',
                        '5. volume': 'volume'
                    }

                    # Apply column renaming
                    df = df.rename(columns={col: new_col for col, new_col in col_mapping.items() if col in df.columns})

                    # Ensure numeric columns
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Sort by date (descending)
                    df = df.sort_index(ascending=False)

                    # Store processed dataframe
                    processed_df = df

                    # 3. Calculate technical indicators directly
                    # TechnicalIndicators already imported at the top level

                    # Calculate indicators using the TechnicalIndicators class
                    indicators = {}

                    # Calculate SMA
                    sma_period = 50
                    sma = TechnicalIndicators.calculate_sma(df, period=sma_period)
                    indicators["sma"] = {
                        "value": float(sma.iloc[0]) if isinstance(sma, pd.Series) and not sma.empty and not pd.isna(sma.iloc[0]) else sma if isinstance(sma, (int, float)) else None,
                        "period": sma_period
                    }

                    # Calculate EMA
                    ema_period = 20
                    ema = TechnicalIndicators.calculate_ema(df, period=ema_period)
                    indicators["ema"] = {
                        "value": float(ema.iloc[0]) if isinstance(ema, pd.Series) and not ema.empty and not pd.isna(ema.iloc[0]) else ema if isinstance(ema, (int, float)) else None,
                        "period": ema_period
                    }

                    # Calculate RSI
                    rsi = TechnicalIndicators.calculate_rsi(df)
                    indicators["rsi"] = {
                        "value": float(rsi.iloc[0]) if isinstance(rsi, pd.Series) and not rsi.empty and not pd.isna(rsi.iloc[0]) else rsi if isinstance(rsi, (int, float)) else None,
                        "period": 14  # Standard period
                    }

                    # Calculate MACD
                    macd, signal, _ = TechnicalIndicators.calculate_macd(df)
                    indicators["macd"] = {
                        "value": float(macd.iloc[0]) if isinstance(macd, pd.Series) and not macd.empty and not pd.isna(macd.iloc[0]) else macd if isinstance(macd, (int, float)) else None,
                        "signal": float(signal.iloc[0]) if isinstance(signal, pd.Series) and not signal.empty and not pd.isna(signal.iloc[0]) else signal if isinstance(signal, (int, float)) else None,
                        "histogram": float(macd.iloc[0] - signal.iloc[0]) if isinstance(macd, pd.Series) and isinstance(signal, pd.Series) and not macd.empty and not signal.empty and not pd.isna(macd.iloc[0]) and not pd.isna(signal.iloc[0]) else None
                    }

                    # Calculate Bollinger Bands
                    upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df)
                    indicators["bbands"] = {
                        "upper": float(upper.iloc[0]) if isinstance(upper, pd.Series) and not upper.empty and not pd.isna(upper.iloc[0]) else upper if isinstance(upper, (int, float)) else None,
                        "middle": float(middle.iloc[0]) if isinstance(middle, pd.Series) and not middle.empty and not pd.isna(middle.iloc[0]) else middle if isinstance(middle, (int, float)) else None,
                        "lower": float(lower.iloc[0]) if isinstance(lower, pd.Series) and not lower.empty and not pd.isna(lower.iloc[0]) else lower if isinstance(lower, (int, float)) else None
                    }

                    # Calculate ATR
                    atr = TechnicalIndicators.calculate_atr(df)
                    indicators["atr"] = {
                        "value": float(atr.iloc[0]) if isinstance(atr, pd.Series) and not atr.empty and not pd.isna(atr.iloc[0]) else atr if isinstance(atr, (int, float)) else None,
                        "period": 14  # Standard period
                    }

                    # Calculate Stochastic Oscillator
                    k, d = TechnicalIndicators.calculate_stochastic(df)
                    indicators["stochastic"] = {
                        "k": float(k.iloc[0]) if isinstance(k, pd.Series) and not k.empty and not pd.isna(k.iloc[0]) else k if isinstance(k, (int, float)) else None,
                        "d": float(d.iloc[0]) if isinstance(d, pd.Series) and not d.empty and not pd.isna(d.iloc[0]) else d if isinstance(d, (int, float)) else None
                    }

                    # Calculate ADX
                    adx = TechnicalIndicators.calculate_adx(df)
                    indicators["adx"] = {
                        "value": float(adx.iloc[0]) if isinstance(adx, pd.Series) and not adx.empty and not pd.isna(adx.iloc[0]) else adx if isinstance(adx, (int, float)) else None
                    }

                    # Calculate VWAP if it's an intraday timeframe
                    is_extended = interval in ["daily", "weekly", "monthly"]
                    if not is_extended:
                        vwap = TechnicalIndicators.calculate_vwap(df)
                        indicators["vwap"] = {
                            "value": float(vwap.iloc[0]) if isinstance(vwap, pd.Series) and not vwap.empty and not pd.isna(vwap.iloc[0]) else vwap if isinstance(vwap, (int, float)) else None
                        }

                    # 4. Generate price data
                    price_data = {}
                    if not df.empty and 'close' in df.columns:
                        current_price = float(df['close'].iloc[0])
                        previous_price = float(df['close'].iloc[1]) if len(df) > 1 else current_price
                        price_change = current_price - previous_price
                        price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0

                        price_data = {
                            "current": current_price,
                            "open": float(df['open'].iloc[0]) if 'open' in df.columns else None,
                            "high": float(df['high'].iloc[0]) if 'high' in df.columns else None,
                            "low": float(df['low'].iloc[0]) if 'low' in df.columns else None,
                            "change": price_change,
                            "change_percent": price_change_pct,
                            "timestamp": df.index[0]
                        }

                    # 5. Determine trend
                    trend_direction = "neutral"
                    trend_strength = "weak"
                    macd_trend = "neutral"

                    # Use SMA and EMA for trend direction
                    if indicators["sma"]["value"] is not None and indicators["ema"]["value"] is not None and current_price:
                        if current_price > indicators["sma"]["value"] and current_price > indicators["ema"]["value"]:
                            trend_direction = "uptrend"
                        elif current_price < indicators["sma"]["value"] and current_price < indicators["ema"]["value"]:
                            trend_direction = "downtrend"

                    # Use ADX for trend strength
                    if indicators["adx"]["value"] is not None:
                        adx_value = indicators["adx"]["value"]
                        if adx_value > 25:
                            trend_strength = "strong"
                        elif adx_value < 15:
                            trend_strength = "weak"
                        else:
                            trend_strength = "moderate"

                    # Use MACD for trend momentum
                    if indicators["macd"]["value"] is not None and indicators["macd"]["signal"] is not None:
                        if indicators["macd"]["value"] > indicators["macd"]["signal"]:
                            macd_trend = "bullish"
                        else:
                            macd_trend = "bearish"

                    trend_data = {
                        "direction": trend_direction,
                        "strength": trend_strength,
                        "macd_trend": macd_trend
                    }

                    # 6. Calculate support and resistance levels
                    # PriceLevelAnalyzer and LevelType already imported at the top level

                    price_level_analyzer = PriceLevelAnalyzer()

                    # Calculate support and resistance for this timeframe
                    support_levels, resistance_levels = [], []
                    if not df.empty and len(df) > 10:  # Need enough data points
                        # Get raw levels
                        raw_levels = price_level_analyzer.identify_key_levels(df, current_price)

                        # Format for response
                        for level in raw_levels:
                            if level["type"] == LevelType.SUPPORT and level["price"] < current_price:
                                support_levels.append({
                                    "price": level["price"],
                                    "strength": level["strength"],
                                    "distance": abs(current_price - level["price"]),
                                    "distance_percent": abs(current_price - level["price"]) / current_price * 100
                                })
                            elif level["type"] == LevelType.RESISTANCE and level["price"] > current_price:
                                resistance_levels.append({
                                    "price": level["price"],
                                    "strength": level["strength"],
                                    "distance": abs(current_price - level["price"]),
                                    "distance_percent": abs(current_price - level["price"]) / current_price * 100
                                })

                        # Sort by price (ascending for support, descending for resistance)
                        support_levels = sorted(support_levels, key=lambda x: x["price"], reverse=True)
                        resistance_levels = sorted(resistance_levels, key=lambda x: x["price"])

                    # Calculate volume comment if we have enough data
                    volume_comment = "Volume data unavailable"
                    if processed_df is not None and 'volume' in processed_df.columns and len(processed_df) >= 10:
                        volume_comment = self.get_volume_comment(processed_df, interval)

                    # Return comprehensive analysis result with all calculated data
                    return {
                        "interval": interval,
                        "current_price": price_data.get("current"),
                        "latest_open": price_data.get("open"),
                        "price_change_pct": price_data.get("change_percent", 0),
                        "change_direction": "↑" if price_data.get("change_percent", 0) > 0 else "↓" if price_data.get("change_percent", 0) < 0 else "→",
                        "most_recent_date": price_data.get("timestamp"),
                        "indicators": indicators,
                        "trend": trend_data,
                        "support_levels": support_levels,
                        "resistance_levels": resistance_levels,
                        "is_extended": is_extended,
                        "processed_df": processed_df,
                        "volume_comment": volume_comment
                    }

            # Fallback if data processing fails
            logger.warning(f"Could not process market data for {symbol} ({interval})")
            return {
                "interval": interval,
                "current_price": None,
                "latest_open": None,
                "price_change_pct": 0,
                "change_direction": "→",
                "most_recent_date": None,
                "indicators": {},
                "trend": {"direction": "neutral", "strength": "weak", "macd_trend": "neutral"},
                "support_levels": [],
                "resistance_levels": [],
                "is_extended": interval in ["daily", "weekly", "monthly"],
                "processed_df": None,
                "volume_comment": "Volume data unavailable"
            }
        except Exception as e:
            logger.error(f"Error analyzing timeframe {interval} for {symbol}: {e}")
            return {
                "interval": interval,
                "current_price": None,
                "latest_open": None,
                "price_change_pct": 0,
                "change_direction": "→",
                "most_recent_date": None,
                "indicators": {},
                "trend": {"direction": "neutral", "strength": "weak", "macd_trend": "neutral"},
                "support_levels": [],
                "resistance_levels": [],
                "is_extended": interval in ["daily", "weekly", "monthly"],
                "processed_df": None,
                "volume_comment": "Volume data unavailable"
            }
    
    async def _fetch_market_data(self, symbol, asset_type, interval):
        """
        Helper function to fetch market data based on asset type and interval.
        
        Parameters:
            symbol: The asset symbol to analyze
            asset_type: Either "stock" or "crypto"
            interval: The timeframe to analyze
            
        Returns:
            tuple: (market_data, time_series_key)
        """
        try:
            # Get raw market data using direct function call
            market_data = await get_raw_market_data(symbol=symbol, asset_type=asset_type, timeframe=interval)
            
            # Get time series key
            time_series_key = None
            for key in market_data.keys():
                if key.startswith("Time Series") or key == "data":
                    time_series_key = key
                    break
            
            return market_data, time_series_key
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol} on {interval}: {e}")
            return None, None
    
    async def _analyze_multi_timeframe_levels(self, symbol, asset_type, current_price, main_interval, latest_atr=None):
        """
        Analyze price levels across multiple timeframes to find key support and resistance zones.
        Directly uses MarketManager methods and PriceLevelAnalyzer without relying on MCP tools.

        Parameters:
            symbol: The asset symbol to analyze
            asset_type: Either "stock" or "crypto"
            current_price: The current price of the asset
            main_interval: The primary timeframe for analysis
            latest_atr: The ATR value for the main timeframe (if available)

        Returns:
            dict: Multi-timeframe level analysis
        """
        try:
            # Initialize cache key for this analysis
            cache_key = f"mtf_levels_{symbol}_{asset_type}_{main_interval}"

            # Check if we have a cached result that's still valid (within 5 minutes)
            if hasattr(self, '_mtf_cache') and cache_key in self._mtf_cache:
                cached_data = self._mtf_cache[cache_key]
                cache_age = datetime.now(UTC) - cached_data.get('timestamp', datetime.min.replace(tzinfo=UTC))

                # Use cached data if it's less than 5 minutes old
                if cache_age.total_seconds() < 300:  # 5 minutes in seconds
                    logger.debug(f"Using cached multi-timeframe data for {symbol} ({main_interval})")
                    return cached_data

            market = "USD"  # Default market for crypto

            # Get timeframe hierarchy for the main interval
            higher_timeframes = self.TIMEFRAME_HIERARCHY.get(main_interval, [])

            # Initialize collections for multi-timeframe analysis
            all_levels = []
            timeframe_levels = {}

            # Collect price levels from each timeframe
            # First, analyze the main timeframe
            main_timeframe_data = await self._analyze_timeframe(symbol, asset_type, market, main_interval)
            if main_timeframe_data:
                main_support = main_timeframe_data.get("support_levels", [])
                main_resistance = main_timeframe_data.get("resistance_levels", [])

                # Add all main timeframe levels with their source
                for level in main_support:
                    level_info = {
                        "price": level["price"],
                        "strength": level["strength"],
                        "type": "support",
                        "timeframes": [main_interval]
                    }
                    all_levels.append(level_info)
                    timeframe_levels[f"{level['price']:.2f}"] = level_info

                for level in main_resistance:
                    level_info = {
                        "price": level["price"],
                        "strength": level["strength"],
                        "type": "resistance",
                        "timeframes": [main_interval]
                    }
                    all_levels.append(level_info)
                    timeframe_levels[f"{level['price']:.2f}"] = level_info

            # Then analyze higher timeframes
            for higher_tf in higher_timeframes:
                # Get data for this timeframe
                tf_data = await self._analyze_timeframe(symbol, asset_type, market, higher_tf)
                if tf_data:
                    tf_support = tf_data.get("support_levels", [])
                    tf_resistance = tf_data.get("resistance_levels", [])

                    # Process support levels from this timeframe
                    for level in tf_support:
                        price_key = f"{level['price']:.2f}"
                        if price_key in timeframe_levels:
                            # Update existing level
                            existing = timeframe_levels[price_key]
                            existing["strength"] += level["strength"] * 0.8  # Higher timeframes get 80% weight
                            if higher_tf not in existing["timeframes"]:
                                existing["timeframes"].append(higher_tf)
                        else:
                            # Add new level
                            level_info = {
                                "price": level["price"],
                                "strength": level["strength"] * 0.8,  # Higher timeframes get 80% weight
                                "type": "support",
                                "timeframes": [higher_tf]
                            }
                            all_levels.append(level_info)
                            timeframe_levels[price_key] = level_info

                    # Process resistance levels from this timeframe
                    for level in tf_resistance:
                        price_key = f"{level['price']:.2f}"
                        if price_key in timeframe_levels:
                            # Update existing level
                            existing = timeframe_levels[price_key]
                            existing["strength"] += level["strength"] * 0.8  # Higher timeframes get 80% weight
                            if higher_tf not in existing["timeframes"]:
                                existing["timeframes"].append(higher_tf)
                        else:
                            # Add new level
                            level_info = {
                                "price": level["price"],
                                "strength": level["strength"] * 0.8,  # Higher timeframes get 80% weight
                                "type": "resistance",
                                "timeframes": [higher_tf]
                            }
                            all_levels.append(level_info)
                            timeframe_levels[price_key] = level_info

            # Consolidate similar price levels using clustering
            # PriceLevelAnalyzer already imported at the top level

            price_level_analyzer = PriceLevelAnalyzer()

            # Use the price level analyzer to consolidate the levels
            consolidated_levels = price_level_analyzer.consolidate_multi_timeframe_levels(all_levels, current_price, latest_atr)

            # Separate into support and resistance
            support_zones = [level for level in consolidated_levels if level["type"] == "support" and level["price"] < current_price]
            resistance_zones = [level for level in consolidated_levels if level["type"] == "resistance" and level["price"] > current_price]

            # Sort by proximity to current price
            support_zones = sorted(support_zones, key=lambda x: abs(current_price - x["price"]))
            resistance_zones = sorted(resistance_zones, key=lambda x: abs(current_price - x["price"]))

            # Create summary text
            summary = "MULTI-TIMEFRAME LEVEL ANALYSIS:\n"

            # Add key support levels
            summary += "\nKey Support Levels (across timeframes):\n"
            for zone in support_zones[:3]:  # Top 3 support zones
                if "price" in zone and zone["price"] < current_price:
                    # Format timeframes for display
                    timeframes_data = zone.get("timeframes", [])
                    if isinstance(timeframes_data, list):
                        timeframes = "/".join(timeframes_data)
                    elif isinstance(timeframes_data, str):
                        timeframes = timeframes_data
                    else:
                        timeframes = str(timeframes_data)

                    strength = zone.get('strength', 0)
                    summary += f"- ${zone['price']:.2f} (strength: {strength:.1f}, timeframes: {timeframes})\n"

            # Add key resistance levels
            summary += "\nKey Resistance Levels (across timeframes):\n"
            for zone in resistance_zones[:3]:  # Top 3 resistance zones
                if "price" in zone and zone["price"] > current_price:
                    # Format timeframes for display
                    timeframes_data = zone.get("timeframes", [])
                    if isinstance(timeframes_data, list):
                        timeframes = "/".join(timeframes_data)
                    elif isinstance(timeframes_data, str):
                        timeframes = timeframes_data
                    else:
                        timeframes = str(timeframes_data)

                    strength = zone.get('strength', 0)
                    summary += f"- ${zone['price']:.2f} (strength: {strength:.1f}, timeframes: {timeframes})\n"

            # Determine overall signal based on price position relative to levels
            signal = "NEUTRAL"
            reasoning = "Price is in a neutral zone with no clear bias."

            # Check if price is near strong support
            if support_zones and abs(current_price - support_zones[0]["price"]) / current_price < 0.02:
                signal = "BUY"
                reasoning = f"Price is close to strong support at ${support_zones[0]['price']:.2f}."
            # Check if price is near strong resistance
            elif resistance_zones and abs(current_price - resistance_zones[0]["price"]) / current_price < 0.02:
                signal = "SELL"
                reasoning = f"Price is close to strong resistance at ${resistance_zones[0]['price']:.2f}."
            # Check if price has a lot of room to nearby resistance
            elif resistance_zones and (resistance_zones[0]["price"] - current_price) / current_price > 0.05:
                signal = "BUY"
                reasoning = f"Price has significant room to run before hitting resistance at ${resistance_zones[0]['price']:.2f}."
            # Check if price has recently broken above resistance
            elif main_timeframe_data and main_timeframe_data.get("trend", {}).get("direction") == "uptrend":
                signal = "BUY"
                reasoning = "Price is in an uptrend across multiple timeframes."
            # Check if price has recently broken below support
            elif main_timeframe_data and main_timeframe_data.get("trend", {}).get("direction") == "downtrend":
                signal = "SELL"
                reasoning = "Price is in a downtrend across multiple timeframes."

            # Prepare result with current timestamp
            result = {
                "support_zones": support_zones,
                "resistance_zones": resistance_zones,
                "summary": summary,
                "signal": signal,
                "reasoning": reasoning,
                "timestamp": datetime.now(UTC)
            }

            # Store in cache for future requests
            if hasattr(self, '_mtf_cache'):
                self._mtf_cache[cache_key] = result

            # Format timestamp for return value
            result["timestamp"] = result["timestamp"].isoformat()
            return result
        except Exception as e:
            logger.error(f"Error in _analyze_multi_timeframe_levels for {symbol}: {e}")
            return None
            
    def is_recent_date(self, date_str, max_days=7):
        """
        Check if a date string is within max_days of current date.
        Useful for filtering news articles or search results.
        
        Args:
            date_str (str): Date string in various formats
            max_days (int): Maximum age in days to consider "recent"
            
        Returns:
            bool: True if date is within max_days of current date
        """
        try:
            # Try parsing different date formats
            try:
                # Try ISO format first
                article_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except ValueError:
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%b %d %Y", "%d %b %Y", "%B %d, %Y"]:
                    try:
                        article_date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If we get here, no format worked
                    return False
                    
            # Calculate time difference using system time
            now = datetime.now()
            delta = now - article_date
            return delta.days <= max_days
        except (ValueError, TypeError):
            return False
    
    def _format_price(self, value: Any) -> str:
        """
        Format a price value into a user-friendly string with two decimals and commas.
        """
        try:
            return f"{float(value):,.2f}"
        except Exception:
            return str(value)

    def _format_volume(self, value: Any) -> str:
        """
        Format a volume value into a comma-separated integer string.
        """
        try:
            return f"{int(float(value)):,}"
        except Exception:
            return str(value)
            
    def _sanitize_numeric_field(self, value: Any) -> str:
        """
        Sanitize a numeric field by stripping out any characters that are not digits,
        a decimal point, or a minus sign.
        
        Parameters:
            value (Any): The value to sanitize
            
        Returns:
            str: Sanitized numeric string
        """
        # Return empty string if None to avoid type errors
        if value is None:
            return ""
            
        # Convert to string and remove whitespace
        v_str = str(value).strip()
        
        # Handle percentage values
        is_percentage = v_str.endswith('%')
        if is_percentage:
            v_str = v_str.rstrip('%')
        
        # Clean the string of non-numeric characters while preserving decimal point and minus sign
        # Using a direct pattern match first for efficiency
        if re.fullmatch(r'[-+]?\d*\.?\d+', v_str):
            return v_str + ('%' if is_percentage else '')
        
        # If the string doesn't fully match a numeric pattern, filter out invalid characters
        cleaned = re.sub(r'[^\d\.\-]', '', v_str)
        
        # Add back percentage sign if it was present
        if is_percentage:
            cleaned += '%'
            
        return cleaned
            
    def format_indicator_value(self, value):
        """
        Helper function to format indicator values with proper handling of None values and conversion errors.
        Returns formatted string with dollar sign and two decimal places for valid numerical values.
        Returns "N/A" for None values or values that cannot be converted to float.
        
        Note: Uses DataFormatter for consistent formatting across the application.
        """
        from market.formatters import DataFormatter
        return DataFormatter.format_price(value)
        
    def get_volume_comment(self, df, interval: str) -> str:
        """
        Compute a volume comment based on the intraday volume data and the analysis timeframe.

        For a 5-minute timeframe, a shorter moving average window (last 30 candles)
        and tighter threshold values are used. For longer timeframes, the average is taken
        over all available data.

        Parameters:
            df: DataFrame containing intraday data.
            interval: The timeframe of the data (e.g., "5min", "15min", etc.).

        Returns:
            str: A comment describing the volume relative to recent averages.
        """
        import pandas as pd
        
        # Use "5. volume" if it exists, otherwise fallback to "volume" if already renamed
        vol_col = "5. volume" if "5. volume" in df.columns else "volume" if "volume" in df.columns else None
        if not vol_col:
            return "Volume data unavailable."
        
        # Convert volume data to numeric, handling any non-numeric values
        try:
            # Create a copy of the dataframe to avoid SettingWithCopyWarning
            df_copy = df.copy()
            df_copy[vol_col] = pd.to_numeric(df_copy[vol_col], errors='coerce')
            
            # Set parameters based on interval
            if interval == "5min":
                num_candles = min(30, len(df_copy))
                high_volume_threshold = 1.3  # 30% above average
                low_volume_threshold = 0.8   # 20% below average
            else:
                num_candles = len(df_copy)
                high_volume_threshold = 1.5
                low_volume_threshold = 0.75

            # Calculate average volume and get the latest volume
            avg_volume = df_copy[vol_col].tail(num_candles).mean()
            latest_volume = df_copy.iloc[-1][vol_col]
            
            # Check for NaN or zero values
            if pd.isna(latest_volume) or pd.isna(avg_volume) or avg_volume == 0:
                return "Volume data analysis unavailable due to incomplete data."
                
            volume_ratio = latest_volume / avg_volume
        except Exception as e:
            logger.warning(f"Error processing volume data: {str(e)}")
            return "Volume data couldn't be analyzed due to data format issues."

        # Determine volume comment based on the ratio
        if volume_ratio > high_volume_threshold:
            vol_comment = "Volume is significantly high, reinforcing the trend."
        elif volume_ratio < low_volume_threshold:
            vol_comment = "Volume is relatively low, advising caution on this move."
        else:
            vol_comment = "Volume is at average levels, suggesting normal market activity."

        logger.debug(f"Timeframe: {interval}, Latest Volume: {latest_volume}, Avg Volume (over last {num_candles if interval=='5min' else 'all'} candles): {avg_volume:.2f}, Volume Ratio: {volume_ratio:.2f}")
        return vol_comment
        
    async def search_for_asset(self, query: str, search_type: str = "web", max_results: int = 5):
        """
        Search for asset information using the Tavily API.
        
        Parameters:
            query: The search query about an asset or market
            search_type: Type of search - "web" for general search or "news" for recent news
            max_results: Maximum number of results to return
            
        Returns:
            Dict containing search results, with asset information if found
        """
        try:
            # Use MCP tool to search for asset information
            
            # Check if search is for a specific symbol
            potential_symbol = None
            match = re.search(r'\$([A-Za-z0-9]+)', query)
            if match:
                potential_symbol = match.group(1).upper()
            else:
                # Look for text that could be a ticker
                for word in query.split():
                    if len(word) <= 5 and word.upper() == word and word.isalpha():
                        potential_symbol = word.upper()
                        break
            
            # If we found a potential symbol, check if it's a valid asset
            if potential_symbol:
                # Try to get price data to validate the symbol using direct function call
                try:
                    # Check if it's a stock
                    stock_data = await get_technical_indicators(symbol=potential_symbol, asset_type="stock", timeframe="daily")
                    if "error" not in stock_data:
                        # It's a valid stock
                        return {
                            "query": query,
                            "asset_found": True,
                            "symbol": potential_symbol,
                            "asset_type": "stock",
                            "price_data": stock_data.get("price_data", {}),
                            "technical_data": stock_data,
                            "timestamp": datetime.now(UTC).isoformat()
                        }
                except Exception:
                    pass
                
                # Check if it's a crypto using direct function call
                try:
                    crypto_data = await get_technical_indicators(symbol=potential_symbol, asset_type="crypto", timeframe="daily")
                    if "error" not in crypto_data:
                        # It's a valid crypto
                        return {
                            "query": query,
                            "asset_found": True,
                            "symbol": potential_symbol,
                            "asset_type": "crypto",
                            "price_data": crypto_data.get("price_data", {}),
                            "technical_data": crypto_data,
                            "timestamp": datetime.now(UTC).isoformat()
                        }
                except Exception:
                    pass
            
            # If we couldn't find a valid asset or no potential symbol was found,
            # perform a general search using the Tavily API
            # Note: Need to use mcp.search_market_information since we didn't import this function
            search_results = await mcp.search_market_information(query, search_type)
            
            return {
                "query": query,
                "asset_found": False,
                "potential_symbol": potential_symbol,
                "search_results": search_results,
                "timestamp": datetime.now(UTC).isoformat()
            }
        except Exception as e:
            logger.error(f"Error searching for asset with query '{query}': {e}")
            return {
                "query": query,
                "asset_found": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat()
            }
    
    async def process_trend_following_strategy(self, symbol: str, asset_type: str = "stock", timeframe: str = "daily"):
        """
        Generates a trend following strategy for the given asset.
        
        Parameters:
            symbol: The ticker symbol to analyze
            asset_type: Either "stock" or "crypto"
            timeframe: The timeframe for analysis
            
        Returns:
            Dictionary containing strategy details and recommendations
        """
        try:
            # Use MCP tools to get comprehensive data
            
            # Get technical indicators using direct function call
            tech_data = await get_technical_indicators(symbol=symbol, asset_type=asset_type, timeframe=timeframe)
            if "error" in tech_data:
                return {"error": tech_data["error"], "symbol": symbol}
                
            # Get price levels using direct function call
            price_levels = await get_price_levels(symbol=symbol, asset_type=asset_type, timeframe=timeframe)
            
            # Get multi-timeframe analysis for broader context using direct function call
            mtf_analysis = await analyze_multi_timeframe(symbol=symbol, asset_type=asset_type, primary_timeframe=timeframe)
            
            # Extract key data points
            current_price = tech_data.get("price_data", {}).get("current", 0)
            rsi = tech_data.get("indicators", {}).get("rsi")
            macd = tech_data.get("indicators", {}).get("macd", {}).get("value")
            macd_signal = tech_data.get("indicators", {}).get("macd", {}).get("signal")
            stoch_k = tech_data.get("indicators", {}).get("stochastic", {}).get("k")
            atr = tech_data.get("indicators", {}).get("atr")
            
            # Get support and resistance levels
            support_levels = price_levels.get("formatted_support", [])
            resistance_levels = price_levels.get("formatted_resistance", [])
            
            # Nearest support and resistance
            nearest_support = next((float(level["price"]) for level in support_levels if "price" in level), current_price * 0.95)
            nearest_resistance = next((float(level["price"]) for level in resistance_levels if "price" in level), current_price * 1.05)
            
            # Default strategy components
            position_sizing = "Allocate 1-2% of portfolio per trade"
            risk_reward = 2.0  # Default risk/reward ratio
            stop_loss_pct = 0.02  # Default 2% stop loss
            take_profit_pct = 0.04  # Default 4% take profit
            
            # Calculate stop loss and take profit levels
            if atr:
                # Use ATR for more accurate stop loss calculation
                stop_distance = atr * 2
                stop_loss = current_price - stop_distance if mtf_analysis.get("signal") == "BUY" else current_price + stop_distance
                take_profit = current_price + (stop_distance * risk_reward) if mtf_analysis.get("signal") == "BUY" else current_price - (stop_distance * risk_reward)
            else:
                # Fallback to percentage-based
                stop_loss = current_price * (1 - stop_loss_pct) if mtf_analysis.get("signal") == "BUY" else current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct) if mtf_analysis.get("signal") == "BUY" else current_price * (1 - take_profit_pct)
            
            # Calculate risk-reward ratio based on current price, stop loss, and nearest support/resistance
            if mtf_analysis.get("signal") == "BUY":
                potential_reward = nearest_resistance - current_price
                potential_risk = current_price - stop_loss
            else:
                potential_reward = current_price - nearest_support
                potential_risk = stop_loss - current_price
                
            actual_risk_reward = abs(potential_reward / potential_risk) if potential_risk != 0 else risk_reward
            
            # Entry and exit rules based on trend and indicators
            entry_rules = []
            exit_rules = []
            
            # Build entry rules
            if mtf_analysis.get("signal") == "BUY":
                entry_rules.append("Enter when price breaks above nearest resistance with increased volume")
                entry_rules.append("Enter on pullbacks to moving average when trend is bullish")
                entry_rules.append("Consider entering when RSI rebounds from oversold conditions")
            else:
                entry_rules.append("Enter when price breaks below nearest support with increased volume")
                entry_rules.append("Enter on bounces to moving average when trend is bearish")
                entry_rules.append("Consider entering when RSI turns down from overbought conditions")
                
            # Build exit rules
            exit_rules.append("Exit when price reaches the calculated take profit level")
            exit_rules.append("Exit when price hits the stop loss level")
            exit_rules.append("Consider partial profit taking at key resistance/support levels")
            
            if mtf_analysis.get("signal") == "BUY":
                exit_rules.append("Exit long positions when MACD crosses below signal line")
            else:
                exit_rules.append("Exit short positions when MACD crosses above signal line")
            
            # Create strategy object with all components
            strategy = {
                "symbol": symbol,
                "asset_type": asset_type,
                "timeframe": timeframe,
                "current_price": current_price,
                "trend_signal": mtf_analysis.get("signal"),
                "reasoning": mtf_analysis.get("reasoning"),
                "strategy_type": "Trend Following",
                "position_sizing": position_sizing,
                "risk_reward_ratio": round(actual_risk_reward, 2),
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2),
                "entry_rules": entry_rules,
                "exit_rules": exit_rules,
                "nearest_support": round(nearest_support, 2),
                "nearest_resistance": round(nearest_resistance, 2),
                "technical_readings": {
                    "rsi": rsi,
                    "macd": macd,
                    "stochastic": stoch_k
                },
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            return strategy
        except Exception as e:
            logger.error(f"Error generating trend following strategy for {symbol}: {e}")
            return {
                "error": f"Failed to generate strategy: {str(e)}",
                "symbol": symbol,
                "timeframe": timeframe
            }
    
    async def generate_market_strategy(self, symbol: str, asset_type: str = "stock", timeframe: str = "daily"):
        """
        Generate a comprehensive market strategy for an asset.
        This method combines technical, sentiment, and multi-timeframe analysis.
        
        Parameters:
            symbol: The ticker symbol
            asset_type: Either "stock" or "crypto"
            timeframe: The timeframe for analysis
            
        Returns:
            Dictionary containing the complete market strategy
        """
        try:
            # Use MCP tools to get comprehensive data
            
            # Get trend following strategy
            trend_strategy = await self.process_trend_following_strategy(symbol, asset_type, timeframe)
            if "error" in trend_strategy:
                return {"error": trend_strategy["error"], "symbol": symbol}
            
            # Get news sentiment using direct function call
            sentiment_data = await get_news_sentiment(symbol=symbol)
            
            # Analyze the asset
            analysis = await self.analyze_asset(symbol, asset_type, "USD", timeframe, False)
            
            # Format the strategy response
            strategy = {
                "symbol": symbol,
                "asset_type": asset_type,
                "timeframe": timeframe,
                "current_price": trend_strategy.get("current_price"),
                "signal": trend_strategy.get("trend_signal"),
                "reasoning": trend_strategy.get("reasoning"),
                "analysis": analysis,
                "strategy": {
                    "type": "Trend Following",
                    "position_sizing": trend_strategy.get("position_sizing"),
                    "risk_reward_ratio": trend_strategy.get("risk_reward_ratio"),
                    "stop_loss": trend_strategy.get("stop_loss"),
                    "take_profit": trend_strategy.get("take_profit"),
                    "entry_rules": trend_strategy.get("entry_rules"),
                    "exit_rules": trend_strategy.get("exit_rules")
                },
                "price_levels": {
                    "nearest_support": trend_strategy.get("nearest_support"),
                    "nearest_resistance": trend_strategy.get("nearest_resistance")
                },
                "sentiment": {
                    "score": sentiment_data.get("sentiment", {}).get("score"),
                    "label": sentiment_data.get("sentiment", {}).get("label"),
                    "news_count": sentiment_data.get("news_count")
                },
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            # Add most relevant news article if available
            if sentiment_data.get("most_relevant_article"):
                strategy["sentiment"]["most_relevant_article"] = sentiment_data.get("most_relevant_article")
            
            return strategy
        except Exception as e:
            logger.error(f"Error generating market strategy for {symbol}: {e}")
            return {
                "error": f"Failed to generate market strategy: {str(e)}",
                "symbol": symbol
            }