import os
import json
import logging
import asyncio
import re
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
from market.mcp_tools import get_technical_indicators, get_news_sentiment, get_raw_market_data, add_market_analysis_to_context
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
            
            # Add market analysis context content (as system instruction)
            is_analysis_request = self._is_analysis_request(text)
            if is_analysis_request:
                # Extract symbol and timeframe
                symbol = self._extract_ticker(text)
                timeframe = self._extract_interval(text)
                
                if symbol:
                    # Detect asset type
                    asset_type = self._determine_asset_type(symbol)
                    
                    # Get market analysis as string
                    try:
                        # Use the already imported add_market_analysis_to_context function
                        
                        # Call WITHOUT messages_list to get the string content
                        market_analysis_context_str = await add_market_analysis_to_context(
                            self,
                            symbol=symbol,
                            asset_type=asset_type,
                            timeframe=timeframe or "daily"
                            # messages_list is omitted here
                        )
                        if market_analysis_context_str:
                            system_prompts_content.append(market_analysis_context_str)
                    except Exception as e:
                        logger.error(f"Error adding market analysis context: {e}", exc_info=True)
                        system_prompts_content.append(f"Note: Market data for {symbol} could not be retrieved. Error: {str(e)}")
            
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
        Completely implemented using MCP tools.
        
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
            # --- Prepare System Prompt ---
            system_prompts_content = []  # List to hold all system instruction strings
            
            # Get market analysis template from PromptManager
            market_analysis_template = self.prompt_manager.get_template_section('market_analysis_template')
            if market_analysis_template:
                analysis_prompt = market_analysis_template.get("header", "") + \
                                 market_analysis_template.get("sentiment_section", "") + \
                                 market_analysis_template.get("recommendation_section", "")
            else:
                # Fallback to a basic template if not found
                analysis_prompt = "Analyze this asset based on technical indicators, price levels, and market sentiment."
                
            system_prompts_content.append(analysis_prompt)
            
            # Generate dynamic multi-timeframe analysis instead of using a static template
            # First, get data across multiple timeframes to build a data-driven MTF context
            current_price = None
            latest_atr = None
            mtf_context = ""
            
            # Get multi-timeframe support/resistance levels
            try:
                # First get data for the primary timeframe to extract current price and ATR using direct function call
                primary_tech_data = await get_technical_indicators(symbol=symbol, asset_type=asset_type, timeframe=interval)
                if primary_tech_data and "price_data" in primary_tech_data:
                    current_price = primary_tech_data.get("price_data", {}).get("current")
                    latest_atr = primary_tech_data.get("indicators", {}).get("atr", {}).get("value")
                
                # Now get multi-timeframe analysis
                if current_price:
                    # Get MTF analysis which includes consolidated support/resistance zones using direct function calls
                    mtf_analysis = await analyze_multi_timeframe(symbol=symbol, asset_type=asset_type, primary_timeframe=interval)
                    price_levels = await get_price_levels(symbol=symbol, asset_type=asset_type, timeframe=interval)
                    
                    # Generate a text summary of MTF analysis that reflects the data
                    mtf_context = "MULTI-TIMEFRAME ANALYSIS:\n\n"
                    
                    # Add signal and reasoning from MTF analysis
                    mtf_context += f"Overall Signal: {mtf_analysis.get('signal', 'NEUTRAL')}\n"
                    mtf_context += f"Reasoning: {mtf_analysis.get('reasoning', 'Insufficient data')}\n\n"
                    
                    # Add key support levels
                    mtf_context += "Key Support Levels (across timeframes):\n"
                    support_zones = price_levels.get("consolidated_support", [])
                    for zone in support_zones[:3]:  # Top 3 support zones
                        if "price" in zone and zone["price"] < current_price:
                            # Handle timeframes that might be a string, list, or any other type
                            timeframes_data = zone.get("timeframes", [])
                            if isinstance(timeframes_data, list):
                                timeframes = "/".join(timeframes_data)
                            elif isinstance(timeframes_data, str):
                                timeframes = timeframes_data
                            else:
                                # Handle float or other types by converting to string
                                timeframes = str(timeframes_data)
                                
                            strength = zone.get('strength', 0)
                            mtf_context += f"- ${zone['price']} (strength: {strength:.1f}, timeframes: {timeframes})\n"
                    
                    # Add key resistance levels  
                    mtf_context += "\nKey Resistance Levels (across timeframes):\n"
                    resistance_zones = price_levels.get("consolidated_resistance", [])
                    for zone in resistance_zones[:3]:  # Top 3 resistance zones
                        if "price" in zone and zone["price"] > current_price:
                            # Handle timeframes that might be a string, list, or any other type
                            timeframes_data = zone.get("timeframes", [])
                            if isinstance(timeframes_data, list):
                                timeframes = "/".join(timeframes_data)
                            elif isinstance(timeframes_data, str):
                                timeframes = timeframes_data
                            else:
                                # Handle float or other types by converting to string
                                timeframes = str(timeframes_data)
                                
                            strength = zone.get('strength', 0)
                            mtf_context += f"- ${zone['price']} (strength: {strength:.1f}, timeframes: {timeframes})\n"
            except Exception as e:
                logger.error(f"Error generating MTF context: {e}")
                # Fallback to generic MTF guidelines
                mtf_context = f"""
                Multi-timeframe Analysis Guidelines:
                - Primary timeframe: {interval}
                - When analyzing, consider both higher and lower timeframes
                - Higher timeframes show overall trend direction
                - Lower timeframes reveal entry/exit opportunities
                - Confluence of signals across timeframes indicates stronger support/resistance
                """
            
            # Add the data-driven MTF context
            system_prompts_content.append(mtf_context)
            
            # Gather technical analysis data using direct function calls
            tech_data = await get_technical_indicators(symbol=symbol, asset_type=asset_type, timeframe=interval)
            price_levels = await get_price_levels(symbol=symbol, asset_type=asset_type, timeframe=interval)
            sentiment = await get_news_sentiment(symbol=symbol)
            
            # Get multi-timeframe analysis if not already fetched
            if 'mtf_analysis' not in locals() or not mtf_analysis:
                mtf_analysis = await analyze_multi_timeframe(symbol=symbol, asset_type=asset_type, primary_timeframe=interval)
            
            # Add market data to context
            analysis_context = f"""
            ## Technical Analysis Data for {symbol} ({interval})
            
            Current Price: ${tech_data.get('price_data', {}).get('current', 'N/A')}
            
            Technical Indicators:
            {json.dumps(tech_data.get('indicators', {}), indent=2)}
            
            Support/Resistance Levels:
            Support: {json.dumps(price_levels.get('formatted_support', []))} below current price
            Resistance: {json.dumps(price_levels.get('formatted_resistance', []))} above current price
            
            News Sentiment:
            Score: {sentiment.get('sentiment', {}).get('score', 'N/A')} ({sentiment.get('sentiment', {}).get('label', 'NEUTRAL')})
            News Count: {sentiment.get('news_count', 0)}
            Most Relevant Article: {sentiment.get('most_relevant_article', {}).get('title', 'N/A') if sentiment.get('most_relevant_article') else 'N/A'}
            
            Signal: {mtf_analysis.get('signal', 'NEUTRAL')}
            Reasoning: {mtf_analysis.get('reasoning', 'Insufficient data')}
            
            Please provide a comprehensive analysis of {symbol} based on this data. Include:
            1. Overall trend analysis across multiple timeframes
            2. Key support and resistance levels
            3. Technical indicator analysis
            4. Market sentiment impact
            5. Potential trade scenarios with risk/reward ratios
            6. Clear trading bias (bullish, bearish, or neutral)
            """
            
            system_prompts_content.append(analysis_context)
            
            # Combine all system instructions into a single string
            final_system_prompt = "\n\n".join(system_prompts_content)
            
            # --- Prepare Messages List (User/Assistant roles ONLY) ---
            messages_for_llm = []
            
            # Add user message requesting analysis
            user_message = f"Please analyze {symbol} on the {interval} timeframe and provide trading insights."
            messages_for_llm.append({"role": "user", "content": user_message})
            
            # Generate the analysis using Anthropic client with correct structure
            try:
                response = await self.anthropic_client.messages.create(
                    messages=messages_for_llm,      # ONLY user message
                    system=final_system_prompt,     # Pass system instructions here
                    model=self.settings.SOCIAL_MODEL,
                    max_tokens=4000,
                    temperature=0
                )
            except Exception as api_error:
                # Log the specific API error
                logger.error(f"Anthropic API call failed in analyze_asset: {api_error}", exc_info=True)
                # Re-raise to be caught by the outer try-except
                raise api_error
            
            # Extract the analysis text from Anthropic's response structure
            analysis = ""
            if hasattr(response, 'content') and response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        analysis += block.text
            else:
                logger.warning("Received empty content from Anthropic API for analysis")
                analysis = "Sorry, I encountered an issue generating the analysis."
            
            # Format for user display if requested
            if for_user_display:
                symbol_caps = symbol.upper()
                asset_type_caps = asset_type.upper()
                price = tech_data.get('price_data', {}).get('current', 'N/A')
                change_pct = tech_data.get('price_data', {}).get('change_percent', 0)
                change_direction = "📈" if change_pct >= 0 else "📉"
                sentiment_label = sentiment.get('sentiment', {}).get('label', 'NEUTRAL')
                
                # Extract article highlight if available
                article_highlight = ""
                most_relevant = sentiment.get('most_relevant_article')
                if most_relevant:
                    article_highlight = f"📑 LATEST NEWS: \"{most_relevant.get('title', '')}\"\nSummary: {most_relevant.get('summary', '')}\n\n"
                
                formatted_analysis = f"""
                ⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️
                
                🇺🇸 MAGANALYSIS {asset_type_caps} ANALYSIS 🇺🇸
                
                {symbol_caps} {interval.upper()}
                
                💰 CURRENT {symbol_caps} PRICE: ${price}
                
                📊 24-HOUR CHANGE: {change_direction} {abs(change_pct):.2f}%
                
                ================================
                📰🔎  NEWS SENTIMENT  📰🔎
                ================================
                
                {sentiment_label}
                
                {article_highlight}
                
                =================================
                🚨 MARKET STRATEGY 🚨
                =================================
                
                {analysis}
                
                ⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️
                """
                
                # Process the formatted analysis through MessageProcessor
                from utils.message_handling import MessageProcessor
                from utils.mcp_message_handling import MCPMessageProcessor
                
                # Clean the message for standard formatting
                cleaned_analysis = MessageProcessor.clean_message(formatted_analysis)
                
                # Format specifically for the platform if platform is provided
                if 'platform' in locals() or 'platform' in globals():
                    if platform == "telegram":
                        cleaned_analysis = MCPMessageProcessor.format_for_telegram(cleaned_analysis)
                
                return cleaned_analysis
            
            return analysis
            
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
        import re
        
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
        
        Parameters:
            symbol: The asset symbol to analyze
            asset_type: Either "stock" or "crypto"
            market: The market for crypto assets (e.g., "USD")
            interval: The timeframe to analyze
            
        Returns:
            dict: Analysis results for the timeframe
        """
        try:
            # Get technical indicators data using direct function calls
            tech_data = await get_technical_indicators(symbol=symbol, asset_type=asset_type, timeframe=interval)
            price_levels = await get_price_levels(symbol=symbol, asset_type=asset_type, timeframe=interval)
            
            # Extract key data
            price_data = tech_data.get("price_data", {})
            indicators = tech_data.get("indicators", {})
            trend_data = tech_data.get("trend", {})
            
            # Extract current price and metadata
            current_price = price_data.get("current")
            price_change_pct = price_data.get("change_percent", 0)
            change_direction = "↑" if price_change_pct > 0 else "↓" if price_change_pct < 0 else "→"
            
            # Extract formatted support and resistance levels
            support_levels = price_levels.get("formatted_support", [])
            resistance_levels = price_levels.get("formatted_resistance", [])
            
            # Determine if this is an extended timeframe
            is_extended = interval in ["daily", "weekly", "monthly"]
            
            # Return timeframe analysis
            return {
                "interval": interval,
                "current_price": current_price,
                "latest_open": price_data.get("open"),
                "price_change_pct": price_change_pct,
                "change_direction": change_direction,
                "most_recent_date": price_data.get("timestamp"),
                "indicators": indicators,
                "trend": trend_data,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "is_extended": is_extended,
                "processed_df": None  # We don't need to return the raw dataframe when using MCP
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
                "processed_df": None
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
            
            # Get comprehensive price level data across timeframes using direct function calls
            mtf_analysis = await analyze_multi_timeframe(symbol=symbol, asset_type=asset_type, primary_timeframe=main_interval)
            price_levels = await get_price_levels(symbol=symbol, asset_type=asset_type, timeframe=main_interval)
            
            if "error" in mtf_analysis or "error" in price_levels:
                logger.warning(f"Error in multi-timeframe analysis for {symbol}: {mtf_analysis.get('error') or price_levels.get('error')}")
                return None
                
            # Extract support and resistance zones
            support_zones = price_levels.get("consolidated_support", [])
            resistance_zones = price_levels.get("consolidated_resistance", [])
            
            # Create summary text
            summary = "MULTI-TIMEFRAME LEVEL ANALYSIS:\n"
            
            # Add key support levels
            summary += "\nKey Support Levels (across timeframes):\n"
            for zone in support_zones[:3]:  # Top 3 support zones
                if "price" in zone and zone["price"] < current_price:
                    timeframes = "/".join(zone["timeframes"])
                    strength = zone.get('strength', 0)
                    summary += f"- ${zone['price']} (strength: {strength:.1f}, timeframes: {timeframes})\n"
            
            # Add key resistance levels  
            summary += "\nKey Resistance Levels (across timeframes):\n"
            for zone in resistance_zones[:3]:  # Top 3 resistance zones
                if "price" in zone and zone["price"] > current_price:
                    timeframes = "/".join(zone["timeframes"])
                    strength = zone.get('strength', 0)
                    summary += f"- ${zone['price']} (strength: {strength:.1f}, timeframes: {timeframes})\n"
            
            # Prepare result with current timestamp
            result = {
                "support_zones": support_zones,
                "resistance_zones": resistance_zones,
                "summary": summary,
                "signal": mtf_analysis.get("signal", "NEUTRAL"),
                "reasoning": mtf_analysis.get("reasoning", "Insufficient data"),
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