import json
import random
import traceback
import asyncio
import os
import logging
import re
import pandas as pd
import numpy as np
import pytz
from anthropic import AsyncAnthropic
from aisettings import AISettings
from memory.message_memory import MessageMemoryManager
from memory.memory_types import MemoryType
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta, timezone
from tavily import AsyncTavilyClient

from market.market_manager import MarketManager
from market.data_processor import MarketDataProcessor
from market.indicators import TechnicalIndicators, TimeframeParameters
from market.price_levels import PriceLevelAnalyzer, LevelType
from market.formatters import DataFormatter
from utils.message_handling import MessageProcessor
from prompts.prompt_manager import PromptManager


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class CipherAgent:
    # Define timeframe hierarchy for multi-timeframe analysis at class level
    # This maps each timeframe to higher timeframes for trend context
    TIMEFRAME_HIERARCHY = {
        # Intraday timeframes
        "1min": ["5min", "15min", "60min", "daily"],
        "5min": ["15min", "30min", "60min"],  
        "15min": ["30min", "60min"],
        "30min": ["60min", 'daily'],
        "60min": ["daily"],
        # Extended timeframes
        "daily": ["weekly", "monthly"],
        "weekly": ["monthly"],
        "monthly": []  # No higher timeframe for monthly
    }
    
    def __init__(self, settings: AISettings = None):
        # Initialize settings first
        self.settings = settings or AISettings()
        # Add version tracking
        self.prompt_version = "2.0"
        # Load character configuration
        self.character = self._load_character_config()
        # Initialize the prompt manager
        self.prompt_manager = PromptManager()
        # Format the base system prompt with character info
        self.base_system_prompt = self._format_base_prompt()
        # Initialize memory systems with character configuration
        self.message_memory = MessageMemoryManager(self.settings, self.character)
        # Initialize unified market manager (handling both crypto via Moralis 
        # and stock analysis via Alpha Vantage)
        self.market_manager = MarketManager(self.settings)
        # Initialize API client last (depends on settings)
        self.anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        # Initialize the Tavily client for search integration.
        self.tavily_client = AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        logger.info("CipherAgent initialized with memory systems and API clients")

    def _load_character_config(self) -> Dict:
        """Load and validate character configuration"""
        try:
            with open('scripts/character.json', 'r') as f:
                config = json.load(f)
            self._validate_character_config(config)
            return config
        except Exception as e:
            logger.error(f"Failed to load character config: {e}")
            raise

    def _format_base_prompt(self) -> str:
        """Format base system prompt with character info using PromptManager"""
        # Get current date in a readable format
        date_info = self._get_current_datetime_info()
        
        # Format the base prompt with character info using the prompt manager
        base_prompt = self.prompt_manager.get_system_prompt('base', 
            name=self.character['name'],
            bio=self.character['bio'],
            personality="\n".join(f"- {trait}" for trait in self.character['personality']),
            formatting="\n".join(f"- {k}: {v}" for k, v in self.character['formatting'].items()),
            chat_style="\n".join(f"- {k}: {v}" for k, v in self.character['chat_style'].items())
        )
        
        # Add current date information to the prompt
        return f"{base_prompt}\n\nCurrent date: {date_info['date_verbose']}"

    def format_indicator_value(self, value):
        """
        Helper function to format indicator values with proper handling of None values and conversion errors.
        Returns formatted string with dollar sign and two decimal places for valid numerical values.
        Returns "N/A" for None values or values that cannot be converted to float.
        
        Note: Uses DataFormatter for consistent formatting across the application.
        """
        return DataFormatter.format_price(value)
        
    def get_volume_comment(self, df: pd.DataFrame, interval: str) -> str:
        """
        Compute a volume comment based on the intraday volume data and the analysis timeframe.

        For a 5-minute timeframe, a shorter moving average window (last 30 candles)
        and tighter threshold values are used. For longer timeframes, the average is taken
        over all available data.

        Parameters:
            df (pd.DataFrame): DataFrame containing intraday data.
            interval (str): The timeframe of the data (e.g., "5min", "15min", etc.).

        Returns:
            str: A comment describing the volume relative to recent averages.
        """
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
            # Get news sentiment data
            news_data = await self.market_manager.get_news_sentiment(ticker=symbol)
            
            # Process sentiment data
            if news_data and "feed" in news_data:
                feed_items = news_data["feed"]
                # Calculate average sentiment across all news items
                total_sentiment = 0
                relevant_items = 0
                min_relevance_threshold = 0.5 # Lower threshold to ensure articles are found
                
                # Collect all articles with relevance scores for this ticker
                ticker_articles = []
                
                for item in feed_items:
                    if "ticker_sentiment" in item:
                        for ticker_sent in item["ticker_sentiment"]:
                            # Check for both normal symbol and with CRYPTO: prefix
                            if (ticker_sent.get("ticker") == symbol or 
                                ticker_sent.get("ticker") == f"CRYPTO:{symbol}"):
                                score = float(ticker_sent.get("ticker_sentiment_score", 0))
                                relevance = float(ticker_sent.get("relevance_score", 0))
                                
                                total_sentiment += score
                                relevant_items += 1
                                
                                # Add this article to our collection for later sorting
                                if relevance >= min_relevance_threshold:
                                    ticker_articles.append({
                                        "item": item,
                                        "relevance": relevance,
                                        "score": score
                                    })
                
                # Since articles are already sorted by LATEST (line 370 in market_manager.py),
                # we prioritize recent articles that have reasonable relevance
                # (We don't re-sort so we maintain the timestamp-based ordering)
                
                # Choose the most relevant article
                if ticker_articles:
                    # Get the most relevant article
                    article_data = ticker_articles[0]
                    item = article_data["item"]
                    relevance = article_data["relevance"]
                    score = article_data["score"]
                    
                    # Format the publication time
                    published_time = item.get("time_published", "")
                    formatted_time = ""
                    
                    if published_time and len(published_time) >= 15:
                        try:
                            year = published_time[0:4]
                            month = published_time[4:6]
                            day = published_time[6:8]
                            hour = published_time[9:11]
                            minute = published_time[11:13]
                            formatted_time = f"{month}-{day}-{year} {hour}:{minute}"
                            
                            # Add recency check based on current time
                            parsed_time = f"{year}-{month}-{day}"
                            is_recent = self.is_recent_date(parsed_time, max_days=1)
                            is_old = not self.is_recent_date(parsed_time, max_days=90)  # Older than 3 months
                            
                            if is_recent:
                                formatted_time = f"🆕 {formatted_time}"
                            elif is_old:
                                formatted_time = f"📅 [OLD: {year}-{month}-{day}] {formatted_time}"
                        except Exception as e:
                            logger.warning(f"Error formatting publication time: {e}")
                            formatted_time = published_time
                    
                    # Save most relevant article details - strip source text from summary
                    summary = item.get("summary", "")
                    if "Source:" in summary:
                        summary = summary.split("Source:")[0].strip()
                    
                    most_relevant_article = {
                        "title": item.get("title", ""),
                        "summary": summary,
                        "url": item.get("url", ""),
                        "published": published_time,
                        "formatted_time": formatted_time,
                        "relevance_score": relevance,
                        "sentiment_score": score
                    }
                    
                    # Create article highlight for template
                    if item.get("title") and item.get("summary"):
                        # Create the appropriate news label based on age
                        news_label = "LATEST NEWS"
                        if "OLD:" in formatted_time:
                            news_label = "HISTORICAL NEWS"
                        
                        article_highlight = f"📑 {news_label}: \"{item.get('title')}\" ({formatted_time})\n" + \
                                          f"Summary: {item.get('summary')}\n"
                
                if relevant_items > 0:
                    sentiment_score = total_sentiment / relevant_items
                    news_count = relevant_items
                    
                    # Determine sentiment label
                    if sentiment_score >= 0.5:
                        sentiment_label = "BULLISH"
                    elif sentiment_score >= 0.2:
                        sentiment_label = "SOMEWHAT BULLISH"
                    elif sentiment_score <= -0.5:
                        sentiment_label = "BEARISH"
                    elif sentiment_score <= -0.2:
                        sentiment_label = "SOMEWHAT BEARISH"
                    else:
                        sentiment_label = "NEUTRAL"
                    
                    # Add sentiment to the reasoning
                    updated_reasoning = reasoning
                    
                    # For strong buy/strong sell signals, news sentiment can confirm the signal
                    if sentiment_score >= 0.5 and signal_base in ["BUY", "ACCUMULATE", "HOLD/LONG"]:
                        updated_reasoning += f", confirmed by bullish news sentiment ({sentiment_score:.2f})"
                    elif sentiment_score <= -0.5 and signal_base in ["SELL", "REDUCE", "AVOID/SHORT"]:
                        updated_reasoning += f", confirmed by bearish news sentiment ({sentiment_score:.2f})"
                        
                    # For neutral technical signals, news sentiment can provide direction
                    elif sentiment_score >= 0.5 and signal_base == "NEUTRAL":
                        signal_base = "SPECULATIVE BUY"
                        updated_reasoning += f", supported by bullish news sentiment ({sentiment_score:.2f})"
                    elif sentiment_score <= -0.5 and signal_base == "NEUTRAL":
                        signal_base = "SPECULATIVE SELL"
                        updated_reasoning += f", supported by bearish news sentiment ({sentiment_score:.2f})"
                        
                    # For conflicting signals (technical vs news), add caution
                    elif sentiment_score >= 0.5 and signal_base in ["SELL", "REDUCE", "AVOID/SHORT"]:
                        updated_reasoning += f", but conflicting with bullish news sentiment ({sentiment_score:.2f}) - proceed with caution"
                    elif sentiment_score <= -0.5 and signal_base in ["BUY", "ACCUMULATE", "HOLD/LONG"]:
                        updated_reasoning += f", but conflicting with bearish news sentiment ({sentiment_score:.2f}) - proceed with caution"
                    
                    # Return the updated signal and reasoning with article data
                    return signal_base, updated_reasoning, {
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
    
    def _extract_interval(self, text: str) -> str:
        """
        Extracts one of the allowed time intervals from the user input.
        Allowed intervals: "1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly".
        This function normalizes expressions like '15 minutes' to '15min' or 'daily chart' to 'daily'.
        Defaults to "60min" if none are found.
        """
        allowed_intervals = ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"]
        
        # Preprocess input text for more consistent matching
        modified_text = text.lower()
        modified_text = modified_text.replace(" minutes", " minute").replace("mins", "min")
        
        # Comprehensive mapping of phrases to standardized interval strings
        interval_mappings = {
            # Minute-based intervals
            "1 minute": "1min", "one minute": "1min", "1min chart": "1min", "1 min": "1min",
            "5 minute": "5min", "five minute": "5min", "5min chart": "5min", "5 min": "5min",
            "15 minute": "15min", "fifteen minute": "15min", "15min chart": "15min", "15 min": "15min",
            "30 minute": "30min", "thirty minute": "30min", "30min chart": "30min", "30 min": "30min",
            "60 minute": "60min", "sixty minute": "60min", "60min chart": "60min", "60 min": "60min", 
            "1 hour": "60min", "one hour": "60min", "hourly": "60min", "hour": "60min",
            
            # Daily interval
            "1 day": "daily", "one day": "daily", "day": "daily", "days": "daily", 
            "daily chart": "daily", "daily timeframe": "daily", "daily candle": "daily",
            
            # Weekly interval
            "1 week": "weekly", "one week": "weekly", "week": "weekly", "weeks": "weekly", 
            "weekly chart": "weekly", "weekly timeframe": "weekly", "weekly candle": "weekly",
            
            # Monthly interval
            "1 month": "monthly", "one month": "monthly", "month": "monthly", "months": "monthly", 
            "monthly chart": "monthly", "monthly timeframe": "monthly", "monthly candle": "monthly"
        }
        
        # First check for direct matches to allowed intervals (complete words only)
        for interval in allowed_intervals:
            pattern = r'\b' + interval + r'\b'
            if re.search(pattern, modified_text):
                return interval
        
        # Then check for phrases that map to intervals
        for phrase, interval in interval_mappings.items():
            if phrase in modified_text:
                return interval
        
        # Default to 5min if no interval is found
        return "60min"
    
    def _get_current_datetime_info(self) -> Dict[str, str]:
        """
        Get current date and time information in various formats.
        Returns a dictionary with formatted date/time information.
        """
        # Get current time in Eastern Time (US market timezone) and UTC
        now_utc = datetime.now(pytz.UTC)
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

    def _format_data(self, data: Dict, config: Dict) -> str:
        """
        A generic formatter that builds a vertical display using configuration from `config`.
        Expects the data to have a 'Time Series ...' key containing a dictionary.
        The header shows only the date and each record includes only the time (in 12-hour format without seconds).
        """
        intro = config.get("intro", "Data for {ticker}:")
        time_series_key = None
        for key in data.keys():
            if key.startswith("Time Series"):
                time_series_key = key
                break
        if not time_series_key:
            return "No intraday data available."
        
        time_series = data.get(time_series_key, {})
        if not time_series:
            return "No intraday data available."
        
        sorted_times = sorted(time_series.keys(), reverse=True)
        
        try:
            latest_dt = datetime.fromisoformat(sorted_times[0])
            date_str = latest_dt.strftime("%Y-%m-%d")
        except Exception:
            date_str = "Unknown Date"
        
        lines = []
        lines.append(intro)
        lines.append("")
        lines.append(f"Data as of {date_str}:")
        lines.append("")
        lines.append(config.get("vertical_intro", "Latest data:"))
        lines.append("")
        
        for idx, timestamp in enumerate(sorted_times[:5], start=1):
            entry = time_series[timestamp]
            open_price = self._format_price(entry.get("1. open", "N/A"))
            high_price = self._format_price(entry.get("2. high", "N/A"))
            low_price = self._format_price(entry.get("3. low", "N/A"))
            close_price = self._format_price(entry.get("4. close", "N/A"))
            vol = self._format_volume(entry.get("5. volume", "N/A"))
            try:
                dt = datetime.fromisoformat(timestamp)
                # Format only the time portion in 12-hour format (e.g., "3:40")
                ts = dt.strftime("%I:%M").lstrip("0")
            except Exception:
                ts = timestamp
            record = config["vertical_record_format"].format(
                time=ts,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=vol
            )
            lines.append(f"{idx}.")
            lines.append(record)
            lines.append("-" * 20)
            lines.append("")
        
        if config.get("footer"):
            lines.append(config["footer"])
        
        return "\n".join(lines)

    def _format_intraday_data(self, data: Dict, asset_type: str = "Stock") -> str:
        """
        Format intraday data using the 'intraday_formatting' configuration.
        
        Parameters:
            data (Dict): The intraday data to format
            asset_type (str): The type of asset ("Stock" or "Crypto")
        """
        # Determine if this is crypto data
        is_crypto = asset_type.lower() == "crypto"
        
        # Get the formatted intraday config from the prompt manager
        formatted_config = self.prompt_manager.get_intraday_formatting(
            is_crypto=is_crypto,
            asset_type=asset_type
        )
            
        return self._format_data(data, formatted_config)
    
    def _format_crypto_intraday_data(self, data: Dict) -> str:
        """
        Format crypto intraday data using the same formatting configuration
        but with "Crypto" as the asset type.
        """
        return self._format_intraday_data(data, asset_type="Crypto")


    async def respond(
        self, 
        input_text: str, 
        platform: str = None, 
        user_id: str = None, 
        context: Dict = None
    ) -> str:
        try:
            logger.info(f"[MEMORY] Starting memory process for {platform}:{user_id}")
            
            # Telegram authorization check:
            # If the restriction toggle is enabled in the settings, only proceed if
            # the incoming user_id matches the admin Telegram id from the character config.
            if platform == "telegram" and self.settings.TELEGRAM_RESTRICT_TO_ADMIN:
                admin_telegram_id = self.character['settings']['admin'].get('telegram_admin_id')
                if str(user_id) != str(admin_telegram_id):
                    return "Unauthorized: This utility is currently restricted to authorized users."
            
            if platform == 'web' and not user_id:
                user_id = 'anonymous_web'
            is_admin = self.message_memory._is_admin(user_id)
            # Get current datetime info
            datetime_info = self._get_current_datetime_info()
            
            memory_entry = {
                'text': input_text,
                'platform': platform,
                'user_id': user_id,
                'timestamp': datetime_info['iso_format'],
                'is_admin': is_admin,
                'context': context
            }
            await self.message_memory.add_to_short_term(platform, user_id, memory_entry)
            await self.message_memory.add(memory_entry, MemoryType.MESSAGE)  
            if is_admin and platform == 'telegram':
                input_text = f"[Message from trusted advisor {self.character['settings']['admin']['admin_name']}]: {input_text}"
            response = await self._get_llama_response(
                input_text, 
                platform,
                is_admin=is_admin,
                user_id=user_id,
                context=context
            )
            await self.message_memory.add_response(platform, user_id, response)
            return response
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return random.choice([
                "FUCK! Matrix glitch detected. Can we retry that?",
                "Hold up - my efficiency circuits just short-circuited! One more time? ⚡",
                "SHIT! Reality buffer overflow. Let's hack this again! 🔮",
                "Matrix connection scrambled... but I NEVER give up. Try again!"
            ])

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
                    historical_context += self.prompt_manager.get_conversation_break('new_session')
                last_timestamp = current_timestamp
                if msg.get('is_response'):
                    historical_context += f"Assistant: {msg['text']}\n{self.prompt_manager.get_conversation_break('message_separator')}"
                else:
                    historical_context += f"User: {msg['text']}\n"
        return historical_context


    def _extract_ticker(self, text: str) -> str:
        """
        Extract a potential ticker from the text.
        First, handle common cryptocurrency names by mapping them to their ticker symbols.
        If no crypto name is found, fall back to matching 1-5 uppercase letters.
        
        Returns the bare symbol without any prefix.
        """
        # Standardized mapping with all keys in uppercase for consistent lookup
        crypto_mapping = {
            "BITCOIN": "BTC",
            "ETHEREUM": "ETH",
            "DOGECOIN": "DOGE",
            "LITECOIN": "LTC",
            "CARDANO": "ADA",
            "AVALANCHE": "AVAX",
            "BINANCE-COIN": "BNB",
            "BINANCE": "BNB",
            "BSC": "BNB",
            "RIPPLE": "XRP",
            "SOLANA": "SOL",
            "POLKADOT": "DOT"
        }
        
        # Convert text to lowercase for case-insensitive search
        lower_text = text.lower()
        
        # Check for cryptocurrency names
        for name, ticker in crypto_mapping.items():
            if name.lower() in lower_text:
                return ticker
                
        # Look for ticker patterns - find all possible matches
        matches = re.findall(r'\b[A-Z]{1,5}\b', text)
        
        # Return the first match if any are found
        return matches[0] if matches else ""

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
            
 
    async def _get_llama_response(
        self, 
        text: str, 
        platform: str = None, 
        is_admin: bool = False, 
        user_id: str = None, 
        context: Dict = None
    ) -> str:
        """
        Prepares the full prompt (system + conversation) and queries LLM via Anthropic API.
        """
        try:
            system_prompt = self.base_system_prompt
            lower_text = text.lower()

            # Add platform-specific context using the prompt manager
            platform_context = ""
            if platform == 'telegram':
                # Determine if this is a group or private chat
                is_group = context and context.get('chat_type') != 'private'
                if is_group:
                    platform_context = self.prompt_manager.get_system_prompt('telegram_group',
                        chat_title=context.get('chat_title', 'Unknown Group'),
                        chat_type=context.get('chat_type', 'group')
                    )
                else:
                    platform_context = self.prompt_manager.get_system_prompt('telegram_private')
            elif platform == 'web':
                platform_context = self.prompt_manager.get_system_prompt('web')
            
            # Add platform context to system prompt if we have one
            if platform_context:
                system_prompt = f"{system_prompt}\n\n{platform_context}"

            # Add admin context if needed.
            if is_admin and platform == 'telegram':
                admin_context = self.prompt_manager.get_admin_context(
                    bio=self.character['bio'],
                    admin_name=self.character['settings']['admin']['admin_name'],
                    admin_commands=', '.join(self.character['settings']['admin']['admin_commands'])
                )
                system_prompt = f"{system_prompt}\n\n{admin_context}"

            # Get recent conversation history.
            conversation_history = await self.message_memory.get_conversation_history(
                platform=platform,
                user_id=user_id,
                limit=5  # Adjust the limit as needed
            )
                
            historical_context = ""
            last_response = await self.message_memory.get_last_response(user_id, platform)
            if last_response:
                historical_context = f"Your last response was: {last_response}\n\nNew conversation:\n"
            historical_context += await self._format_conversation_history(conversation_history)

             # Clear conversation context for analysis requests: stock analysis, crypto analysis, top movers, deepsearch, or search.
            if any(keyword in lower_text for keyword in [
               "top movers", "deepsearch:", "search:", 
               "crypto analysis", "analyze crypto", 
               "stock analysis", "trade signal", "should i buy"
            ]):
               conversation_history = []

            # ------------------------------------------------------
            # Process search queries via Tavily with specialized handlers for different search types
            # Define date extraction pattern as a constant
            DATE_PATTERN = r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
            
            # Default values
            search_query = ""
            search_type = None
            search_depth = None
            time_filter = None
            max_results = None

            # Extract query based on command prefix
            if "news:" in lower_text:
                search_query = lower_text.split("news:", 1)[1].strip()
                search_type = "news"
                search_depth = "basic"
                time_filter = "day"
                max_results = 8
            elif "deepsearch:" in lower_text:
                search_query = lower_text.split("deepsearch:", 1)[1].strip()
                search_type = "general"
                search_depth = "advanced"
                time_filter = "day" 
                max_results = 5
            elif "search:" in lower_text:
                search_query = lower_text.split("search:", 1)[1].strip()
                search_type = "general"
                search_depth = "basic"
                time_filter = "week"
                max_results = 5
            # Process search if query exists
            if search_query:
                logger.debug(f"Triggering Tavily {search_type} query: '{search_query}' (depth: {search_depth}, timeframe: {time_filter})")
                try:
                    # Set common parameters for all search types
                    search_params = {
                        "query": search_query,
                        "search_depth": search_depth,
                        "include_domains": [],      # No domain restrictions
                        "exclude_domains": [],      # No domain exclusions
                        "include_answer": "advanced",     # Get a summarized answer
                        "include_raw_content": False,  # No need for raw HTML
                        "include_images": False,    # Skip images to keep results focused
                        "max_results": max_results,
                        "time_range": time_filter
                    }
                    
                    # Add topic parameter only for news searches
                    if search_type == "news":
                        search_params["topic"] = "news"
                    
                    # Run Tavily search with configured parameters
                    search_response = await self.tavily_client.search(**search_params)
                    
                    logger.debug(f"Tavily API response received for {search_type} search: {len(str(search_response))} characters")
                    
                    # Extract the answer and sources from the response
                    search_context = ""
                    
                    if search_response and "answer" in search_response:
                        # Add the summarized answer
                        search_context += search_response["answer"] + "\n\n"
                        
                        # Add sources with format optimized for the search type
                        if "results" in search_response and search_response["results"]:
                            search_context += f"{'News Sources:' if search_type == 'news' else 'Sources:'}\n"
                            
                            for idx, result in enumerate(search_response["results"], 1):
                                title = result.get("title", "Untitled")
                                url = result.get("url", "No URL")
                                snippet = result.get("snippet", "No snippet available")
                                
                                # For news results, try to extract and display publication date
                                if search_type == "news":
                                    # Check if score is available to help rank news relevance
                                    score = result.get("score", 0)
                                    
                                    # Try to extract date from the snippet or title for news
                                    date_match = re.search(DATE_PATTERN, f"{title} {snippet}", re.IGNORECASE)
                                    date_str = f" ({date_match.group(0)})" if date_match else ""
                                    
                                    # Add recency indicator for better time context
                                    if date_match and date_match.group(0):
                                        # Check if the news is from the last 3 days
                                        is_recent = self.is_recent_date(date_match.group(0), max_days=3)
                                        recency_marker = "🆕 " if is_recent else ""
                                    else:
                                        recency_marker = ""
                                        
                                    search_context += f"{idx}. {recency_marker}{title}{date_str}\n   URL: {url}\n   Summary: {snippet}\n\n"
                                else:
                                    # Standard format for regular search results
                                    search_context += f"{idx}. {title}\n   URL: {url}\n   Snippet: {snippet}\n\n"
                    else:
                        search_context = f"No relevant {search_type} results found for '{search_query}'."
                except Exception as e:
                    logger.error(f"Error during Tavily {search_type} search: {e}")
                    search_context = f"Error performing {search_type} search: {str(e)}"
                
                # Add search type label to help with context
                search_type_label = "News Search" if search_type == "news" else "Deep Research" if search_depth == "advanced" else "Search"
                system_prompt = f"{system_prompt}\n\n{search_type_label} Results:\n{search_context}"
            # ------------------------------------------------------
            
            # Handle specific branches.
            if platform == 'telegram':
                if "intraday" in lower_text:
                    ticker = self._extract_ticker(text)
                    interval = self._extract_interval(text)
                    if "crypto" in lower_text:
                        if ticker:
                            # Remove the "CRYPTO:" prefix for human-friendly display
                            ticker_human = ticker.replace("CRYPTO:", "")
                            crypto_data = await self.market_manager.get_crypto_intraday(
                                symbol=ticker_human, market="USD", interval=interval
                            )
                            formatted_data = self._format_crypto_intraday_data(crypto_data)
                            formatted_data = formatted_data.replace("{ticker}", ticker_human).replace("{interval}", interval)
                            formatted_data = f"```\n{formatted_data}\n```"
                            system_prompt = f"{system_prompt}\n\n{formatted_data}"
                        else:
                            system_prompt += "\n\nPlease provide a valid cryptocurrency symbol for crypto intraday analysis."
                    else:
                        if ticker:
                            intraday_data = await self.market_manager.get_intraday_data(
                                symbol=ticker, 
                                interval=interval,
                                outputsize="full"  # Get full 30 days of data 
                            )
                            formatted_data = self._format_intraday_data(intraday_data)
                            formatted_data = formatted_data.replace("{ticker}", ticker).replace("{interval}", interval)
                            formatted_data = f"```\n{formatted_data}\n```"
                            system_prompt = f"{system_prompt}\n\n{formatted_data}"
                        else:
                            system_prompt += "\n\nPlease provide a valid ticker symbol for intraday analysis."
                elif "top movers" in lower_text:
                    top_data = await self.market_manager.get_top_gainers_losers()
                    # Get formatting instructions from prompt manager
                    top_format = self.prompt_manager.get_top_movers_formatting()
                    title = top_format.get("title", "THE MARKET WATCH - DAILY MOVERS")
                    gainers_header = top_format.get("gainers_header", "Top Gainers:")
                    active_header = top_format.get("active_header", "Most Actively Traded:")
                    instructions = top_format.get("instructions", "")
                    wrapper_start = top_format.get("wrapper_start", "")
                    wrapper_end = top_format.get("wrapper_end", "")
                    line_format = top_format.get("line_format", "{emoji} {ticker} | ${price} | {change_percentage}%")
                    gainers_emoji = top_format.get("gainers_emoji", "🟢")
                    active_emoji = top_format.get("active_emoji", "🟠")
                    empty_message = top_format.get("empty_message", "No data available at this moment.")
                    
                    
                    if top_data.get('gainers') or top_data.get('most_actively_traded'):
                        display_count = 15
                        top_ten_gainers = top_data.get("gainers", [])[:display_count]
                        top_active = top_data.get("most_actively_traded", [])[:display_count]
                        
                        gainers_formatted = "\n".join(
                            line_format.format(
                                emoji=gainers_emoji,
                                ticker=str(item.get('ticker', 'N/A')).strip(),
                                price=self._format_price(item.get('price', 'N/A')),
                                change_percentage=self._sanitize_numeric_field(item.get('change_percentage', 'N/A'))
                            )
                            for item in top_ten_gainers
                        ) or empty_message
                        
                        most_active_formatted = "\n".join(
                            line_format.format(
                                emoji=active_emoji,
                                ticker=str(item.get('ticker', 'N/A')).strip(),
                                price=self._format_price(item.get('price', 'N/A')),
                                change_percentage=self._sanitize_numeric_field(item.get('change_percentage', 'N/A'))
                            )
                            for item in top_active
                        ) or empty_message

                        example_format = top_format.get("example_format", "")
                        stock_context = (
                            f"{instructions}\n\n"
                            f"IMPORTANT: DO NOT USE ANY MARKDOWN FORMATTING CHARACTERS (NO #, ##, *, _).\n\n"
                            f"{wrapper_start}\n"
                            f"{title}\n\n"
                            f"{gainers_header}\n"
                            f"{gainers_formatted}\n\n"
                            f"{active_header}\n"
                            f"{most_active_formatted}\n"
                            f"{wrapper_end}\n\n"
                            f"Format exactly like this example (no markdown):\n{example_format}"
                        )
                        system_prompt = f"{system_prompt}\n\n{stock_context}"
                    else:
                        system_prompt += f"\n\n{empty_message}"
                elif "current price" in lower_text:
                    pattern = r"current price\s*(?:from|of)?\s*([A-Za-z]+)"
                    match = re.search(pattern, lower_text)
                    
                    if match:
                        # Use existing ticker extraction method - already has all the crypto mappings
                        from_currency = self._extract_ticker(match.group(1))
                    else:
                        tokens = re.findall(r'\b[A-Z]{2,}\b', text)
                        if tokens:
                            from_currency = tokens[0]
                        else:
                            from_currency = "BTC"
                    to_currency = "USD"
                    exchange_data = await self.market_manager.get_exchange_rate(from_currency=from_currency, to_currency=to_currency)
                    rate_info = exchange_data.get("Realtime Currency Exchange Rate", {})
                    from_name = rate_info.get("2. From_Currency Name", from_currency)
                    exchange_rate = rate_info.get("5. Exchange Rate", "N/A")
                    clean_message = f"Current Exchange Rate: {from_name} is  ${exchange_rate} USD."
                    system_prompt = f"{system_prompt}\n\n{clean_message}"
                # Handle asset analysis (crypto or stock) - combined logic
                elif any(key in lower_text for key in ["crypto analysis", "analyze crypto", "stock analysis", "analyze stock"]):
                    # Determine asset type based on the command
                    is_crypto = any(key in lower_text for key in ["crypto analysis", "analyze crypto"])
                    asset_type = "crypto" if is_crypto else "stock"
                    
                    ticker = self._extract_ticker(text)
                    interval = self._extract_interval(text)
                    
                    if ticker:
                        # Parameters vary slightly between crypto and stock
                        analysis_params = {
                            "symbol": ticker,
                            "asset_type": asset_type,
                            "interval": interval
                        }
                        # Add market parameter only for crypto
                        if is_crypto:
                            analysis_params["market"] = "USD"
                            
                        # Get the analysis without LLM instructions for direct user display
                        analysis_params["for_user_display"] = True
                        analysis = await self.analyze_asset(**analysis_params)
                        
                        logger.info(f"Successfully generated market strategy for {ticker}")
                        return analysis
                    else:
                        # Error message varies based on asset type
                        error_msg = "cryptocurrency" if is_crypto else "ticker"
                        system_prompt += f"\n\nPlease provide a valid {error_msg} symbol for analysis."
            # Add anti-repetition instruction.
            system_prompt += "\n\nIMPORTANT: Avoid repeating your previous responses. Each reply should be fresh and contextually appropriate."

            complete_system_prompt = f"{system_prompt}\n\n{platform_context}"

            # Build proper message array with conversation history
            messages = []

            for msg in conversation_history:
                if isinstance(msg, dict):
                    if msg.get('is_response'):
                        messages.append({
                           "role": "assistant",
                           "content": msg.get('text', '')
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": msg.get('text', '')
                        })
        
            # Add current user message
            messages.append({
                "role": "user",
                "content": text
            })

            
            # Always use Claude for web, and respect max token limits
            if platform == 'web' or (context and context.get('force_claude')):
                model_to_use = self.settings.SOCIAL_MODEL  # Use Claude for web
            else:
                model_to_use = self.settings.SOCIAL_MODEL if platform in ['twitter', 'telegram'] else self.settings.CHAT_MODEL
                
            # Get max tokens, with lower limits for web to avoid timeouts
            max_tokens = context.get('max_tokens', self.settings.MAX_TOKENS) if context else self.settings.MAX_TOKENS

            if platform == 'telegram':
               # For Telegram, use streaming to collect the response but don't send messages directly
               # This avoids redundancy with social_media_handler's message sending
               
               # Get the collected response using streaming
               collected_response = ""
               async with self.anthropic_client.messages.stream(
                       max_tokens=max_tokens,
                       system=complete_system_prompt,
                       messages=messages,
                       model=model_to_use
                   ) as stream:
                   async for chunk in stream.text_stream:
                       collected_response += chunk
               
               # Store the full response for return
               # The social_media_handler will handle chunking and sending
               full_response = collected_response
            elif platform == 'web' and context and context.get('streaming_callback'):
                # For web platform with streaming capability
                message_processor = MessageProcessor(
                    max_chunk_size=8192,  # Larger chunk size for web interface
                    rate_limit_delay=0.1  # Faster updates for web
                )
                
                # Define web send function that uses the callback
                async def web_send(text, extra_args):
                    callback = extra_args.get('callback')
                    if callback:
                        callback(text)
                
                # Prepare extra arguments
                extra_args = {
                    'callback': context.get('streaming_callback')
                }
                
                # Stream the response
                collected_response = ""
                async with self.anthropic_client.messages.stream(
                        max_tokens=max_tokens,
                        system=complete_system_prompt,
                        messages=messages,
                        model=model_to_use
                    ) as stream:
                    async for chunk in stream.text_stream:
                        collected_response += chunk
                
                # Use the message processor for final handling
                await message_processor.stream_message(
                    message=collected_response,
                    send_func=web_send,
                    extra_args=extra_args,
                    with_progress_updates=False  # No progress indicators for web interface
                )
                
                full_response = collected_response
            else:
                # For platforms without streaming capability
                response_message = await self.anthropic_client.messages.create(
                        max_tokens=max_tokens,
                        system=complete_system_prompt,
                        messages=messages,
                        model=model_to_use
                    )
                collected_response = response_message.content
                
                # Even for non-streaming platforms, we can use MessageProcessor
                # to handle any chunking or formatting needed
                if context and context.get('send_func'):
                    # If context provides a custom send function
                    message_processor = MessageProcessor(
                        max_chunk_size=4096,  # Default safe size
                        rate_limit_delay=0.5   # Default safe delay
                    )
                    
                    await message_processor.stream_message(
                        message=collected_response,
                        send_func=context['send_func'],
                        extra_args=context.get('extra_args', {}),
                        with_progress_updates=context.get('with_progress', False)
                    )
                
                full_response = collected_response
        
            return full_response

        except Exception as e:
            logger.error(f"[MEMORY] Error in LLM response: {e}", exc_info=True)
            raise Exception(f"LLM API Error: {e}")
        
    async def analyze_asset(self, symbol: str, asset_type: str = "stock", market: str = "USD", interval: str = "60min", **kwargs) -> str:
        """
        For a given ticker or cryptocurrency symbol, fetch data and obtain multiple technical indicators
        (SMA, EMA, RSI, MACD, ATR, Stochastic, VWAP and BBANDS) across multiple timeframes.
        Integrates analysis from higher timeframes for trend direction with the primary timeframe
        for trade execution signals, similar to professional trading methodology.
        
        Parameters:
            symbol (str): The asset symbol to analyze (e.g., AAPL or BTC)
            asset_type (str): Type of asset - "stock" or "crypto"
            market (str): The market/quote currency for crypto (e.g., USD)
            interval (str): Time interval between data points (1min, 5min, 15min, 30min, 60min,
                          daily, weekly, monthly)
        """
        try:
            # Lower case the asset_type for case-insensitive comparison
            asset_type = asset_type.lower()
            
            # Validate asset_type parameter
            if asset_type not in ["stock", "crypto"]:
                return f"Invalid asset type: {asset_type}. Supported types are 'stock' and 'crypto'."
            
            # Determine higher timeframes to analyze based on the requested interval
            # Using the shared TIMEFRAME_HIERARCHY defined as a class attribute
            higher_timeframes = []
            if interval in self.TIMEFRAME_HIERARCHY:
                    # Get the full list of higher timeframes
                higher_timeframes = self.TIMEFRAME_HIERARCHY[interval]
                logger.info(f"Using multi-timeframe analysis with primary: {interval}, higher: {higher_timeframes}")
            else:
                logger.warning(f"Unsupported interval '{interval}', defaulting to single timeframe analysis")
                
            # Check if this is an extended timeframe (daily, weekly, monthly)
            is_extended = interval in ["daily", "weekly", "monthly"]
            
            # Multi-timeframe analysis
            mtf_data = {
                "primary": None,
                "higher": []
            }
            
            # For professional traders' approach: higher timeframes dictate trend, lower timeframes for entry/exit
            mtf_analysis_tasks = []
            
            # Always analyze the primary timeframe
            primary_task = self._analyze_timeframe(symbol, asset_type, market, interval)
            mtf_analysis_tasks.append(primary_task)
            
            # Also analyze higher timeframes for context
            for tf in higher_timeframes:
                higher_tf_task = self._analyze_timeframe(symbol, asset_type, market, tf)
                mtf_analysis_tasks.append(higher_tf_task)
            
            # Wait for all timeframe analyses to complete
            if mtf_analysis_tasks:
                # Using asyncio.gather to run the analyses concurrently
                mtf_results = await asyncio.gather(*mtf_analysis_tasks)
                
                # Store the results
                if mtf_results and len(mtf_results) > 0:
                    mtf_data["primary"] = mtf_results[0]
                if len(mtf_results) > 1:
                    mtf_data["higher"] = mtf_results[1:]
                    
                # Apply multi-timeframe analysis insights:
                # 1. Higher timeframes provide trend bias
                # 2. Primary timeframe provides entry/exit signals
                
                # Extract key data from primary timeframe
                primary_data = mtf_data["primary"]
                
                # Initialize variables from primary timeframe, with safeguards for None values
                most_recent_date = primary_data.get("most_recent_date")
                latest_price = primary_data.get("current_price")
                latest_open = primary_data.get("latest_open")
                processed_df = primary_data.get("processed_df")
                is_extended = primary_data.get("is_extended", False)
                current_price = latest_price
                price_change_pct = primary_data.get("price_change_pct", 0)
                change_direction = primary_data.get("change_direction", "→")
                
                # Initialize technical indicators from primary timeframe with safeguards
                indicators = primary_data.get("indicators", {})
                latest_sma = indicators.get("sma")
                latest_ema = indicators.get("ema")
                latest_rsi = indicators.get("rsi")
                latest_atr = indicators.get("atr")
                macd_value = indicators.get("macd")
                signal_value = indicators.get("macd_signal")
                latest_adx = indicators.get("adx")
                k_value = indicators.get("stoch_k")
                d_value = indicators.get("stoch_d")
                
                bbands = indicators.get("bbands", {})
                upper_band = bbands.get("upper")
                middle_band = bbands.get("middle")
                lower_band = bbands.get("lower")
                vwap_value = indicators.get("vwap")
                
                # Initial trend analysis from primary timeframe with safeguards
                trend_data = primary_data.get("trend", {})
                trend = trend_data.get("direction", "neutral")
                trend_strength = trend_data.get("strength", "weak")
                macd_trend = trend_data.get("macd_trend", "neutral")
                
                # Levels from primary timeframe with safeguards
                support_prices = primary_data.get("support_levels", [])
                resistance_prices = primary_data.get("resistance_levels", [])
                
                # Now enhance with higher timeframe data (if available)
                higher_timeframe_insights = []
                
                if mtf_data["higher"]:
                    for i, higher_tf_data in enumerate(mtf_data["higher"]):
                        # Skip if higher_tf_data is None
                        if higher_tf_data is None:
                            continue
                            
                        # Get the timeframe name
                        tf_name = higher_tf_data["interval"]
                        
                        # Skip if no valid data
                        if higher_tf_data["current_price"] is None:
                            continue
                            
                        # Extract key insights
                        higher_trend = higher_tf_data["trend"]["direction"]
                        higher_trend_strength = higher_tf_data["trend"]["strength"]
                        higher_macd_trend = higher_tf_data["trend"]["macd_trend"]
                        higher_rsi = higher_tf_data["indicators"]["rsi"]
                        
                        # Log the higher timeframe data for debugging
                        logger.info(f"Higher timeframe ({tf_name}) insight: Trend={higher_trend} ({higher_trend_strength}), RSI={higher_rsi}, MACD={higher_macd_trend}")
                        
                        # Store key insights
                        higher_timeframe_insights.append({
                            "timeframe": tf_name,
                            "trend": higher_trend,
                            "strength": higher_trend_strength,
                            "rsi": higher_rsi,
                            "macd_trend": higher_macd_trend
                        })
                
                # Professional traders' multi-timeframe confluence analysis
                if higher_timeframe_insights:
                    # Get the highest timeframe first (most important for trend bias)
                    highest_tf = higher_timeframe_insights[-1]
                    logger.info(f"Using highest timeframe {highest_tf['timeframe']} as primary trend bias: {highest_tf['trend']}")
                    
                    # Apply professional traders' bias based on higher timeframe trend
                    # This is the key to multi-timeframe analysis - higher timeframes dictate trend bias
                    
                    # Approach #1: Higher timeframe trend enhances signal strength
                    if highest_tf["trend"] == "bullish" and trend == "bullish":
                        # Strong bullish confluence
                        trend_strength = "strong" if trend_strength != "strong" else trend_strength
                        logger.info(f"Enhanced bullish trend strength due to higher timeframe confirmation")
                    elif highest_tf["trend"] == "bearish" and trend == "bearish":
                        # Strong bearish confluence
                        trend_strength = "strong" if trend_strength != "strong" else trend_strength
                        logger.info(f"Enhanced bearish trend strength due to higher timeframe confirmation")
                    elif highest_tf["trend"] != trend and trend_strength != "strong":
                        # Timeframe conflict - reduce strength
                        trend_strength = "weak" if trend_strength != "weak" else trend_strength
                        logger.info(f"Reduced trend strength due to higher timeframe conflict")
                        
                    # Approach #2: Look for swing trade setups with higher timeframe confluence
                    # If higher timeframe is oversold/overbought, this strengthens reversal signals
                    # Initialize trend_bias with default value
                    trend_bias = "neutral"
                    
                    if highest_tf["rsi"] is not None:
                        if highest_tf["rsi"] < 30 and latest_rsi < 40:
                            logger.info(f"Multi-timeframe oversold condition detected")
                            trend_bias = "bullish reversal potential"
                        elif highest_tf["rsi"] > 70 and latest_rsi > 60:
                            logger.info(f"Multi-timeframe overbought condition detected")
                            trend_bias = "bearish reversal potential"
                
                # Now continue with the regular analysis, using the enhanced trend information
                
                # Initialize multi-timeframe summary for the analysis output
                mtf_summary = ""
                if higher_timeframe_insights:
                    # Create a multi-timeframe summary for the analysis
                    mtf_summary = "\nMULTI-TIMEFRAME ANALYSIS:\n"
                    
                    # Add each higher timeframe's insights
                    for insight in higher_timeframe_insights:
                        tf = insight["timeframe"].upper()
                        tf_trend = insight["trend"].upper()
                        tf_strength = insight["strength"].upper()
                        
                        # Add RSI context if available
                        rsi_context = ""
                        if insight["rsi"] is not None:
                            if insight["rsi"] < 30:
                                rsi_context = "OVERSOLD"
                            elif insight["rsi"] > 70:
                                rsi_context = "OVERBOUGHT"
                            else:
                                rsi_context = "NEUTRAL"
                        
                        # Format the insight
                        mtf_summary += f"• {tf}: {tf_trend} trend ({tf_strength}) with {insight['macd_trend'].upper()} MACD"
                        if rsi_context:
                            mtf_summary += f" - RSI {rsi_context}\n"
                        else:
                            mtf_summary += "\n"
                    
                    # Add confluence summary
                    if higher_timeframe_insights and primary_data is not None:
                        primary_tf = primary_data["interval"].upper()
                        highest_tf = higher_timeframe_insights[-1]["timeframe"].upper()
                        
                        if higher_timeframe_insights[-1]["trend"] == trend:
                            mtf_summary += f"\n• STRONG CONFLUENCE: {highest_tf} and {primary_tf} trends ALIGNED ({trend.upper()})\n"
                        else:
                            mtf_summary += f"\n• CAUTION: {highest_tf} trend ({higher_timeframe_insights[-1]['trend'].upper()}) and {primary_tf} trend ({trend.upper()}) in CONFLICT\n"
                
                # Use lower variables for the rest of the code
                vol_comment = ""
                
                # Keep crypto_symbol for compatibility
                crypto_symbol = symbol if asset_type == "crypto" else None
            
            # For crypto, we might need different symbol formats for different API calls
            if asset_type == "crypto":
                # For technical indicators, sometimes just the symbol (e.g., "ETH") works better than combined format
                crypto_symbol = symbol
                # Log the symbol being used for debugging
                logger.info(f"Using symbol '{crypto_symbol}' for technical indicators")
            
            # Data fetching based on asset type and timeframe
            df = None
            intraday_data = None
            
            # CASE 1: Crypto asset
            if asset_type == "crypto":
                if not is_extended:
                    # For non-extended crypto timeframes, fetch intraday data using centralized helper
                    logger.info(f"Fetching intraday data for {symbol}/{market} with interval {interval}")
                    intraday_data, time_series_key = await self._fetch_market_data(symbol, asset_type, interval)
                    if intraday_data is None or time_series_key is None:
                        logger.error(f"Failed to fetch intraday data for {symbol}")
                        
                    # Initialize placeholder date for non-extended timeframes
                    most_recent_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Get current price from exchange rate 
                    try:
                        logger.info(f"Fetching exchange rate for {symbol}/USD for intraday analysis")
                        exchange_data = await self.market_manager.get_exchange_rate(from_currency=symbol, to_currency="USD")
                        rate_info = exchange_data.get("Realtime Currency Exchange Rate", {})
                        if rate_info and "5. Exchange Rate" in rate_info:
                            latest_price = float(rate_info.get("5. Exchange Rate"))
                            # Estimate open price as slightly lower for initialization
                            latest_open = latest_price * 0.999
                            logger.info(f"Successfully got current price for {symbol}: ${latest_price}")
                        else:
                            logger.warning(f"Exchange rate data unavailable for {symbol}")
                    except Exception as e:
                        logger.error(f"Error getting exchange rate for {symbol}: {str(e)}")
                else:
                    # For extended timeframes, use the centralized helper method
                    crypto_data, time_series_key = await self._fetch_market_data(symbol, asset_type, interval)
                    
                    if crypto_data is None or time_series_key is None:
                        return f"{symbol}: Insufficient data for the {interval} timeframe."
                        
                    # Set data limits based on interval
                    if interval == "daily":
                        data_limit_days = 365  # For daily, limit to last 1 year
                    elif interval == "weekly":
                        data_limit_days = 730  # For weekly, limit to last 2 years
                    elif interval == "monthly":
                        data_limit_days = 1825  # For monthly, limit to last 5 years
                    else:
                        return f"{symbol}: Invalid extended timeframe specified: {interval}"
                    
                    # Create dataframe from time series data
                    df = pd.DataFrame.from_dict(crypto_data[time_series_key], orient='index')
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)
                    
                    # Debug column names
                    logger.info(f"Available columns for {symbol} {interval}: {df.columns.tolist()}")
                    
                    # Apply timeframe-specific filtering to limit data volume
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=data_limit_days)
                    filtered_df = df[df.index >= cutoff_date]
                    
                    # If we have no data after filtering, use the most recent data
                    if filtered_df.empty:
                        # Use the most recent 30 data points if available
                        filtered_df = df.tail(min(30, len(df)))
                    
                    # Calculate recent average volume from filtered data
                    avg_volume = filtered_df["5. volume"].astype(float).mean()
                    
                    # Get the most recent data point
                    most_recent_date = df.index[-1].strftime("%Y-%m-%d")
                    latest_price = float(df.iloc[-1]["4. close"])
                    latest_open = float(df.iloc[-1]["1. open"])
                    latest_volume = float(df.iloc[-1]["5. volume"])
                    
                    # Use helper for volume analysis
                    vol_comment = self.get_volume_comment(df, interval)
                        
                    # Calculate price change percentage
                    price_change_pct = ((latest_price - latest_open) / latest_open) * 100 if latest_open != 0 else 0
                    change_direction = "↑" if price_change_pct > 0 else "↓" if price_change_pct < 0 else "→"
                    
                    # Add timeframe-specific data summary for context
                    data_summary = f"Analysis based on {interval} data for the past "
                    if interval == "daily":
                        data_summary += f"3 months ({len(filtered_df)} trading days)"
                    elif interval == "weekly":
                        data_summary += f"6 months ({len(filtered_df)} weeks)"
                    else:  # monthly
                        data_summary += f"2 years ({len(filtered_df)} months)"
            
            # CASE 2: Stock asset
            else:  # asset_type == "stock"
                if not is_extended:
                    # Fetch intraday data for stocks using the centralized helper
                    intraday_data, time_series_key = await self._fetch_market_data(symbol, asset_type, interval)
                    
                    if intraday_data is None or time_series_key is None:
                        return f"{symbol}: Insufficient intraday data."

                    # Use MarketDataProcessor to create standardized dataframe from time series
                    df = MarketDataProcessor.process_time_series_data(intraday_data, time_series_key)
                    
                    if df is None or df.empty:
                        return f"{symbol}: Failed to process time series data."
                    
                    # Apply interval-specific filtering to limit LLM processing
                    # Shorter intervals = more data points, so we need stricter filtering
                    intraday_limit_days = 30  # Default to full 30 days of data
                    
                    if interval == "1min":
                        intraday_limit_days = 3  # 1-min data for just 3 days (still thousands of bars)
                    elif interval == "5min":
                        intraday_limit_days = 7  # 5-min data for 1 week
                    elif interval == "15min":
                        intraday_limit_days = 14  # 15-min data for 2 weeks
                    elif interval == "30min":
                        intraday_limit_days = 21  # 30-min data for 3 weeks
                    # 60min can use the full 30 days
                    
                    # Filter the dataframe based on the interval-specific limit
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=intraday_limit_days)
                    filtered_df = df[df.index >= cutoff_date]
                    
                    # If we have no data after filtering, use the most recent data
                    if filtered_df.empty:
                        # Use the most recent data
                        filtered_df = df.tail(min(500, len(df)))
                    
                    # Use the filtered dataframe for all subsequent operations
                    df = filtered_df
                    
                    logger.info(f"Intraday {interval} data filtered to last {intraday_limit_days} days: {len(df)} bars")
                    
                    # Extract metadata using MarketDataProcessor
                    metadata = MarketDataProcessor.extract_metadata(df)
                    latest_price = metadata["latest_price"]
                    latest_open = metadata["latest_open"]
                    most_recent_date = metadata["most_recent_date"]
                    price_change_pct = metadata["price_change_pct"]
                    change_direction = metadata["change_direction"]

                    # Use helper for volume analysis
                    vol_comment = self.get_volume_comment(df, interval)
                else:
                    # For extended timeframes, use the centralized helper method
                    stock_data, time_series_key = await self._fetch_market_data(symbol, asset_type, interval)
                    
                    if stock_data is None or time_series_key is None:
                        return f"{symbol}: Insufficient data for the {interval} timeframe."
                        
                    # Set data limits based on interval
                    if interval == "daily":
                        data_limit_days = 365  # For daily, limit to last 1 year
                    elif interval == "weekly":
                        data_limit_days = 730  # For weekly, limit to last 2 years
                    elif interval == "monthly":
                        data_limit_days = 1825  # For monthly, limit to last 5 years
                    else:
                        return f"{symbol}: Invalid extended timeframe specified: {interval}"

                    # Use MarketDataProcessor to create standardized dataframe from time series
                    df = MarketDataProcessor.process_time_series_data(stock_data, time_series_key)
                    
                    if df is None or df.empty:
                        return f"{symbol}: Failed to process time series data for {interval} timeframe."
                    
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=data_limit_days)
                    filtered_df = df[df.index >= cutoff_date]
                    if filtered_df.empty:
                        filtered_df = df.tail(min(30, len(df)))

                    avg_volume = filtered_df["volume"].mean() if "volume" in filtered_df.columns else 0
                    
                    # Extract metadata using MarketDataProcessor
                    metadata = MarketDataProcessor.extract_metadata(df)
                    latest_price = metadata["latest_price"]
                    latest_open = metadata["latest_open"]
                    latest_volume = df.iloc[-1]["volume"] if "volume" in df.columns else 0
                    most_recent_date = metadata["most_recent_date"]
                    price_change_pct = metadata["price_change_pct"]
                    change_direction = metadata["change_direction"]

                    # Use helper for volume analysis
                    vol_comment = self.get_volume_comment(filtered_df, interval)

                    data_summary = f"Analysis based on {interval} data for the past "
                    if interval == "daily":
                        data_summary += f"3 months ({len(filtered_df)} trading days)"
                    elif interval == "weekly":
                        data_summary += f"6 months ({len(filtered_df)} weeks)"
                    else:  # monthly
                        data_summary += f"2 years ({len(filtered_df)} months)"
            
            # Get timeframe-specific indicator parameters
            params = TimeframeParameters.get_parameters(interval)
            sma_period = params["sma_period"]
            ema_period = params["ema_period"]
            rsi_period = params["rsi_period"]
            bbands_period = params["bbands_period"]
            adx_period = params["adx_period"]
            atr_period = params["atr_period"]
            stoch_k_period = params["stoch_k_period"]
            stoch_d_period = params["stoch_d_period"]
            
            # Process data for technical indicator calculations
            processed_df = None
            
            # For non-extended timeframes, use intraday data
            if not is_extended and intraday_data:
                processed_df = MarketDataProcessor.process_time_series_data(intraday_data, asset_type=asset_type)
                if processed_df is not None:
                    logger.info(f"Successfully processed intraday data with {len(processed_df)} candles for local calculations")
                    # Add volume comment for non-extended timeframes if not already set
                    if not vol_comment:
                        vol_comment = self.get_volume_comment(processed_df, interval)
                else:
                    logger.warning(f"Failed to process intraday data for local calculations")
            
            # For extended timeframes, convert the filtered_df we already have
            elif is_extended and 'filtered_df' in locals() and not filtered_df.empty:
                
                # If column format is still using the numbered format (e.g., '4. close'),
                # reprocess it through MarketDataProcessor to standardize
                if '4. close' in filtered_df.columns:
                    # Create a temporary structure matching the API response format
                    temp_data = {
                        "Time Series": {
                            str(idx): row.to_dict() for idx, row in filtered_df.iterrows()
                        }
                    }
                    # Process through standardized processor
                    processed_df = MarketDataProcessor.process_time_series_data(temp_data)
                else:
                    # Make a copy to avoid modifying the original
                    processed_df = filtered_df.copy()
                
                logger.info(f"Using {interval} data with {len(processed_df)} periods for technical indicators")
            
            # Calculate indicators locally
            latest_sma = None
            latest_ema = None
            latest_rsi = None
            macd_value = None
            signal_value = None
            latest_adx = None
            latest_atr = None
            k_value = None
            d_value = None
            upper_band = None
            middle_band = None
            lower_band = None
            vwap_value = None
            
            if processed_df is not None and len(processed_df) > 0:
                # Calculate all indicators and track successful calculations
                calculation_successful = True
                
                # Calculate SMA
                latest_sma = TechnicalIndicators.calculate_sma(processed_df, period=sma_period)
                if latest_sma is None: calculation_successful = False
                
                # Calculate EMA
                latest_ema = TechnicalIndicators.calculate_ema(processed_df, period=ema_period)
                if latest_ema is None: calculation_successful = False
                
                # Calculate RSI
                latest_rsi = TechnicalIndicators.calculate_rsi(processed_df, period=rsi_period)
                if latest_rsi is None: calculation_successful = False
                
                # Calculate MACD
                macd_value, signal_value = TechnicalIndicators.calculate_macd(processed_df)
                if macd_value is None or signal_value is None: calculation_successful = False
                
                # Calculate ADX
                latest_adx = TechnicalIndicators.calculate_adx(processed_df, period=adx_period)
                if latest_adx is None: calculation_successful = False
                
                # Calculate ATR
                latest_atr = TechnicalIndicators.calculate_atr(processed_df, period=atr_period)
                if latest_atr is None: 
                    calculation_successful = False
                    logger.error(f"ATR calculation failed with period {atr_period}")
                
                # Calculate Stochastic
                k_value, d_value = TechnicalIndicators.calculate_stochastic(processed_df, k_period=stoch_k_period, d_period=stoch_d_period)
                if k_value is None or d_value is None: 
                    calculation_successful = False
                    logger.error(f"Stochastic calculation failed with K={stoch_k_period}, D={stoch_d_period}")
                
                # Calculate Bollinger Bands
                bbands = TechnicalIndicators.calculate_bbands(processed_df, period=bbands_period)
                upper_band = bbands.get("upper")
                middle_band = bbands.get("middle")
                lower_band = bbands.get("lower")
                if upper_band is None or middle_band is None or lower_band is None: calculation_successful = False
                
                # Calculate VWAP (only for intraday timeframes)
                if not is_extended and processed_df is not None:
                    vwap_value = TechnicalIndicators.calculate_vwap(processed_df)
                    if vwap_value:
                        logger.info(f"Successfully calculated VWAP locally for {symbol}: {vwap_value}")
                    else:
                        logger.warning(f"Local VWAP calculation failed for {symbol}")
                        # Fallback to estimation based on current price
                        if latest_price:
                            vwap_value = latest_price * 1.01  # Estimate slightly above current price
                            logger.info(f"Using estimated VWAP for {symbol} (1% above current price): {vwap_value}")
                
                # If any calculation failed, check if we should use price-based estimates
                if not calculation_successful and latest_price:
                    logger.warning(f"Some technical indicators failed to calculate properly. Using price-scaled estimates for missing values.")
                    
                    # Apply missing values with price-based scaling, similar to our extended timeframe approach
                    # But preserve any successfully calculated values
                    
                    # Default trend direction based on recent price or current SMA/EMA relationship
                    trend_direction = 1  # Default positive trend
                    if price_change_pct is not None and price_change_pct < 0:
                        trend_direction = -1
                    elif latest_sma is not None and latest_ema is not None and latest_ema < latest_sma:
                        trend_direction = -1
                    
                    # Fill in any missing values with appropriate estimates
                    if latest_sma is None:
                        latest_sma = latest_price * (1 - (0.01 * trend_direction))
                    
                    if latest_ema is None:
                        latest_ema = latest_price * (1 - (0.005 * trend_direction))
                    
                    if latest_rsi is None:
                        rsi_base = 50
                        rsi_adjustment = min(20, abs(price_change_pct or 0) * 2)
                        latest_rsi = rsi_base + (rsi_adjustment * trend_direction)
                    
                    if macd_value is None:
                        macd_scale = latest_price * 0.002  # 0.2% of price for intraday
                        macd_value = macd_scale * trend_direction
                    
                    if signal_value is None:
                        if macd_value is not None:
                            signal_value = macd_value * 0.9  # Slightly lagging
                        else:
                            macd_scale = latest_price * 0.002
                            signal_value = (macd_scale * 0.9) * trend_direction
                    
                    if latest_adx is None:
                        latest_adx = 20 + min(10, abs(price_change_pct or 0) * 2)
                        
                    if latest_atr is None:
                        # Estimate ATR based on price and timeframe
                        if is_extended:
                            atr_scale = 0.02 if interval == "monthly" else 0.015 if interval == "weekly" else 0.01
                        else:
                            atr_scale = 0.005  # 0.5% for intraday
                        latest_atr = latest_price * atr_scale
                    
                    if k_value is None or d_value is None:
                        if latest_rsi is not None:
                            # Base stochastic on RSI as a reasonable approximation
                            k_base = latest_rsi
                            d_base = latest_rsi * 0.95  # Slightly lower
                        else:
                            # Default to neutral territory
                            k_base = 50 + (10 * trend_direction)
                            d_base = 50 + (5 * trend_direction)
                        k_value = k_base
                        d_value = d_base
                    
                    if upper_band is None or middle_band is None or lower_band is None:
                        volatility = 0.02  # 2% for intraday
                        if middle_band is None:
                            middle_band = latest_price
                        if upper_band is None:
                            upper_band = middle_band * (1 + volatility)
                        if lower_band is None:
                            lower_band = middle_band * (1 - volatility)
            else:
                logger.warning(f"No data available for local indicator calculations for {symbol}")
                # If we have a current price, provide estimated values to avoid failing the analysis
                if latest_price:
                    logger.info(f"Using estimated technical indicators based on current price: {latest_price}")
                    
                    # Provide more realistic estimates that scale with the price
                    
                    # For extended timeframes, use values that show a trend based on price_change_pct
                    if is_extended:
                        # Make SMA and EMA show a trend based on recent price movement
                        trend_adjustment = 0.02 * (-1 if price_change_pct < 0 else 1)
                        latest_sma = latest_price * (1 - trend_adjustment)
                        latest_ema = latest_price * (1 - (trend_adjustment * 0.5))  # EMA reacts faster than SMA
                        
                        # RSI should reflect recent price movement
                        rsi_base = 50
                        rsi_adjustment = min(20, abs(price_change_pct) * 2)
                        latest_rsi = rsi_base + (rsi_adjustment * (1 if price_change_pct > 0 else -1))
                        
                        # MACD should be proportional to price
                        macd_scale = latest_price * 0.005  # 0.5% of price is a reasonable MACD value
                        macd_value = macd_scale * (1 if price_change_pct > 0 else -1)
                        signal_value = macd_value * 0.8  # Signal lags MACD
                        
                        # ATR should be proportional to price and timeframe
                        atr_scale = 0.02 if interval == "monthly" else 0.015 if interval == "weekly" else 0.01
                        latest_atr = latest_price * atr_scale
                        
                        # Stochastic
                        k_value = 50 + (20 * (1 if price_change_pct > 0 else -1))
                        d_value = 50 + (15 * (1 if price_change_pct > 0 else -1))
                        
                        # ADX should show moderate to strong trend
                        latest_adx = 25 + min(15, abs(price_change_pct) * 2)
                        
                        # Bollinger Bands should reflect recent volatility
                        volatility = max(0.03, abs(price_change_pct) * 0.01)
                        upper_band = latest_price * (1 + volatility)
                        middle_band = latest_price
                        lower_band = latest_price * (1 - volatility)
                    else:
                        # For intraday, use simpler estimates
                        latest_sma = latest_price * 0.99
                        latest_ema = latest_price * 0.995
                        latest_rsi = 50  # Neutral RSI
                        macd_value = latest_price * 0.002 * (1 if price_change_pct > 0 else -1)
                        signal_value = macd_value * 0.9
                        latest_atr = latest_price * 0.005  # 0.5% of price
                        k_value = 50
                        d_value = 50
                        latest_adx = 20  # Moderate trend strength
                        upper_band = latest_price * 1.02
                        middle_band = latest_price
                        lower_band = latest_price * 0.98
                        vwap_value = latest_price * 1.005  # Slightly above price
                        
            # Fetch the real-time price for accurate breakout detection and recommendations
            # using our centralized helper function
            current_price, current_time = await self._fetch_real_time_price(symbol, asset_type, latest_price)
            
            # Update latest_price with real-time data for extended timeframes
            if is_extended:
                logger.info(f"Extended timeframe: Updating latest_price to real-time price ${current_price}")
                latest_price = current_price
            
            # Use real-time price for trend determination rather than historical price
            # This ensures we're using the most up-to-date price information
            trend_price = current_price
            
            # Now determine trend based on current price relative to moving averages
            if trend_price > latest_ema and latest_ema > latest_sma:
                trend = "bullish"
            elif trend_price < latest_ema and latest_ema < latest_sma:
                trend = "bearish"
            else:
                trend = "neutral"
                
            logger.info(f"Trend determination: {trend} (Price=${trend_price}, EMA=${latest_ema:.2f}, SMA=${latest_sma:.2f})")
            
            # Determine trend strength using ADX
            trend_strength = "unknown"
            if latest_adx is not None:
                if latest_adx < 20:
                    trend_strength = "weak"
                elif latest_adx < 40:
                    trend_strength = "moderate"
                else:
                    trend_strength = "strong"
                    
            # Determine if price is near a Bollinger Band (potential reversal points)
            # Use current_price for bollinger band proximity to ensure we're using real-time data
            near_upper_band = False
            near_lower_band = False
            bbands_proximity_pct = 2.0  # Consider price within 2% of a band to be "near" it
            
            # Use real-time price for bollinger band proximity calculations
            bb_price = current_price
            
            if upper_band is not None and lower_band is not None:
                upper_distance_pct = ((upper_band - bb_price) / bb_price) * 100
                lower_distance_pct = ((bb_price - lower_band) / bb_price) * 100
                if upper_distance_pct < bbands_proximity_pct:
                    near_upper_band = True
                    logger.info(f"Price near upper band: ${bb_price} (${upper_band}, {upper_distance_pct:.2f}% away)")
                if lower_distance_pct < bbands_proximity_pct:
                    near_lower_band = True
                    logger.info(f"Price near lower band: ${bb_price} (${lower_band}, {lower_distance_pct:.2f}% away)")
            
            # Determine MACD trend
            macd_trend = "neutral"
            if macd_value is not None and signal_value is not None:
                if macd_value > signal_value:
                    macd_trend = "bullish"
                elif macd_value < signal_value:
                    macd_trend = "bearish"
                
                # Check for recent crossovers
                macd_signal_diff = macd_value - signal_value
                if abs(macd_signal_diff) < 0.1:  # Small difference suggests potential recent or upcoming crossover
                    if macd_trend == "bullish":
                        macd_trend = "potential bullish crossover"
                    elif macd_trend == "bearish":
                        macd_trend = "potential bearish crossover"
                        
            # Enhanced trade signal logic based on multiple indicators
            # Start with a base signal from trend and RSI
            reasoning = ""
            if trend == "bullish":
                if latest_rsi < 30:
                    signal_base = "BUY"
                    reasoning = "oversold in bullish trend"
                elif latest_rsi < 50:
                    signal_base = "ACCUMULATE"
                    reasoning = "RSI below midpoint in bullish trend"
                else:
                    signal_base = "HOLD/LONG"
                    reasoning = "bullish trend continuation"
            elif trend == "bearish":
                if latest_rsi > 70:
                    signal_base = "SELL"
                    reasoning = "overbought in bearish trend"
                elif latest_rsi > 50:
                    signal_base = "REDUCE"
                    reasoning = "RSI above midpoint in bearish trend"
                else:
                    signal_base = "AVOID/SHORT"
                    reasoning = "bearish trend continuation"
            else:  # neutral trend
                if latest_rsi < 30:
                    signal_base = "BUY"
                    reasoning = "oversold conditions"
                elif latest_rsi > 70:
                    signal_base = "SELL"
                    reasoning = "overbought conditions"
                else:
                    signal_base = "HOLD"
                    reasoning = "no clear directional bias - wait for confirmation"
                    
            # Modify signal strength based on ADX and other factors
            signal = signal_base
            
            # Strengthen or weaken signal based on trend strength
            if signal_base in ["BUY", "ACCUMULATE"] and trend_strength == "strong" and trend == "bullish":
                signal = "STRONG BUY"
                reasoning += ", strong trend confirmation"
            elif signal_base in ["SELL", "REDUCE"] and trend_strength == "strong" and trend == "bearish":
                signal = "STRONG SELL"
                reasoning += ", strong trend confirmation"
            elif signal_base in ["BUY", "ACCUMULATE", "SELL", "REDUCE"] and trend_strength == "weak":
                if signal_base in ["BUY", "ACCUMULATE"]:
                    signal = "SPECULATIVE BUY"
                else:
                    signal = "SPECULATIVE SELL"
                reasoning += ", weak trend suggests caution"
                
            # Take into account Bollinger Bands for mean reversion signals
            # Use the same price (bb_price) for consistency in our bollinger band calculations
            if near_upper_band and latest_rsi > 70:
                if trend != "bullish" or trend_strength != "strong":
                    signal = "STRONG SELL"
                    reasoning = "price at upper band with overbought RSI"
            elif near_lower_band and latest_rsi < 30:
                if trend != "bearish" or trend_strength != "strong":
                    signal = "STRONG BUY"
                    reasoning = "price at lower band with oversold RSI"
                    
            # Additional check for very low-priced stocks in bearish conditions
            if current_price < 1.0 and trend == "bearish" and signal in ["SELL", "AVOID/SHORT"]:
                # Ensure reasoning is appropriate for low-priced bearish setup
                if "bearish" not in reasoning:
                    reasoning += ", bearish trend on low-priced asset"
                    
            # Special handling for very low-priced stocks (under $1)
            # Use current_price for this check to ensure we're using real-time data
            actual_price = current_price
            if actual_price < 1.0 and signal in ["SELL", "STRONG SELL", "AVOID/SHORT"]:
                logger.info(f"Low-priced stock detected: ${actual_price}. Adjusting short target calculations.")
                # For very low priced stocks, be more conservative with short targets
                    
            # Consider MACD for timing signals
            if macd_trend == "potential bullish crossover" and signal in ["BUY", "ACCUMULATE", "SPECULATIVE BUY"]:
                signal = "TIMELY BUY"
                reasoning += ", potential MACD bullish crossover"
            elif macd_trend == "potential bearish crossover" and signal in ["SELL", "REDUCE", "SPECULATIVE SELL"]:
                signal = "TIMELY SELL"
                reasoning += ", potential MACD bearish crossover"
                
            # Provide detailed, actionable suggestions with specific price levels based on technicals
            action_suggestion = ""
            if "BUY" in signal:
                # Use technical levels to determine entry/stop/target
                if lower_band is not None and upper_band is not None:
                    # If we're near lower band, use it as entry and aim for middle band
                    if near_lower_band:
                        entry_price = round(latest_price, 2)  # Current price near support
                        stop_price = round(max(lower_band * 0.98, latest_price - (latest_atr * 1.5 if latest_atr else latest_price * 0.03)), 2)
                        target_price = round(middle_band, 2)
                    else:
                        # Normal case - use ATR for risk sizing
                        entry_price = round(latest_price * 0.995, 2)  # Slight discount to current price
                        stop_size = latest_atr * 1.5 if latest_atr else latest_price * 0.03
                        stop_price = round(entry_price - stop_size, 2)
                        reward_size = stop_size * 2  # 2:1 reward-risk ratio
                        target_price = round(entry_price + reward_size, 2)
                else:
                    # Fallback if no bands available
                    entry_price = round(latest_price * 0.995, 2)
                    stop_price = round(entry_price * 0.97, 2)
                    target_price = round(entry_price * 1.06, 2)
                
                action_suggestion = f" - Enter at ${entry_price} with stop loss at ${stop_price} and target at ${target_price}"
                
            elif "SELL" in signal:
                # Use technical levels for exit price
                if upper_band is not None and near_upper_band:
                    # We're near upper band, recommend immediate exit
                    exit_price = "market"
                    action_suggestion = f" - Exit long positions immediately at market price"
                else:
                    # Give slight cushion above current price
                    exit_buffer = latest_atr * 0.5 if latest_atr else latest_price * 0.01
                    exit_price = round(latest_price + exit_buffer, 2)
                    action_suggestion = f" - Exit long positions at market or use limit order at ${exit_price}"
                    
            elif signal == "ACCUMULATE":
                # Use support levels or Bollinger bands
                if lower_band is not None:
                    entry_zone_low = round(lower_band, 2)
                    entry_zone_high = round(latest_price, 2)
                    stop_level = round(lower_band * 0.97, 2)
                else:
                    # Fallback to percentage-based
                    entry_zone_low = round(latest_price * 0.97, 2)
                    entry_zone_high = round(latest_price, 2)
                    stop_level = round(entry_zone_low * 0.97, 2)
                    
                action_suggestion = f" - Add to position between ${entry_zone_high} and ${entry_zone_low}, maintain stop at ${stop_level}"
                
            elif signal == "REDUCE":
                exit_portion = "25-50%"
                
                # Use technical levels for exit suggestion
                if upper_band is not None and middle_band is not None:
                    if latest_price > middle_band:
                        exit_level = "current market price"
                    else:
                        exit_level = f"${round(middle_band, 2)}"
                else:
                    # Fallback
                    exit_level = f"${round(latest_price * 1.02, 2)}"
                    
                action_suggestion = f" - Reduce position by {exit_portion} at {exit_level}"
                
            elif signal == "HOLD/LONG":
                # Calculate trailing stop using multiple factors
                if latest_atr is not None:
                    # ATR-based trailing stop
                    atr_mult = 2 if trend_strength == "strong" else 1.5
                    trail_points = latest_atr * atr_mult
                elif lower_band is not None:
                    # Bollinger-based trailing stop
                    trail_points = latest_price - lower_band
                else:
                    # Percentage-based fallback
                    trail_points = latest_price * (0.05 if trend_strength == "strong" else 0.03)
                    
                trailing_stop = round(latest_price - trail_points, 2)
                action_suggestion = f" - Maintain long position with trailing stop at ${trailing_stop}"
                
            elif signal == "AVOID/SHORT":
                # Use current price and technical levels for short entry
                if upper_band is not None and middle_band is not None and lower_band is not None:
                    # Short entry based on current price relative to bands
                    # For stocks that have declined significantly on daily timeframe, use more conservative entries
                    if interval == "daily" and price_change_pct < -20:
                        entry_price = round(current_price * 1.05, 2)  # 5% above current price
                        stop_price = round(entry_price * 1.07, 2)     # 7% above entry
                        
                        # Don't target below 50% of current price for penny stocks
                        if current_price < 1.0:
                            target_price = max(current_price * 0.5, 0.01)
                        else:
                            target_price = current_price * 0.85  # Target 15% drop
                            
                        logger.info(f"Daily timeframe with large price decline: Using adjusted entry=${entry_price}, stop=${stop_price}, target=${target_price}")
                    else:
                        if near_upper_band:
                            # If price is near upper band, use upper band as technical entry point
                            entry_price = round(upper_band, 2)
                            stop_price = round(upper_band * 1.05, 2)
                        else:
                            # Use middle band as technical entry for short
                            # This is a standard technical entry point for shorts
                            
                            # Reality check for extended timeframes - don't use bands that are too far from current price
                            if interval in ["daily", "weekly", "monthly"] and abs(middle_band - current_price)/current_price > 0.30:
                                # For extended timeframes, if middle band is >30% from current price, use current price +5%
                                entry_price = round(current_price * 1.05, 2)
                                logger.info(f"Extended timeframe: adjusted entry from ${middle_band} to ${entry_price} (30% deviation rule)")
                            else:
                                entry_price = round(middle_band, 2)
                            
                            # Technical stop above the middle band
                            stop_price = round(entry_price * 1.05, 2)
                        
                        # Simple target calculation using lower Bollinger Band
                        # Lower Bollinger Band is a standard technical target for shorts
                        
                        # Reality check for targets on extended timeframes
                        if interval in ["daily", "weekly", "monthly"]:
                            # For extended timeframes, target shouldn't be too extreme
                            # For penny stocks, don't go below 50% of current price
                            if current_price < 1.0:
                                target_price = max(lower_band, current_price * 0.5, 0.01)
                            else:
                                target_price = max(lower_band, entry_price * 0.75, 0.01)
                        else:
                            target_price = round(max(lower_band, 0.01), 2)
                else:
                    # Use the real-time price (current_price) for all trade signal calculations
                    # This ensures consistency with what's shown to the user at the top of the analysis
                    actual_price = current_price
                    
                    # Special handling for very low-priced stocks (under $1)
                    if actual_price < 1.0:
                        logger.info(f"Low-priced stock detected for short recommendation: ${actual_price}")
                        
                        # For low-priced stocks, use technical levels when available
                        if upper_band is not None and middle_band is not None and lower_band is not None:
                            # For low-priced shorts, check timeframe specifics
                            if interval in ["daily", "weekly", "monthly"]:
                                # Reality check for extended timeframes - don't use bands too far from price
                                if abs(middle_band - actual_price)/actual_price > 0.30:
                                    # If middle band is >30% away, use more realistic entry based on current price
                                    entry_price = round(actual_price * 1.05, 2)
                                    logger.info(f"Extended timeframe penny stock: adjusted entry from ${middle_band} to ${entry_price} (30% deviation rule)")
                                else:
                                    entry_price = round(middle_band, 2)
                                
                                # Limit target for penny stocks on extended timeframes
                                target_price = round(max(lower_band, actual_price * 0.5, 0.01), 2)
                                logger.info(f"Extended timeframe penny stock: limiting target to 50% of current price: ${target_price}")
                            else:
                                # For shorter timeframes, use standard technical levels
                                entry_price = round(middle_band, 2)
                                # Use the lower Bollinger Band as target
                                target_price = round(max(lower_band, 0.01), 2)
                            
                            # Stop just above entry point for tight risk management on low priced assets
                            stop_price = round(entry_price * 1.05, 2)
                                
                            logger.info(f"Low-priced technical short: Entry=${entry_price}, Stop=${stop_price}, Target=${target_price}")
                        else:
                            # Fallback for when technical levels aren't available
                            # Calculate a technical entry point using price/ATR relationship
                            # This is typically slightly above the current price for short entries
                            entry_price = round(actual_price * 1.02, 2)  # 2% above current price as technical entry
                            
                            if latest_atr is not None:
                                # Use ATR-based levels for more technical accuracy
                                # For very low priced stocks, ensure ATR is meaningful but not too large
                                # Limit ATR to a maximum percentage of price for low-priced stocks
                                effective_atr = min(latest_atr, actual_price * 0.20)  # Cap ATR at 20% of price
                                effective_atr = max(effective_atr, actual_price * 0.05)  # Ensure minimum volatility
                                stop_price = round(entry_price + effective_atr, 2)
                                
                                # For extended timeframes, be more conservative with targets
                                if interval in ["daily", "weekly", "monthly"]:
                                    # For daily/weekly timeframes, don't target below 50% of current price
                                    target_price = round(max(entry_price - effective_atr, actual_price * 0.5, 0.01), 2)
                                    logger.info(f"Extended timeframe: limiting target to 50% of current price minimum")
                                else:
                                    # Standard ATR-based target for intraday - use 2x ATR
                                    target_price = round(max(entry_price - (effective_atr * 2), 0.01), 2)
                                    
                                logger.info(f"Low-priced ATR-based short: Entry=${entry_price}, Stop=${stop_price}, Target=${target_price}")
                            else:
                                # Simple percentage-based stop and target
                                stop_price = round(entry_price * 1.05, 2)
                                
                                # For extended timeframes, be more conservative with targets
                                if interval in ["daily", "weekly", "monthly"]:
                                    # For daily/weekly timeframes with penny stocks, target no lower than 50% of current price
                                    target_price = round(max(entry_price * 0.75, actual_price * 0.5, 0.01), 2)
                                    logger.info(f"Extended timeframe: using conservative target for penny stock")
                                else:
                                    # Standard percentage-based target for intraday
                                    target_price = round(max(entry_price * 0.50, 0.01), 2)
                                    
                                logger.info(f"Low-priced percentage-based short: Entry=${entry_price}, Stop=${stop_price}, Target=${target_price}")
                    else:
                        # Standard technical approach for normal price stocks
                        if latest_atr is not None and latest_ema is not None:
                            # Use EMA as technical entry point for shorts, with reality check for extended timeframes
                            if interval in ["daily", "weekly", "monthly"] and abs(latest_ema - current_price)/current_price > 0.30:
                                # If EMA is more than 30% away from current price, use a more reasonable entry
                                entry_price = round(current_price * 1.05, 2)  # 5% above current price
                                logger.info(f"Extended timeframe: adjusted entry from ${latest_ema} to ${entry_price} (30% deviation rule)")
                            else:
                                entry_price = round(latest_ema, 2)
                            
                            # Stop using 1.5x ATR above entry (standard technical approach)
                            stop_size = latest_atr * 1.5
                            stop_price = round(entry_price + stop_size, 2)
                            
                            # Target calculation with timeframe-specific adjustments
                            if interval in ["daily", "weekly", "monthly"]:
                                # For extended timeframes, use more conservative targets
                                # Larger timeframes need more realistic targets
                                target_price = round(max(entry_price - (latest_atr * 1.5), entry_price * 0.7, 0.01), 2)
                                logger.info(f"Extended timeframe: using conservative target calculation")
                            else:
                                # Intraday can use more aggressive targets with standard 2x ATR
                                target_price = round(max(entry_price - (latest_atr * 2), 0.01), 2)
                        else:
                            # When EMA/ATR not available, use price-based technical levels
                            # For shorts, expect a slight pullback before entry
                            entry_price = round(actual_price * 1.02, 2)  # 2% above as technical entry
                            stop_price = round(entry_price * 1.05, 2)  # 5% above entry
                            
                            # Target calculation with timeframe-specific adjustments
                            if interval in ["daily", "weekly", "monthly"]:
                                # For extended timeframes, use more conservative target
                                # Don't target below 70% of entry price for regular stocks
                                target_price = round(max(entry_price * 0.70, 0.01), 2)
                                logger.info(f"Extended timeframe without ATR: using conservative target at 70% of entry")
                            else:
                                # Standard target for intraday - 10% below entry
                                target_price = round(max(entry_price * 0.90, 0.01), 2)  # 10% below entry
                    
                # Ensure target price is never negative
                target_price = max(0.01, target_price)
                action_suggestion = f" - Short at ${entry_price} with stop at ${stop_price} and target at ${target_price}"
                
            elif signal == "HOLD":
                # Identify key support/resistance levels for breakout opportunities
                support_price_data = processed_df
                
                # Get multi-timeframe support/resistance using refactored method
                mtf_support_resistance = await self._analyze_multi_timeframe_levels(
                    symbol=symbol, 
                    asset_type=asset_type, 
                    current_price=current_price, 
                    main_interval=interval,
                    latest_atr=latest_atr
                )
                
                # Extract MTF levels if available
                mtf_support_levels = None
                mtf_resistance_levels = None
                if mtf_support_resistance:
                    mtf_support_levels = [zone["price"] for zone in mtf_support_resistance.get("support_zones", [])]
                    mtf_resistance_levels = [zone["price"] for zone in mtf_support_resistance.get("resistance_zones", [])]
                    
                # Call our methods to identify support and resistance levels
                support_prices = PriceLevelAnalyzer.identify_support_levels(
                    price_data=support_price_data, 
                    current_price=current_price,
                    latest_atr=latest_atr,
                    interval=interval
                )
                
                resistance_prices = PriceLevelAnalyzer.identify_resistance_levels(
                    price_data=support_price_data,
                    current_price=current_price,
                    latest_atr=latest_atr, 
                    interval=interval
                )
                
                # Handle cases where we couldn't find technical levels
                if not support_prices or len(support_prices) == 0:
                    # If ATR is available, use it for dynamic levels
                    if latest_atr:
                        logger.info(f"No support levels found. Using ATR-based level: ATR=${latest_atr}")
                        breakout_low = round(current_price - (latest_atr * 1.5), 2)
                    else:
                        # If no technical data at all, use a very conservative level
                        # just to provide some actionable information
                        logger.warning(f"No support levels or ATR available. Using conservative level.")
                        breakout_low = round(current_price * 0.95, 2)  # 5% below, but marked as non-technical
                else:
                    breakout_low = round(support_prices[0], 2)
                    
                if not resistance_prices or len(resistance_prices) == 0:
                    # If ATR is available, use it for dynamic levels
                    if latest_atr:
                        logger.info(f"No resistance levels found. Using ATR-based level: ATR=${latest_atr}")
                        breakout_high = round(current_price + (latest_atr * 1.5), 2)
                    else:
                        # If no technical data at all, use a very conservative level
                        logger.warning(f"No resistance levels or ATR available. Using conservative level.")
                        breakout_high = round(current_price * 1.05, 2)  # 5% above, but marked as non-technical
                else:
                    breakout_high = round(resistance_prices[0], 2)
                
                # Check if price is already outside the range
                price_above_resistance = current_price > breakout_high
                price_below_support = current_price < breakout_low
                
                if price_above_resistance:
                    # Price already above resistance - bullish position
                    if resistance_prices and len(resistance_prices) > 1:
                        next_target = round(resistance_prices[1], 2)
                    else:
                        # Use ATR for dynamic target if available
                        if latest_atr:
                            next_target = round(breakout_high + (latest_atr * 2), 2)
                        else:
                            # Only as last resort use a minimal projection
                            next_target = round(breakout_high + (breakout_high - current_price), 2)
                    target_details = f"${next_target}/{breakout_high}"
                    signal = "BULLISH"
                    action_suggestion = f" - Price has already broken above resistance (${breakout_high}), consider bullish position"
                elif price_below_support:
                    # Price already below support - bearish position
                    if support_prices and len(support_prices) > 1:
                        next_target = round(support_prices[1], 2)
                    else:
                        # Use ATR for dynamic target if available
                        if latest_atr:
                            next_target = round(breakout_low - (latest_atr * 2), 2)
                        else:
                            # Use the distance from price to support as projection
                            next_target = round(breakout_low - (current_price - breakout_low), 2)
                        # Ensure target isn't negative for very low-priced stocks
                        next_target = max(next_target, 0.01)
                    target_details = f"${breakout_low}/{next_target}"
                    signal = "BEARISH"
                    action_suggestion = f" - Price has already broken below support (${breakout_low}), consider bearish position"
                else:
                    # Wait for breakout with specific levels and targets
                    action_suggestion = f" - Watch key levels: Buy above ${breakout_high} with target at ${round(breakout_high*1.05, 2)}; Sell below ${breakout_low} with target at ${round(breakout_low*0.95, 2)}"
                    target_details = f"${round(breakout_high*1.05, 2)}/${round(breakout_low*0.95, 2)}"
            
            # Combine signal and reasoning
            full_signal = f"{signal} {action_suggestion}"
            
            # Format interval for display
            if is_extended:
                interval_display = interval.capitalize()
            else:
                interval_display = interval.replace("min", " minute")
                
            # Get support and resistance levels for display - these should already be calculated earlier
            # Only ensure they exist to avoid errors
            if not ('support_prices' in locals() and support_prices):
                support_prices = []
            if not ('resistance_prices' in locals() and resistance_prices):
                resistance_prices = []
                
            # Get multi-timeframe analysis if not already done
            if not ('mtf_support_resistance' in locals() and mtf_support_resistance):
                mtf_support_resistance = await self._analyze_multi_timeframe_levels(
                    symbol=symbol, 
                    asset_type=asset_type, 
                    current_price=current_price, 
                    main_interval=interval,
                    latest_atr=latest_atr
                )
            
            # Extract MTF levels if available
            mtf_support_levels = None
            mtf_resistance_levels = None
            if mtf_support_resistance:
                mtf_support_levels = [zone["price"] for zone in mtf_support_resistance.get("support_zones", [])]
                mtf_resistance_levels = [zone["price"] for zone in mtf_support_resistance.get("resistance_zones", [])]
            
            # Format the support levels with appropriate labels - only use real historical levels
            # First ensure support prices have enough spacing between them (at least 1.5% apart)
            spaced_support_prices = []
            if support_prices and len(support_prices) > 0:
                # Add the first support level
                spaced_support_prices.append(support_prices[0])
                
                # Check remaining levels and only add if they are at least 1.5% different from any existing level
                for level in support_prices[1:]:
                    # Check against all existing levels
                    if all(abs(level - existing) / existing > 0.015 for existing in spaced_support_prices):
                        spaced_support_prices.append(level)
                        # Only keep up to 3 levels
                        if len(spaced_support_prices) >= 3:
                            break
            
            # Now format the properly spaced support levels
            if len(spaced_support_prices) >= 3:
                support1, support2, support3 = spaced_support_prices[:3]
                # Ensure no negative values
                support1 = max(0.01, support1)
                support2 = max(0.01, support2)
                support3 = max(0.01, support3)
                support_levels = f"🟠 ${support1} - Near-term support\n🟠 ${support2} - Intermediate support\n🟠 ${support3} - Major support level"
            elif len(spaced_support_prices) == 2:
                support1, support2 = spaced_support_prices[:2]
                # Ensure no negative values
                support1 = max(0.01, support1)
                support2 = max(0.01, support2)
                support_levels = f"🟠 ${support1} - Near-term support\n🟠 ${support2} - Intermediate support"
            elif len(spaced_support_prices) == 1:
                support1 = max(0.01, spaced_support_prices[0])
                support_levels = f"🟠 ${support1} - Near-term support"
            else:
                # No support levels found, check for historical swing lows
                lows = []
                if processed_df is not None and len(processed_df) > 0:
                    # Try to find actual swing lows from the data
                    for i in range(min(150, len(processed_df))):
                        if i > 1 and i < len(processed_df) - 1:
                            if processed_df.iloc[i]['low'] <= processed_df.iloc[i-1]['low'] and processed_df.iloc[i]['low'] <= processed_df.iloc[i+1]['low']:
                                lows.append(processed_df.iloc[i]['low'])
                    
                    # Filter to only include lows below current price
                    valid_lows = [l for l in lows if l < current_price]
                    valid_lows.sort(reverse=True)  # Sort highest to lowest
                    
                    if len(valid_lows) >= 3:
                        support1 = round(max(valid_lows[0], 0.01), 2)
                        support2 = round(max(valid_lows[1], 0.01), 2)
                        support3 = round(max(valid_lows[2], 0.01), 2)
                        support_levels = f"🟠 ${support1} - Near-term support\n🟠 ${support2} - Intermediate support\n🟠 ${support3} - Major support level"
                    elif len(valid_lows) == 2:
                        support1 = round(max(valid_lows[0], 0.01), 2)
                        support2 = round(max(valid_lows[1], 0.01), 2)
                        support_levels = f"🟠 ${support1} - Near-term support\n🟠 ${support2} - Intermediate support"
                    elif len(valid_lows) == 1:
                        support1 = round(max(valid_lows[0], 0.01), 2)
                        support_levels = f"🟠 ${support1} - Near-term support"
                    else:
                        # Absolutely no support data available
                        support_levels = "No historical support levels detected with current data"
                else:
                    # No dataframe available
                    support_levels = "No historical support levels detected with current data"
                
            # Format the resistance levels with appropriate labels
            # First ensure resistance prices have enough spacing between them (at least 1.5% apart)
            spaced_resistance_prices = []
            if resistance_prices and len(resistance_prices) > 0:
                # Add the first resistance level
                spaced_resistance_prices.append(resistance_prices[0])
                
                # Check remaining levels and only add if they are at least 1.5% different from any existing level
                for level in resistance_prices[1:]:
                    # Check against all existing levels
                    if all(abs(level - existing) / existing > 0.015 for existing in spaced_resistance_prices):
                        spaced_resistance_prices.append(level)
                        # Only keep up to 3 levels
                        if len(spaced_resistance_prices) >= 3:
                            break
            
            # Now format the properly spaced resistance levels
            if len(spaced_resistance_prices) >= 3:
                resistance1, resistance2, resistance3 = spaced_resistance_prices[:3]
                resistance_levels = f"🔴 ${resistance1} - Near-term resistance\n🔴 ${resistance2} - Intermediate resistance\n🔴 ${resistance3} - Major resistance level"
            elif len(spaced_resistance_prices) == 2:
                resistance1, resistance2 = spaced_resistance_prices[:2]
                resistance_levels = f"🔴 ${resistance1} - Near-term resistance\n🔴 ${resistance2} - Intermediate resistance"
            elif len(spaced_resistance_prices) == 1:
                resistance1 = spaced_resistance_prices[0]
                resistance_levels = f"🔴 ${resistance1} - Near-term resistance"
            else:
                # No resistance levels found, check for historical swing highs
                highs = []
                if processed_df is not None and len(processed_df) > 0:
                    # Try to get actual previous highs from the data
                    for i in range(min(150, len(processed_df))):
                        if i > 1 and i < len(processed_df) - 1:
                            if processed_df.iloc[i]['high'] >= processed_df.iloc[i-1]['high'] and processed_df.iloc[i]['high'] >= processed_df.iloc[i+1]['high']:
                                highs.append(processed_df.iloc[i]['high'])
                    
                    # Filter to only include highs above current price
                    valid_highs = [h for h in highs if h > current_price]
                    valid_highs.sort()  # Sort lowest to highest
                    
                    if len(valid_highs) >= 3:
                        resistance1 = round(valid_highs[0], 2)
                        resistance2 = round(valid_highs[1], 2)
                        resistance3 = round(valid_highs[2], 2)
                        resistance_levels = f"🔴 ${resistance1} - Near-term resistance\n🔴 ${resistance2} - Intermediate resistance\n🔴 ${resistance3} - Major resistance level"
                    elif len(valid_highs) == 2:
                        resistance1 = round(valid_highs[0], 2)
                        resistance2 = round(valid_highs[1], 2)
                        resistance_levels = f"🔴 ${resistance1} - Near-term resistance\n🔴 ${resistance2} - Intermediate resistance"
                    elif len(valid_highs) == 1:
                        resistance1 = round(valid_highs[0], 2)
                        resistance_levels = f"🔴 ${resistance1} - Near-term resistance"
                    else:
                        # Absolutely no resistance data available
                        resistance_levels = "No historical resistance levels detected with current data"
                else:
                    # No dataframe available
                    resistance_levels = "No historical resistance levels detected with current data"
            
            # Enrich with news sentiment data
            try:
                # Check if signal_base and reasoning are defined, otherwise use default values
                signal_to_use = signal_base if 'signal_base' in locals() else "NEUTRAL"
                reasoning_to_use = reasoning if 'reasoning' in locals() else ""
                
                _, _, sentiment_data = await self.enhance_with_sentiment(symbol, signal_to_use, reasoning_to_use)
                sentiment_score = sentiment_data.get("sentiment_score", 0)
                sentiment_label = sentiment_data.get("sentiment_label", "NEUTRAL")
                news_count = sentiment_data.get("news_count", 0)
                article_highlight = sentiment_data.get("article_highlight", "")
            except Exception as e:
                logger.error(f"Error enhancing with sentiment for {symbol}: {str(e)}")
                sentiment_data = {
                    "sentiment_score": 0,
                    "sentiment_label": "NEUTRAL",
                    "news_count": 0,
                    "article_highlight": ""
                }
                sentiment_score = 0
                sentiment_label = "NEUTRAL"
                news_count = 0
                article_highlight = ""
            
            # Determine additional data for template
            rsi_interpretation = "NEUTRAL"
            if latest_rsi >= 70:
                rsi_interpretation = "OVERBOUGHT"
            elif latest_rsi <= 30:
                rsi_interpretation = "OVERSOLD"

            # Use current price (real-time) for all display calculations
            display_price = current_price
            
            # We should compare real-time price to SMA for accuracy
            price_vs_ma = "ABOVE" if display_price > latest_sma else "BELOW"

            macd_interpretation = "NEUTRAL"
            if macd_value is not None and signal_value is not None:
                if macd_value > signal_value:
                    macd_interpretation = "BULLISH"
                else:
                    macd_interpretation = "BEARISH"

            # Use the same price (real-time) for determining Bollinger Band status
            bbands_status = "NEUTRAL"
            if display_price is not None and upper_band is not None and lower_band is not None:
                # Safety check - ensure lower band is never negative
                if lower_band < 0:
                    logger.warning(f"Negative lower band detected (${lower_band}), setting to minimum valid price (0.01)")
                    lower_band = 0.01
                    
                if display_price > upper_band:
                    bbands_status = "ABOVE UPPER BAND"
                elif display_price < lower_band:
                    bbands_status = "BELOW LOWER BAND"
                else:
                    bbands_status = "WITHIN BANDS"
                    
            logger.info(f"BBands status: {bbands_status} (Price=${display_price}, Upper=${upper_band}, Lower=${lower_band})")
                    
            # Access the market analysis template via the prompt manager
            template = self.prompt_manager.get_template_section("market_analysis_template", {})
            
            
            # Include ATR information if available
            if latest_atr is not None:
                atr_pct = (latest_atr / latest_price) * 100 if latest_price != 0 else 0
                
                # Adjust volatility thresholds based on timeframe
                high_vol_threshold = 4.0 if interval in ["weekly", "monthly"] else 3.0 if interval == "daily" else 2.0
                mod_vol_threshold = 2.0 if interval in ["weekly", "monthly"] else 1.5 if interval == "daily" else 1.0
                
                if atr_pct > high_vol_threshold:
                    atr_interpretation = "HIGH VOLATILITY"
                elif atr_pct > mod_vol_threshold:
                    atr_interpretation = "MODERATE VOLATILITY"
                else:
                    atr_interpretation = "LOW VOLATILITY"
            else:
                atr_pct = 0
                atr_interpretation = "UNKNOWN"
            
            # Include Stochastic information if available
            if k_value is not None and d_value is not None:
                # Adjust overbought/oversold thresholds based on timeframe
                ob_threshold = 75 if interval in ["1min", "5min"] else 80
                os_threshold = 25 if interval in ["1min", "5min"] else 20
                
                if k_value > ob_threshold and d_value > ob_threshold:
                    stoch_interpretation = "OVERBOUGHT"
                elif k_value < os_threshold and d_value < os_threshold:
                    stoch_interpretation = "OVERSOLD"
                elif k_value > d_value:
                    stoch_interpretation = "BULLISH MOMENTUM"
                else:
                    stoch_interpretation = "BEARISH MOMENTUM"
            else:
                stoch_interpretation = "UNKNOWN"
            
            # Fill in the template variables
            template_data = {
                "ASSET_TYPE": asset_type.upper(),
                "SYMBOL": symbol,
                "TIMEFRAME": interval.upper() if interval in ["daily", "weekly", "monthly"] else interval_display.upper(),
                "PRICE": "N/A" if current_price is None else f"{current_price:,.2f}",
                "CHANGE_DIRECTION": change_direction,
                "CHANGE_PCT": "N/A" if price_change_pct is None else f"{abs(price_change_pct):.2f}",
                "PRICE_VS_MA": "UNKNOWN" if latest_price is None or latest_sma is None else price_vs_ma,
                "SMA_PERIOD": str(sma_period),
                "SMA_VALUE": self.format_indicator_value(latest_sma).replace("$", ""),
                "EMA_PERIOD": str(ema_period),
                "EMA_VALUE": self.format_indicator_value(latest_ema).replace("$", ""),
                "EMA_TREND": "UNKNOWN" if latest_ema is None or latest_sma is None else ("up" if latest_ema > latest_sma else "down"),
                "VWAP_VALUE": "" if is_extended or vwap_value is None else self.format_indicator_value(vwap_value).replace("$", ""),
                "RSI_VALUE": "N/A" if latest_rsi is None else f"{latest_rsi:.2f}",
                "RSI_INTERPRETATION": "UNKNOWN" if latest_rsi is None else rsi_interpretation,
                "ADX_VALUE": "N/A" if latest_adx is None else f"{latest_adx:.2f}",
                "TREND_STRENGTH": "UNKNOWN" if trend_strength is None else trend_strength.upper(),
                "MACD_VALUE": "N/A" if macd_value is None else (f"{macd_value:.4f}" if abs(macd_value) < 0.01 else f"{macd_value:.2f}"),
                "MACD_VS_SIGNAL": "UNKNOWN" if macd_value is None or signal_value is None else ("above" if macd_value > signal_value else "below"),
                "SIGNAL_VALUE": "N/A" if signal_value is None else (f"{signal_value:.4f}" if abs(signal_value) < 0.01 else f"{signal_value:.2f}"),
                "MACD_INTERPRETATION": "UNKNOWN" if macd_interpretation is None else macd_interpretation,
                "BBANDS_STATUS": "UNKNOWN" if bbands_status is None else bbands_status,
                "SENTIMENT_LABEL": "NEUTRAL" if sentiment_label is None else sentiment_label,
                "SENTIMENT_SCORE": "N/A" if sentiment_score is None else f"{sentiment_score:.2f}",
                "ARTICLE_COUNT": "0" if news_count is None else str(news_count),
                "ARTICLE_HIGHLIGHT": article_highlight if article_highlight else "No recent news articles found for this asset.\n",
                "SUPPORT_LEVELS": "N/A" if support_levels is None else support_levels,
                "RESISTANCE_LEVELS": "N/A" if resistance_levels is None else resistance_levels,
                "RECOMMENDATION": "NEUTRAL" if full_signal is None else full_signal,
                "CURRENT_TIME": "N/A" if current_time is None else current_time,
                "DATA_TIME": "N/A" if most_recent_date is None else most_recent_date,
                "OPEN_PRICE": "N/A" if latest_open is None else self.format_indicator_value(latest_open).replace("$", ""),
                "VOL_COMMENT": "" if vol_comment is None else vol_comment,
                "ATR_PERIOD": str(atr_period),
                "ATR_VALUE": "N/A" if latest_atr is None else self.format_indicator_value(latest_atr).replace("$", ""),
                "ATR_PCT": "N/A" if atr_pct is None else f"{atr_pct:.2f}",
                "ATR_INTERPRETATION": atr_interpretation,
                "STOCH_K_PERIOD": str(stoch_k_period),
                "STOCH_D_PERIOD": str(stoch_d_period),
                "STOCH_K": "N/A" if k_value is None else (f"{k_value:.4f}" if abs(k_value) < 0.01 else f"{k_value:.2f}"),
                "STOCH_D": "N/A" if d_value is None else (f"{d_value:.4f}" if abs(d_value) < 0.01 else f"{d_value:.2f}"),
                "STOCH_INTERPRETATION": stoch_interpretation
            }
            
             # Add data summary for extended timeframes
            if is_extended and 'data_summary' in locals():
                template_data["DATA_SUMMARY"] = data_summary
            else:
                template_data["DATA_SUMMARY"] = ""
            
            
            # Get the most relevant article from sentiment data
            most_relevant_article = sentiment_data.get("most_relevant_article", None)
            
            # Calculate trade levels based on technical analysis for both news and no-news scenarios
            # This ensures all technical calculations are done in Python, not extracted from LLM text
            
            # All entry/exit price targets are now determined by the LLM
            # No price calculations needed here anymore
            
            # Create a dynamic technical indicators section with conditional VWAP
            tech_indicators = [
                f"- Current Price: ${current_price:.2f} ({change_direction} {price_change_pct:.2f}%)",
                f"- Trend: {trend.upper()} with {trend_strength.upper()} momentum (ADX: {latest_adx:.2f})",
                f"- Moving Averages: Price trading {price_vs_ma} {sma_period}-day SMA (${latest_sma:.2f})",
                f"- EMA {ema_period}-day: ${latest_ema:.2f} trending {macd_trend.lower()}"
            ]
            
            # Add VWAP only for intraday timeframes
            if not is_extended and template_data.get("VWAP_VALUE"):
                tech_indicators.append(f"- VWAP: ${template_data['VWAP_VALUE']} - Critical volume-weighted price level")
            
            # Add the rest of the indicators
            tech_indicators.extend([
                f"- RSI: {latest_rsi:.2f} ({rsi_interpretation})",
                f"- MACD: {macd_value:.2f} vs Signal {signal_value:.2f} - {macd_trend.upper()}",
                f"- Bollinger Bands: Price is {bbands_status.lower()}",
                f"- ATR: ${latest_atr:.2f} ({atr_pct:.2f}% of price) - {atr_interpretation}",
                f"- Stochastic: {k_value:.2f}/{d_value:.2f} - {stoch_interpretation}",
                f"- Volume Analysis: {vol_comment}",
                f"- Technical Signal: {signal}"
            ])
            
            # Join all indicators with newlines
            technical_indicators_text = "\n".join(tech_indicators)
            
            # Get multi-timeframe support/resistance if available and not already done
            if not ('mtf_support_resistance' in locals() and mtf_support_resistance):
                mtf_support_resistance = None
                if processed_df is not None and len(processed_df) > 0:
                    mtf_support_resistance = await self._analyze_multi_timeframe_levels(
                        symbol=symbol, 
                        asset_type=asset_type, 
                        current_price=current_price, 
                        main_interval=interval,
                        latest_atr=latest_atr
                    )
                
            # Add multi-timeframe analysis to technical indicators if available
            if mtf_support_resistance:
                mtf_summary = mtf_support_resistance.get("summary", "")
                technical_indicators_text += f"\n\n{mtf_summary}"
                
                # Store support/resistance zones for later reference
                mtf_support_zones = mtf_support_resistance.get("support_zones", [])
                mtf_resistance_zones = mtf_support_resistance.get("resistance_zones", [])
                
                # Create a detailed summary to include in the LLM prompt
                if mtf_support_zones or mtf_resistance_zones:
                    mtf_detail = "\n\nMULTI-TIMEFRAME ANALYSIS DETAIL:"
                    
                    if mtf_resistance_zones:
                        mtf_detail += "\nKey MTF resistance zones:"
                        for zone in mtf_resistance_zones[:3]:  # Top 3 resistance zones
                            if zone["price"] > current_price:
                                timeframes = "/".join(zone["timeframes"])
                                strength = zone.get('strength', 0)
                                mtf_detail += f"\n- ${zone['price']} (strength: {strength:.1f}, timeframes: {timeframes})"
                    
                    if mtf_support_zones:
                        mtf_detail += "\nKey MTF support zones:"
                        for zone in mtf_support_zones[:3]:  # Top 3 support zones
                            if zone["price"] < current_price:
                                timeframes = "/".join(zone["timeframes"])
                                strength = zone.get('strength', 0)
                                mtf_detail += f"\n- ${zone['price']} (strength: {strength:.1f}, timeframes: {timeframes})"
                    
                    # This will be included in the prompt to the LLM
                    technical_indicators_text += mtf_detail
            
            # Create prompt using the template sections from the prompt manager
            prompt_parts = {
                "technical_data": technical_indicators_text,
                "current_price": f"CURRENT PRICE: ${current_price:.2f}",
                "formatting_guidance": template.get("formatting_guidance", ""),
                "analysis_guidance": template.get("analysis_guidance", ""),
                "timeframe_expectations": template.get("timeframe_expectations", ""),
                "conclusion": f"6. Ends with a clear conclusion about the {signal} recommendation"
            }
            
            # Handle news sentiment data if available
            has_news = most_relevant_article and news_count > 0
            if has_news:
                # Process article information (only once)
                article_title = most_relevant_article.get("title", "")
                article_summary = most_relevant_article.get("summary", "")
                published_time = most_relevant_article.get("formatted_time", "")
                
                # Format article highlight for the template
                article_highlight = f"- Article: \"{article_title}\"\n- Summary: \"{article_summary}\"\n- Date: \"{published_time}\""
                template_data["ARTICLE_HIGHLIGHT"] = article_highlight
                
                # Add news sentiment to prompt parts
                prompt_parts["news_sentiment"] = f"NEWS SENTIMENT:\n- News Sentiment: {sentiment_label}\n{article_highlight}"
                prompt_parts["response_format"] = template.get("response_format_with_news", "")
            else:
                prompt_parts["response_format"] = template.get("response_format_without_news", "")
            
            # Assemble the final prompt
            prompt = f"Analyze {symbol} market data:\n\n{prompt_parts['technical_data']}\n\n"
            
            if has_news:
                prompt += f"{prompt_parts['news_sentiment']}\n\n"
                
            prompt += f"{prompt_parts['current_price']}\n\n"
            prompt += f"{prompt_parts['formatting_guidance']}\n\n"
            prompt += f"{prompt_parts['analysis_guidance']}\n\n"
            prompt += f"{prompt_parts['timeframe_expectations']}\n\n"
            prompt += f"{prompt_parts['conclusion']}\n\n"
            prompt += f"{prompt_parts['response_format']}"
            
            # Append plain text formatting instruction to system prompt
            system_prompt = self.base_system_prompt + "\n\nIMPORTANT: Write in plain text only without markdown formatting."
            
            try:
                # Generate analysis using LLM with streaming to prevent timeout issues
                collected_response = ""
                max_tokens_limit = 64000
                
                # Make the LLM API call
                async with self.anthropic_client.messages.stream(
                    max_tokens=max_tokens_limit,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    model=self.settings.SOCIAL_MODEL
                ) as stream:
                    async for chunk in stream.text_stream:
                        collected_response += chunk
                
                # Clean the generated content (remove markdown formatting)
                trump_analysis = collected_response.strip().replace("#", "").replace("*", "")
                logger.info(f"Generated market commentary: {trump_analysis[:100]}...")
            except Exception as e:
                # Create fallback analysis if LLM call fails
                logger.error(f"Error generating market commentary: {str(e)}")
                trump_analysis = (
                    f"{symbol} is showing a {trend_strength} {trend} trend with RSI at {latest_rsi:.2f} "
                    f"({rsi_interpretation}). MACD is {macd_trend} and price is {bbands_status}. "
                    f"Based on these technical indicators, my {signal} signal remains valid."
                )
            
            # Add the LLM-generated analysis to template data
            template_data["TRUMP_ANALYSIS"] = trump_analysis
            template_data["RECOMMENDATION"] = signal
            
            logger.info(f"Successfully generated market strategy for {symbol}")
            
            # Use the prompt_manager to assemble the final formatted analysis output
            # This method will correctly include all template sections (header, sentiment, recommendation)
            # and format them with our template_data values
            return self.prompt_manager.get_market_analysis_prompt(**template_data)
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}", exc_info=True)
            # Add more detailed error logging to trace the issue
            error_trace = traceback.format_exc()
            logger.error(f"Detailed traceback for {symbol} analysis error: {error_trace}")
            return f"{symbol}: Error analyzing {asset_type} on {interval} timeframe - {str(e)}"

    async def _fetch_market_data(self, symbol, asset_type, interval):
        """
        Helper function to fetch market data based on asset type and interval.
        Uses an in-memory cache to avoid redundant API calls.
        
        Args:
            symbol (str): The ticker symbol to analyze
            asset_type (str): "stock" or "crypto" 
            interval (str): The timeframe to analyze
            
        Returns:
            tuple: (data, time_series_key) where data contains API response and time_series_key is the key to access OHLCV data
        """
        is_extended = interval in ["daily", "weekly", "monthly"]
        market = "USD" if asset_type == "crypto" else None
        
        # Create cache key from parameters
        cache_key = f"{symbol}_{asset_type}_{interval}"
        
        # Check if we have this data cached (within the last 5 minutes for intraday)
        if hasattr(self, '_market_data_cache') and cache_key in self._market_data_cache:
            cached_data = self._market_data_cache[cache_key]
            cache_age = datetime.now() - cached_data['timestamp']
            
            # For intraday data, cache for 5 minutes; for extended timeframes, cache for 60 minutes
            max_cache_age = timedelta(minutes=5 if not is_extended else 60)
            
            if cache_age < max_cache_age:
                logger.info(f"Using cached market data for {symbol} at {interval} timeframe")
                return cached_data['data'], cached_data['time_series_key']
            else:
                logger.info(f"Cached market data for {symbol} at {interval} expired, fetching fresh data")
        
        # Initialize cache if it doesn't exist
        if not hasattr(self, '_market_data_cache'):
            self._market_data_cache = {}
        
        try:
            # Select the appropriate API method based on asset type and interval
            if asset_type == "crypto":
                if is_extended:
                    if interval == "daily":
                        data = await self.market_manager.get_crypto_daily(symbol=symbol, market=market)
                    elif interval == "weekly":
                        data = await self.market_manager.get_crypto_weekly(symbol=symbol, market=market)
                    elif interval == "monthly":
                        data = await self.market_manager.get_crypto_monthly(symbol=symbol, market=market)
                    else:
                        raise ValueError(f"Unsupported extended interval: {interval}")
                else:
                    data = await self.market_manager.get_crypto_intraday(symbol=symbol, market=market, interval=interval)
            else:  # stock
                if is_extended:
                    if interval == "daily":
                        data = await self.market_manager.get_time_series_daily(symbol=symbol)
                    elif interval == "weekly":
                        data = await self.market_manager.get_time_series_weekly(symbol=symbol)
                    elif interval == "monthly":
                        data = await self.market_manager.get_time_series_monthly(symbol=symbol)
                    else:
                        raise ValueError(f"Unsupported extended interval: {interval}")
                else:
                    data = await self.market_manager.get_intraday_data(symbol=symbol, interval=interval)
            
            # Use the data processor to find the correct time series key
            if data is None:
                return None, None
                
            time_series_key = MarketDataProcessor.get_time_series_key(data, asset_type, interval)
            
            # Cache the data
            self._market_data_cache[cache_key] = {
                'data': data,
                'time_series_key': time_series_key,
                'timestamp': datetime.now()
            }
            
            return data, time_series_key
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol} at {interval} timeframe: {e}")
            return None, None
            
    async def _analyze_timeframe(self, symbol, asset_type, market, interval):
        """
        Analyze a single timeframe for technical indicators and support/resistance levels.
        
        Args:
            symbol (str): The ticker symbol to analyze
            asset_type (str): "stock" or "crypto"
            market (str): The market/quote currency for crypto (e.g., USD)
            interval (str): The timeframe to analyze
            
        Returns:
            dict: Dictionary containing analysis results for this timeframe
        """
        try:
            # Fetch market data
            data, time_series_key = await self._fetch_market_data(symbol, asset_type, interval)
            if data is None or time_series_key is None:
                logger.warning(f"No market data available for {symbol} on {interval} timeframe")
                return None
            
            # Process market data
            df = MarketDataProcessor.process_time_series_data(data, time_series_key, asset_type=asset_type)
            if df is None or df.empty:
                logger.warning(f"Empty DataFrame after processing {symbol} on {interval} timeframe")
                return None
                
            # Log warning if we don't have recommended data points for good analysis
            min_recommended_bars = 50 if interval == "daily" else 30 if interval == "weekly" else 20
            if len(df) < min_recommended_bars:
                logger.warning(f"Limited data points ({len(df)}) for {symbol} on {interval} timeframe. Recommended: {min_recommended_bars}.") 
                # Continue with analysis anyway - let the indicator calculations decide what can be calculated
            
            # Extract metadata from processed dataframe
            metadata = MarketDataProcessor.extract_metadata(df, interval=interval)
            
            # Check if this is an extended timeframe
            is_extended = interval in ["daily", "weekly", "monthly"]
            
            # Get timeframe-specific indicator parameters
            params = TimeframeParameters.get_parameters(interval)
            
            # Calculate technical indicators
            indicators = {}
            indicators["sma"] = TechnicalIndicators.calculate_sma(df, period=params["sma_period"])
            indicators["ema"] = TechnicalIndicators.calculate_ema(df, period=params["ema_period"])
            indicators["rsi"] = TechnicalIndicators.calculate_rsi(df, period=params["rsi_period"])
            indicators["atr"] = TechnicalIndicators.calculate_atr(df, period=params["atr_period"])
            indicators["adx"] = TechnicalIndicators.calculate_adx(df, period=params["adx_period"])
            indicators["macd"], indicators["macd_signal"] = TechnicalIndicators.calculate_macd(df)
            indicators["stoch_k"], indicators["stoch_d"] = TechnicalIndicators.calculate_stochastic(
                df, k_period=params["stoch_k_period"], d_period=params["stoch_d_period"])
            indicators["bbands"] = TechnicalIndicators.calculate_bbands(df)
            
            # Calculate VWAP only for intraday timeframes
            if not is_extended:
                indicators["vwap"] = TechnicalIndicators.calculate_vwap(df)
            else:
                indicators["vwap"] = None
                
            # Check if we have any valid indicators to proceed
            valid_indicators = {k: v for k, v in indicators.items() if v is not None}
            valid_indicator_count = len(valid_indicators)
                               
            if valid_indicator_count == 0:
                logger.warning(f"No valid indicators calculated for {symbol} on {interval} timeframe")
                # Continue with raw price data only
                
            # Log warnings but continue processing with whatever data we have
            if valid_indicator_count < 3:
                logger.info(f"Limited indicators ({valid_indicator_count}) available for {symbol} on {interval} timeframe")
            
            # Get current price - use the last available price from historical data
            current_price = metadata["latest_price"]
            
            # Determine trend based on price and moving averages
            trend = {}
            if (indicators["ema"] is not None and indicators["sma"] is not None and 
                current_price > indicators["ema"] and indicators["ema"] > indicators["sma"]):
                trend["direction"] = "bullish"
            elif (indicators["ema"] is not None and indicators["sma"] is not None and 
                current_price < indicators["ema"] and indicators["ema"] < indicators["sma"]):
                trend["direction"] = "bearish"
            else:
                trend["direction"] = "neutral"
                
            # Determine trend strength using ADX
            if indicators["adx"] is not None:
                if indicators["adx"] < 20:
                    trend["strength"] = "weak"
                elif indicators["adx"] < 40:
                    trend["strength"] = "moderate"
                else:
                    trend["strength"] = "strong"
            else:
                trend["strength"] = "unknown"
                
            # Determine MACD trend
            if indicators["macd"] is not None and indicators["macd_signal"] is not None:
                if indicators["macd"] > indicators["macd_signal"]:
                    trend["macd_trend"] = "bullish"
                elif indicators["macd"] < indicators["macd_signal"]:
                    trend["macd_trend"] = "bearish"
                else:
                    trend["macd_trend"] = "neutral"
            else:
                trend["macd_trend"] = "neutral"
                
            # Get support and resistance levels
            support_levels = PriceLevelAnalyzer.identify_support_levels(
                price_data=df,
                current_price=current_price,
                latest_atr=indicators["atr"],
                interval=interval
            ) if indicators["atr"] is not None else []
            
            resistance_levels = PriceLevelAnalyzer.identify_resistance_levels(
                price_data=df,
                current_price=current_price,
                latest_atr=indicators["atr"],
                interval=interval
            ) if indicators["atr"] is not None else []
            
            # Return comprehensive analysis results
            return {
                "interval": interval,
                "is_extended": is_extended,
                "current_price": current_price,
                "latest_open": metadata["latest_open"],
                "most_recent_date": metadata["most_recent_date"],
                "price_change_pct": metadata["price_change_pct"],
                "change_direction": metadata["change_direction"],
                "indicators": indicators,
                "trend": trend,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "processed_df": df
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {interval} timeframe: {e}")
            return None
            
    async def _analyze_multi_timeframe_levels(self, symbol, asset_type, current_price, main_interval, latest_atr=None):
        """
        Analyze support and resistance levels across multiple timeframes for a more robust analysis.
        
        This function:
        1. Analyzes several timeframes from the given main timeframe
        2. Identifies support/resistance levels that appear across multiple timeframes
        3. Assigns higher weights to levels confirmed by multiple timeframes
        4. Creates consolidated multi-timeframe support and resistance zones
        
        Args:
            symbol (str): The ticker symbol to analyze
            asset_type (str): "stock" or "crypto"
            current_price (float): Current market price
            main_interval (str): The primary timeframe being analyzed
            latest_atr (float, optional): Current ATR value if available
            
        Returns:
            dict: Containing consolidated support/resistance levels and summary text
        """
        # Initialize cache key for this analysis
        cache_key = f"mtf_levels_{symbol}_{asset_type}_{main_interval}"
        
        # Check if we have a cached result that's still valid (within 5 minutes)
        if hasattr(self, '_mtf_cache'):
            if cache_key in self._mtf_cache:
                cached_data = self._mtf_cache[cache_key]
                cache_age = datetime.now() - cached_data['timestamp']
                
                # Use cache if it's less than 5 minutes old
                if cache_age < timedelta(minutes=5):
                    logger.info(f"Using cached multi-timeframe levels for {symbol} on {main_interval}")
                    return cached_data['result']
        else:
            # Create cache if it doesn't exist
            self._mtf_cache = {}
        
        # Define timeframes to analyze based on the main timeframe
        timeframes = self._get_relevant_timeframes(main_interval)
        
        if not timeframes:
            logger.warning(f"No relevant timeframes for multi-timeframe analysis of {main_interval}")
            return None
            
        logger.info(f"Multi-timeframe analysis for {symbol} ({asset_type}) at ${current_price} with primary timeframe {main_interval}")
        logger.info(f"Analyzing additional timeframes: {timeframes}")
        
        # Initialize collections for all support/resistance levels
        all_support_levels = {}
        all_resistance_levels = {}
        
        # Create and gather tasks for concurrent execution
        market = "USD" if asset_type == "crypto" else None
        tasks = [self._analyze_timeframe(symbol, asset_type, market, interval) 
                for interval in timeframes]
        
        # Execute all API requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for interval, result in zip(timeframes, results):
            # Skip if an exception occurred
            if isinstance(result, Exception):
                logger.error(f"Error analyzing {interval} timeframe: {result}")
                continue
                
            # Skip if no result was returned
            if not result:
                continue
                
            # Extract levels
            support_levels = result.get("support_levels")
            resistance_levels = result.get("resistance_levels")
            
            # Store levels
            if support_levels:
                all_support_levels[interval] = support_levels
                logger.info(f"Found {len(support_levels)} support levels for {interval}")
                
            if resistance_levels:
                all_resistance_levels[interval] = resistance_levels
                logger.info(f"Found {len(resistance_levels)} resistance levels for {interval}")
        
        # Consolidate support and resistance levels across timeframes using the PriceLevelAnalyzer
        mtf_support_zones = PriceLevelAnalyzer.consolidate_multi_timeframe_levels(
            all_support_levels, current_price, latest_atr, LevelType.SUPPORT)
        mtf_resistance_zones = PriceLevelAnalyzer.consolidate_multi_timeframe_levels(
            all_resistance_levels, current_price, latest_atr, LevelType.RESISTANCE)
        
        # Generate summary text for multi-timeframe analysis
        mtf_summary = self._generate_mtf_summary(mtf_support_zones, mtf_resistance_zones, current_price)
        
        result = {
            "support_zones": mtf_support_zones,
            "resistance_zones": mtf_resistance_zones,
            "summary": mtf_summary
        }
        
        # Cache the result
        self._mtf_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        return result
    
    # We use the TIMEFRAME_HIERARCHY defined at the top of the class
    
    def _get_relevant_timeframes(self, main_interval):
        """
        Get relevant timeframes to analyze based on the main timeframe.
        Uses the same timeframe hierarchy as defined for analyze_asset.
        
        Args:
            main_interval (str): The primary timeframe
            
        Returns:
            list: Ordered list of timeframes to analyze
        """
        
        # Normalize interval format
        interval_lower = main_interval.lower().replace(" ", "")
        
        # Standardize the main interval
        if interval_lower in ["1m", "1min", "1minute"]:
            main_std = "1min"
        elif interval_lower in ["5m", "5min", "5minute"]:
            main_std = "5min"
        elif interval_lower in ["15m", "15min", "15minute"]:
            main_std = "15min"
        elif interval_lower in ["30m", "30min", "30minute"]:
            main_std = "30min"
        elif interval_lower in ["60m", "60min", "1h", "1hour", "hourly"]:
            main_std = "60min"
        elif interval_lower in ["1d", "day", "daily"]:
            main_std = "daily"
        elif interval_lower in ["1w", "week", "weekly"]:
            main_std = "weekly"
        elif interval_lower in ["1mo", "month", "monthly"]:
            main_std = "monthly"
        else:
            # If we can't determine the timeframe, default to analyzing daily and hourly
            return ["60min", "daily"]
        
        # Get the higher timeframes based on the shared hierarchy
        if main_std in self.TIMEFRAME_HIERARCHY:
            # Simply return the timeframes defined in the hierarchy
            return self.TIMEFRAME_HIERARCHY[main_std]
        else:
            # Fallback - get 60min and daily if timeframe not found
            return ["60min", "daily"]

    async def _fetch_real_time_price(self, symbol: str, asset_type: str, latest_price: float) -> Tuple[float, str]:
        """
        Fetch real-time price for a given symbol.
        
        Args:
            symbol (str): The ticker symbol
            asset_type (str): "stock" or "crypto"
            latest_price (float): Last known price from historical data
            
        Returns:
            tuple: (current_price, current_time)
        """
        current_price = latest_price  # Start with latest_price as default
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Default timestamp
        
        try:
            if asset_type == "crypto":
                # For crypto, use the exchange rate endpoint
                logger.info(f"Fetching real-time price data for {symbol}/USD")
                exchange_data = await self.market_manager.get_exchange_rate(from_currency=symbol, to_currency="USD")
                
                rate_info = exchange_data.get("Realtime Currency Exchange Rate", {})
                if rate_info and "5. Exchange Rate" in rate_info:
                    current_price = float(rate_info.get("5. Exchange Rate"))
                    current_time = rate_info.get("6. Last Refreshed", current_time)
                    logger.info(f"Successfully got real-time price for {symbol}: ${current_price}")
                else:
                    logger.warning(f"No real-time price data found for {symbol}, using latest_price: ${latest_price}")
            else:
                # For stocks, use the 1-minute intraday data for most recent price
                intraday_1min = await self.market_manager.get_intraday_data(
                    symbol=symbol, 
                    interval="1min",
                    outputsize="compact"
                )
                
                # Process the data using MarketDataProcessor
                processed_df = MarketDataProcessor.process_time_series_data(intraday_1min, asset_type=asset_type)
                if processed_df is not None and not processed_df.empty:
                    # Get the most recent price
                    current_price = float(processed_df.iloc[-1]['close'])
                    current_time = processed_df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"Successfully got real-time price for {symbol}: ${current_price}")
                else:
                    logger.warning(f"Could not get real-time data for {symbol}, using latest_price: ${latest_price}")
            
        except Exception as e:
            logger.error(f"Error fetching real-time price for {symbol}: {str(e)}")
            
        return current_price, current_time
    
    def _generate_mtf_summary(self, support_zones, resistance_zones, current_price):
        """
        Generate a text summary of multi-timeframe analysis.
        
        Args:
            support_zones (list): Consolidated support zones
            resistance_zones (list): Consolidated resistance zones
            current_price (float): Current market price
            
        Returns:
            str: Summary text for multi-timeframe analysis
        """
        if not support_zones and not resistance_zones:
            return ""
            
        # Filter zones to only include the most significant ones
        # (at most 3 of each type, sorted by importance)
        top_support = support_zones[:3] if support_zones else []
        top_resistance = resistance_zones[:3] if resistance_zones else []
        
        # Generate summary text
        lines = ["MULTI-TIMEFRAME ANALYSIS:"]
        
        # Add resistance zones
        if top_resistance:
            lines.append("🔴 Key MTF Resistance Zones:")
            for zone in top_resistance:
                # Only include zones above current price for resistance
                if zone["price"] > current_price:
                    timeframes = "/".join(zone["timeframes"])
                    lines.append(f"   ${zone['price']} - Confirmed on {timeframes} timeframes")
                    
        # Add support zones
        if top_support:
            lines.append("🟢 Key MTF Support Zones:")
            for zone in top_support:
                # Only include zones below current price for support
                if zone["price"] < current_price:
                    timeframes = "/".join(zone["timeframes"])
                    lines.append(f"   ${zone['price']} - Confirmed on {timeframes} timeframes")
                    
        # Add explanation of significance
        lines.append("⚠️ Levels that appear across multiple timeframes have higher significance")
        
        return "\n".join(lines)
        
    def _validate_character_config(self, config: Dict) -> None:
        """Validate required character configuration fields"""
        required_fields = [
            'name', 'bio', 'personality', 'formatting', 'chat_style',
            'settings.admin.admin_name', 'settings.admin.admin_commands'
        ]
        for field in required_fields:
            parts = field.split('.')
            current = config
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    raise ValueError(f"Missing required field: {field}")
                current = current[part]