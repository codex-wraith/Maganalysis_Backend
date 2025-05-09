from market.indicators import TechnicalIndicators, TimeframeParameters
from market.price_levels import PriceLevelAnalyzer, LevelType
from market.formatters import DataFormatter
from mcp.server.fastmcp.exceptions import ToolError
import logging
import json
import os
import time
import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

# Module-level MCP tool definitions
# These functions are available for direct import and for MCP registration

# We'll use these later but can't import directly due to circular imports
# app and get_app will be imported only when needed

# Global variables for accessing app state
market_manager = None
http_session = None

def init_global_dependencies():
    """Initialize global dependencies for the module"""
    global market_manager, http_session
    # Get app state if not already initialized
    if not market_manager or not http_session:
        try:
            # Import here to avoid circular import
            from main import get_app
            app_instance = get_app()
            if app_instance:
                market_manager = app_instance.state.market_manager
                http_session = app_instance.state.http
        except Exception as e:
            logger.error(f"Failed to initialize global dependencies: {e}")
            # Will use parameters passed to register_market_tools instead

# Define the tool functions at module level
@mcp.tool()
async def get_raw_market_data(symbol: str, asset_type: str = "stock", timeframe: str = "daily"):
    """
    Get raw market data for a symbol.
    
    Args:
        symbol: The stock or crypto symbol (e.g., AAPL, BTC)
        asset_type: Either "stock" or "crypto"
        timeframe: One of "1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"
        
    Returns:
        Raw dictionary containing the market data
    """
    global market_manager, http_session
    
    try:
        # Initialize dependencies if needed
        if not market_manager:
            init_global_dependencies()
            
        # Get app state if not available globally
        if not market_manager:
            # Import here to avoid circular imports
            from main import app as main_app
            mm = main_app.state.market_manager
            http = main_app.state.http
            cache = main_app.state.cache
        else:
            mm = market_manager
            http = http_session
            # Import app only when needed
            from main import app as main_app
            cache = main_app.state.cache
        
        # Determine if this is a crypto symbol
        is_crypto = asset_type.lower() == "crypto"
        
        # Normalize timeframe parameter
        valid_timeframes = ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"]
        if timeframe not in valid_timeframes:
            timeframe = "daily"  # Default to daily if invalid timeframe
            
        # Cache key for this data
        key = f"{symbol}-{timeframe}"
        ts_now, cached = cache.get(key, (0, None))
        if time.time() - ts_now < 60:
            return cached
            
        # Determine if this is extended timeframe or intraday
        is_extended = timeframe in ["daily", "weekly", "monthly"]
        
        # Get data based on asset type and timeframe
        if is_crypto:
            if is_extended:
                if timeframe == "daily":
                    data = await mm.get_crypto_daily(symbol, "USD", http_session=http)
                elif timeframe == "weekly":
                    data = await mm.get_crypto_weekly(symbol, "USD", http_session=http)
                else:  # monthly
                    data = await mm.get_crypto_monthly(symbol, "USD", http_session=http)
            else:
                data = await mm.get_crypto_intraday(symbol, "USD", interval=timeframe, http_session=http)
        else:
            if is_extended:
                if timeframe == "daily":
                    data = await mm.get_time_series_daily(symbol, http_session=http)
                elif timeframe == "weekly":
                    data = await mm.get_time_series_weekly(symbol, http_session=http)
                else:  # monthly
                    data = await mm.get_time_series_monthly(symbol, http_session=http)
            else:
                data = await mm.get_intraday_data(symbol, interval=timeframe, http_session=http)
        
        # Slice data for performance and cache
        series_key = next((k for k in data.keys() if "Time Series" in k or "Digital Currency" in k), None)
        if series_key and series_key in data:
            sliced = dict(list(data[series_key].items())[:100])  # last 100 candles
            data[series_key] = sliced
        
        # Cache the result
        cache[key] = (time.time(), data)
        return data
            
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        raise ToolError(f"Failed to get market data: {str(e)}")

@mcp.tool()
async def get_market_data(symbol: str, asset_type: str = "stock", timeframe: str = "daily"):
    """
    Get market data for a symbol.
    
    Args:
        symbol: The stock or crypto symbol (e.g., AAPL, BTC)
        asset_type: Either "stock" or "crypto"
        timeframe: One of "1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"
        
    Returns:
        Dictionary containing the market data
    """
    return await get_raw_market_data(symbol, asset_type, timeframe)

@mcp.tool()
async def get_technical_indicators(symbol: str, asset_type: str = "stock", timeframe: str = "daily"):
    """
    Calculate technical indicators for a stock or crypto.
    
    Args:
        symbol: The stock or crypto symbol
        asset_type: Either "stock" or "crypto"
        timeframe: The timeframe for analysis
        
    Returns:
        Dictionary containing the technical indicators
    """
    try:
        # Get market data
        data = await get_raw_market_data(symbol, asset_type, timeframe)
        
        # Process the data
        from market.data_processor import MarketDataProcessor
        
        # Find the time series key
        time_series_key = next((k for k in data.keys() if "Time Series" in k or "Digital Currency" in k), None)
        
        if not time_series_key or not data.get(time_series_key):
            raise ToolError(f"No time series data available for {symbol}")
        
        # Process the data
        df = MarketDataProcessor.process_time_series_data(data, time_series_key, asset_type)
        if df is None or df.empty:
            raise ToolError("Failed to process time series data")
        
        # Get timeframe-specific parameters
        params = TimeframeParameters.get_parameters(timeframe)
        
        # Calculate indicators
        indicators = {}
        
        # Calculate RSI
        indicators["rsi"] = TechnicalIndicators.calculate_rsi(df, period=params["rsi_period"])
        
        # Calculate MACD
        macd, signal = TechnicalIndicators.calculate_macd(df)
        indicators["macd"] = {"value": macd, "signal": signal}
        
        # Calculate Stochastic
        k_value, d_value = TechnicalIndicators.calculate_stochastic(
            df, k_period=params["stoch_k_period"], d_period=params["stoch_d_period"])
        indicators["stochastic"] = {"k": k_value, "d": d_value}
        
        # Calculate Bollinger Bands
        bbands = TechnicalIndicators.calculate_bbands(df, period=params["bbands_period"])
        indicators["bbands"] = bbands
        
        # Calculate Moving Averages
        indicators["sma"] = TechnicalIndicators.calculate_sma(df, period=params["sma_period"])
        indicators["ema"] = TechnicalIndicators.calculate_ema(df, period=params["ema_period"])
        
        # Calculate ATR
        indicators["atr"] = TechnicalIndicators.calculate_atr(df, period=params["atr_period"])
        
        # Calculate ADX
        indicators["adx"] = TechnicalIndicators.calculate_adx(df, period=params["adx_period"])
        
        # Calculate VWAP
        indicators["vwap"] = TechnicalIndicators.calculate_vwap(df)
        
        # Get latest price data
        latest_data = df.iloc[-1]
        current_price = float(latest_data['close'])
        
        # Calculate change
        if len(df) > 1:
            prev_close = float(df.iloc[-2]['close'])
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
        else:
            change = 0
            change_percent = 0
        
        # Determine indicator signals
        rsi_signal = "oversold" if indicators["rsi"] < 30 else "overbought" if indicators["rsi"] > 70 else "neutral"
        macd_signal = "bullish" if indicators["macd"]["value"] > indicators["macd"]["signal"] else "bearish"
        stoch_signal = "oversold" if indicators["stochastic"]["k"] < 20 else "overbought" if indicators["stochastic"]["k"] > 80 else "neutral"
        
        # Determine trend signals from moving averages
        sma_signal = "bullish" if current_price > indicators["sma"] else "bearish"
        ema_signal = "bullish" if current_price > indicators["ema"] else "bearish"
        
        # Calculate ADX trend strength signal
        adx_signal = "weak" if indicators["adx"] < 20 else "strong" if indicators["adx"] > 25 else "moderate"
        
        # Format indicator values for display using DataFormatter
        formatted_indicators = {}
        for key, value in indicators.items():
            if isinstance(value, dict):
                # For dictionary indicators like bbands
                formatted_indicators[key] = {k: DataFormatter.format_price(v) for k, v in value.items()}
            else:
                # For simple indicators
                formatted_indicators[key] = DataFormatter.format_price(value)
        
        # Return comprehensive indicator data
        return {
            "indicators": indicators,
            "formatted_indicators": formatted_indicators,
            "signals": {
                "rsi": rsi_signal,
                "macd": macd_signal,
                "stochastic": stoch_signal,
                "sma": sma_signal,
                "ema": ema_signal,
                "adx": adx_signal
            },
            "price_data": {
                "current": current_price,
                "change": change,
                "change_percent": change_percent
            }
        }
    except ToolError as e:
        # Re-raise ToolError as-is
        raise
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {symbol}: {e}")
        raise ToolError(f"Failed to calculate technical indicators: {str(e)}")

@mcp.tool()
async def get_news_sentiment(symbol: str):
    """
    Get news sentiment data for a specific symbol.
    
    Args:
        symbol: The stock or crypto symbol
        
    Returns:
        Dictionary containing sentiment data and news articles
    """
    global market_manager, http_session
    
    try:
        # Initialize dependencies if needed
        if not market_manager:
            init_global_dependencies()
            
        # Get app state if not available globally
        if not market_manager:
            app_instance = app
            mm = app_instance.state.market_manager
            http = app_instance.state.http
        else:
            mm = market_manager
            http = http_session
        
        # Fetch news sentiment from market manager
        sentiment_data = await mm.get_news_sentiment(
            symbol, http_session=http
        )
        
        if sentiment_data and "feed" in sentiment_data:
            articles = sentiment_data["feed"]
            
            # Calculate overall sentiment
            total_score = 0
            positive = 0
            neutral = 0
            negative = 0
            most_relevant_article = None
            
            for article in articles:
                score = article.get("overall_sentiment_score", 0)
                total_score += score
                
                if score > 0.25:
                    positive += 1
                elif score < -0.25:
                    negative += 1
                else:
                    neutral += 1
                    
                # Get most relevant article for this ticker
                if not most_relevant_article and "ticker_sentiment" in article:
                    for ticker_sent in article["ticker_sentiment"]:
                        if ticker_sent.get("ticker") == symbol or ticker_sent.get("ticker") == f"CRYPTO:{symbol}":
                            relevance = float(ticker_sent.get("relevance_score", 0))
                            if relevance >= 0.5:  # Only include highly relevant articles
                                most_relevant_article = {
                                    "title": article.get("title", ""),
                                    "summary": article.get("summary", ""),
                                    "url": article.get("url", ""),
                                    "published": article.get("time_published", ""),
                                    "relevance_score": relevance,
                                    "sentiment_score": score
                                }
                                break
            
            count = len(articles)
            avg_score = total_score / count if count > 0 else 0
            
            # Normalize to 0-100 scale
            normalized_score = (avg_score + 1) * 50
            
            # Calculate percentages
            pos_pct = int((positive / count) * 100) if count > 0 else 0
            neu_pct = int((neutral / count) * 100) if count > 0 else 0
            neg_pct = int((negative / count) * 100) if count > 0 else 0
            
            # Determine sentiment label
            if normalized_score >= 65:
                sentiment_label = "BULLISH"
            elif normalized_score <= 35:
                sentiment_label = "BEARISH"
            else:
                sentiment_label = "NEUTRAL"
            
            return {
                "symbol": symbol,
                "sentiment": {
                    "score": normalized_score,
                    "label": sentiment_label,
                    "positive": pos_pct,
                    "neutral": neu_pct,
                    "negative": neg_pct
                },
                "news_count": count,
                "most_relevant_article": most_relevant_article,
                "feed": articles[:5]  # Limit to 5 articles to keep context size manageable
            }
    except Exception as e:
        logger.error(f"Error getting news sentiment for {symbol}: {e}")
        
    # Return data indicating no sentiment available if API call failed
    return {
        "symbol": symbol,
        "sentiment": {
            "score": None,
            "label": "NEUTRAL",
            "positive": None,
            "neutral": None,
            "negative": None
        },
        "news_count": 0,
        "feed": []
    }

@mcp.tool()
async def search_market_information(query: str, search_type: str = "web"):
    """
    Search for market information using the Tavily API.
    
    Args:
        query: The search query about market, finance, or trading information
        search_type: Type of search - "web" for general search or "news" for recent news
        
    Returns:
        Dictionary containing search results and sources
    """
    try:
        # Import here to avoid circular imports
        from main import app as main_app
        agent = main_app.state.agent
        
        # Validate search type
        if search_type not in ["web", "news"]:
            search_type = "web"  # Default to web search
        
        # Configure search parameters
        search_params = {
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_domains": [
                "finance.yahoo.com", 
                "bloomberg.com", 
                "wsj.com", 
                "ft.com", 
                "cnbc.com",
                "investopedia.com",
                "seekingalpha.com",
                "morningstar.com",
                "markets.businessinsider.com",
                "tradingview.com"
            ],
            "max_results": 10
        }
        
        # Set time range for news search
        if search_type == "news":
            search_params["max_tokens"] = 8000
            search_params["search_depth"] = "basic"
            search_params["include_raw_content"] = False
            
        # Execute the search
        search_response = await agent.tavily_client.search(**search_params)
        
        # Extract and format results
        results = []
        
        if search_response and "results" in search_response:
            for result in search_response["results"][:5]:  # Limit to top 5 results
                results.append({
                    "title": result.get("title", "No title"),
                    "url": result.get("url", ""),
                    "content": result.get("content", "")[:500] + "..." if len(result.get("content", "")) > 500 else result.get("content", ""),
                    "score": result.get("relevance_score", 0)
                })
        
        # Return formatted search results
        return {
            "query": query,
            "search_type": search_type,
            "answer": search_response.get("answer", "No direct answer found."),
            "results": results,
            "timestamp": datetime.now(UTC).isoformat()
        }
            
    except Exception as e:
        # Return error information
        logger.error(f"Error in search_market_information: {str(e)}")
        raise ToolError(f"Search failed: {str(e)}")

# Registration function now just stores references to app dependencies
def register_market_tools(mcp_instance, app_instance, market_manager_instance, http_session_instance, agent_instance):
    """
    Register market tools with the MCP server.
    This function now primarily stores references to app dependencies for use by the module-level functions.
    """
    global market_manager, http_session
    
    # Store references to app dependencies
    market_manager = market_manager_instance
    http_session = http_session_instance
    
    logger.info("Market tools registered (now defined at module level)")
    
    # Nothing else to do since tools are decorated at module level

async def add_market_analysis_to_context(agent, messages_list=None, symbol=None, asset_type=None, timeframe=None, context=None):
    """
    Add market analysis data to either a message list or MCP context.
    
    Args:
        agent: The CipherAgent instance
        messages_list: Optional list of message dictionaries to append to (new API)
        symbol: The asset symbol to analyze
        asset_type: The asset type (stock or crypto)
        timeframe: The timeframe to analyze
        context: Optional legacy MCP context object (old API)
        
    Returns:
        If messages_list is provided: The market analysis message dictionary
        If context is provided: None (modifies the context in place)
        If neither is provided: The market analysis string
    """
    try:
        # Get the MCP instance
        from mcp_server import mcp
        
        # Get technical analysis data
        tech_data = await mcp.tools.get_technical_indicators(symbol, asset_type, timeframe)
        price_levels = await mcp.tools.get_price_levels(symbol, asset_type, timeframe)
        sentiment = await mcp.tools.get_news_sentiment(symbol)
        
        # Format market context
        market_context = f"""
        ## Market Analysis Data for {symbol}
        
        Current Price: ${tech_data.get('price_data', {}).get('current', 'Unknown')}
        Timeframe: {timeframe}
        
        ### Technical Indicators:
        - RSI: {tech_data.get('formatted_indicators', {}).get('rsi', 'Unknown')} ({tech_data.get('signals', {}).get('rsi', 'Unknown')})
        - MACD: {tech_data.get('formatted_indicators', {}).get('macd', {}).get('value', 'Unknown')} ({tech_data.get('signals', {}).get('macd', 'Unknown')})
        - Stochastic: {tech_data.get('formatted_indicators', {}).get('stochastic', {}).get('k', 'Unknown')} ({tech_data.get('signals', {}).get('stochastic', 'Unknown')})
        - SMA: {tech_data.get('formatted_indicators', {}).get('sma', 'Unknown')} ({tech_data.get('signals', {}).get('sma', 'Unknown')})
        - EMA: {tech_data.get('formatted_indicators', {}).get('ema', 'Unknown')} ({tech_data.get('signals', {}).get('ema', 'Unknown')})
        - ADX: {tech_data.get('formatted_indicators', {}).get('adx', 'Unknown')} ({tech_data.get('signals', {}).get('adx', 'Unknown')})
        
        ### Support/Resistance Levels:
        Support Levels: {", ".join([level.get("price", "Unknown") for level in price_levels.get('formatted_support', [])])}
        Resistance Levels: {", ".join([level.get("price", "Unknown") for level in price_levels.get('formatted_resistance', [])])}
        
        ### Market Sentiment:
        Sentiment Score: {sentiment.get('sentiment', {}).get('score', 'Unknown')}
        Sentiment Label: {sentiment.get('sentiment', {}).get('label', 'Unknown')}
        News Count: {sentiment.get('news_count', 0)}
        """
        
        # Add most relevant article if available
        if sentiment.get('most_relevant_article'):
            article = sentiment['most_relevant_article']
            market_context += f"""
            ### Most Relevant News Article:
            Title: {article.get('title', 'Unknown')}
            Summary: {article.get('summary', 'No summary available')}
            """
        
        # Handle different API formats
        if context is not None:
            try:
                # Legacy API - attempt to add to context
                context.add_system_message(market_context)
                return None
            except Exception as e:
                logger.error(f"Error adding to context object: {e}")
                # Fall through to return the string
        
        if messages_list is not None and isinstance(messages_list, list):
            # New API - add to message list
            message_dict = {"role": "system", "content": market_context}
            messages_list.append(message_dict)
            return message_dict
        
        # If neither context nor messages_list is provided, return the string
        return market_context
        
    except Exception as e:
        error_msg = f"Note: Could not retrieve complete market analysis for {symbol}. Error: {str(e)}"
        logger.error(f"Error adding market analysis to context: {e}")
        
        # Handle the error based on what was provided
        if context is not None:
            try:
                context.add_system_message(error_msg)
            except:
                pass
        
        if messages_list is not None and isinstance(messages_list, list):
            error_dict = {"role": "system", "content": error_msg}
            messages_list.append(error_dict)
            return error_dict
            
        return error_msg