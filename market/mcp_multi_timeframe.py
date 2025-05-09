import logging
import json
import aiohttp
from typing import Dict, Any, List, Optional
from market.price_levels import PriceLevelAnalyzer, LevelType
from datetime import datetime, UTC
from mcp.server.fastmcp.exceptions import ToolError
from mcp_server import mcp

# Import required app state
from main import app, get_app
from market.mcp_tools import get_technical_indicators, get_news_sentiment
from market.mcp_price_levels import get_price_levels

logger = logging.getLogger(__name__)

# Global variables for accessing app state
market_manager = None
http_session = None

def init_global_dependencies():
    """Initialize global dependencies for the module"""
    global market_manager, http_session
    # Get app state if not already initialized
    if not market_manager or not http_session:
        try:
            app_instance = get_app()
            market_manager = app_instance.state.market_manager
            http_session = app_instance.state.http
        except Exception as e:
            logger.error(f"Failed to initialize global dependencies: {e}")
            # Will use parameters passed to register_market_tools instead

# Define the tool at module level
@mcp.tool()
async def analyze_multi_timeframe(symbol: str, asset_type: str = "stock", primary_timeframe: str = "daily"):
    """
    Perform multi-timeframe analysis for a symbol.
    REPLACEMENT for original _analyze_multi_timeframe_levels method.
    
    Args:
        symbol: The stock or crypto symbol to analyze
        asset_type: "stock" or "crypto"
        primary_timeframe: The main timeframe for analysis
        
    Returns:
        Comprehensive multi-timeframe analysis with support/resistance levels
    """
    try:
        # Define timeframe hierarchy based on CipherAgent's implementation
        timeframe_hierarchy = {
            "1min": ["5min", "15min", "60min", "daily"],
            "5min": ["15min", "30min", "60min"],  
            "15min": ["30min", "60min"],
            "30min": ["60min", 'daily'],
            "60min": ["daily"],
            "daily": ["weekly", "monthly"],
            "weekly": ["monthly"],
            "monthly": []
        }
        
        # Get higher timeframes based on primary timeframe
        higher_timeframes = timeframe_hierarchy.get(primary_timeframe, [])
        
        # Get primary timeframe data using imported function
        primary_data = await get_technical_indicators(symbol, asset_type, primary_timeframe)
        current_price = primary_data.get("price_data", {}).get("current", 0)
        
        # Get price levels across timeframes using imported function
        levels = await get_price_levels(symbol, asset_type, primary_timeframe)
        
        # Get sentiment data using imported function
        sentiment = await get_news_sentiment(symbol)
        
        # Analyze higher timeframes
        higher_tf_data = {}
        for tf in higher_timeframes:
            higher_tf_data[tf] = await get_technical_indicators(symbol, asset_type, tf)
        
        # Generate trade signal based on multi-timeframe analysis
        # Determine the primary signal
        rsi = primary_data.get("indicators", {}).get("rsi")
        macd_value = primary_data.get("indicators", {}).get("macd", {}).get("value")
        macd_signal = primary_data.get("indicators", {}).get("macd", {}).get("signal")
        stoch_k = primary_data.get("indicators", {}).get("stochastic", {}).get("k")
        
        # Calculate trend score (similar to analyze_asset method)
        trend_indicators = [
            1 if primary_data.get("signals", {}).get("sma") == "bullish" else -1,
            1 if primary_data.get("signals", {}).get("ema") == "bullish" else -1,
            1 if primary_data.get("signals", {}).get("macd") == "bullish" else -1,
            1 if rsi and rsi > 50 else -1,
            1 if stoch_k and stoch_k > 50 else -1
        ]
        
        trend_score = sum(trend_indicators)
        signal = "BUY" if trend_score >= 3 else "SELL" if trend_score <= -3 else "NEUTRAL"
        
        # Generate reasoning
        if signal == "BUY":
            reasoning = "Multiple indicators showing bullish momentum"
        elif signal == "SELL":
            reasoning = "Multiple indicators showing bearish momentum"
        else:
            reasoning = "Mixed signals showing consolidation"
        
        # Enhance with sentiment similar to CipherAgent.enhance_with_sentiment
        if sentiment.get("sentiment", {}).get("score", 50) > 65 and signal == "BUY":
            enhanced_signal = "BUY"
            enhanced_reasoning = reasoning + ", confirmed by bullish news sentiment"
        elif sentiment.get("sentiment", {}).get("score", 50) < 35 and signal == "SELL":
            enhanced_signal = "SELL"
            enhanced_reasoning = reasoning + ", confirmed by bearish news sentiment"
        elif sentiment.get("sentiment", {}).get("score", 50) > 65 and signal == "NEUTRAL":
            enhanced_signal = "SPECULATIVE BUY"
            enhanced_reasoning = reasoning + ", supported by bullish news sentiment"
        elif sentiment.get("sentiment", {}).get("score", 50) < 35 and signal == "NEUTRAL":
            enhanced_signal = "SPECULATIVE SELL"
            enhanced_reasoning = reasoning + ", supported by bearish news sentiment"
        else:
            enhanced_signal = signal
            enhanced_reasoning = reasoning
        
        # Return comprehensive analysis
        return {
            "symbol": symbol,
            "asset_type": asset_type,
            "primary_timeframe": primary_timeframe,
            "current_price": current_price,
            "primary_analysis": primary_data,
            "higher_timeframes": higher_tf_data,
            "support_resistance": levels,
            "sentiment": sentiment,
            "signal": enhanced_signal,
            "reasoning": enhanced_reasoning,
            "trend_score": trend_score,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except ToolError as e:
        # Re-raise ToolError as-is
        raise
    except Exception as e:
        logger.error(f"Error in multi-timeframe analysis: {e}")
        raise ToolError(f"Failed to analyze multi-timeframe data: {str(e)}")

# Create an MCP prompt template specifically for multi-timeframe analysis
@mcp.prompt("mtf_analysis")
def mtf_analysis_prompt():
    """System prompt for multi-timeframe analysis."""
    return """
    You are a professional market analyst specializing in multi-timeframe analysis.
    
    When analyzing markets across timeframes, follow these principles:
    
    1. Higher timeframes dictate the overall trend direction
    2. Lower timeframes provide entry and exit signals within the context of the higher timeframe trend
    3. Support and resistance from higher timeframes carry more weight
    4. Price action near confluence zones (where multiple timeframe S/R levels align) is particularly significant
    5. Always prioritize risk management and don't chase trades with poor risk/reward ratios
    
    In your analysis, include:
    - An overview of the higher timeframe trend
    - Key support/resistance levels across multiple timeframes
    - Confluent price zones where multiple timeframes align
    - Technical indicator readings that confirm the multi-timeframe picture
    - Potential entry/exit prices with stop loss recommendations
    - Different targets based on timeframe (swing vs position targets)
    
    Keep your analysis data-driven, balanced, and focus on helping traders see the complete multi-timeframe picture.
    """

# Registration function now just stores references to app dependencies
def register_multi_timeframe_tools(mcp_instance, app_instance, market_manager_instance, http_session_instance, agent_instance):
    """
    Register multi-timeframe tools with the MCP server.
    This function now primarily stores references to app dependencies for use by the module-level functions.
    """
    global market_manager, http_session
    
    # Store references to app dependencies
    market_manager = market_manager_instance
    http_session = http_session_instance
    
    logger.info("Multi-timeframe tools registered (now defined at module level)")