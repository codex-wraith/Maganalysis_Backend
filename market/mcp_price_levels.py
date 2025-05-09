import logging
import json
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from market.price_levels import PriceLevelAnalyzer, LevelType
from market.data_processor import MarketDataProcessor
from market.formatters import DataFormatter
from datetime import datetime, UTC
from mcp.server.fastmcp.exceptions import ToolError
from mcp_server import mcp

# Import tools we need directly
from market.mcp_tools import get_technical_indicators, get_raw_market_data
# We'll use these later but can't import directly due to circular imports
# app and get_app will be imported only when needed

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
            # Import here to avoid circular import
            from main import get_app
            app_instance = get_app()
            if app_instance:
                market_manager = app_instance.state.market_manager
                http_session = app_instance.state.http
        except Exception as e:
            logger.error(f"Failed to initialize global dependencies: {e}")
            # Will use parameters passed to register_market_tools instead

# Module-level MCP tool definitions
@mcp.tool()
async def get_price_levels(symbol: str, asset_type: str = "stock", timeframe: str = "daily"):
    """
    Get support and resistance levels for a symbol across multiple timeframes.
    REPLACEMENT for original price level identification methods.
    
    Args:
        symbol: The stock or crypto symbol
        asset_type: Either "stock" or "crypto"
        timeframe: The primary timeframe for analysis
        
    Returns:
        Dictionary containing support and resistance levels
    """
    try:
        # Get market data from tools (using direct function call)
        tech_data = await get_technical_indicators(symbol, asset_type, timeframe)
        
        # Get current price and ATR
        current_price = tech_data.get("price_data", {}).get("current", 0)
        latest_atr = tech_data.get("indicators", {}).get("atr")
        
        # Get higher timeframes
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
        
        higher_timeframes = timeframe_hierarchy.get(timeframe, [])
        
        # Analyze each timeframe
        all_support_levels = {}
        all_resistance_levels = {}
        
        # Add primary timeframe data
        primary_data = await get_raw_market_data(symbol, asset_type, timeframe)
            
        time_series_key = next((k for k in primary_data.keys() if "Time Series" in k or "Digital Currency" in k), None)
        if time_series_key:
            df = MarketDataProcessor.process_time_series_data(primary_data, time_series_key, asset_type)
            if df is not None and not df.empty:
                # Use all advanced features of PriceLevelAnalyzer
                support_levels = PriceLevelAnalyzer.identify_support_levels(
                    price_data=df,
                    current_price=current_price,
                    latest_atr=latest_atr,
                    interval=timeframe,
                    include_fibonacci=True,
                    include_psychological=True
                )
                resistance_levels = PriceLevelAnalyzer.identify_resistance_levels(
                    price_data=df,
                    current_price=current_price,
                    latest_atr=latest_atr,
                    interval=timeframe,
                    include_fibonacci=True,
                    include_psychological=True
                )
                all_support_levels[timeframe] = support_levels
                all_resistance_levels[timeframe] = resistance_levels
        
        # Add higher timeframe data
        for tf in higher_timeframes:
            tf_data = await get_raw_market_data(symbol, asset_type, tf)
            if "error" in tf_data:
                continue
                
            tf_time_series_key = next((k for k in tf_data.keys() if "Time Series" in k or "Digital Currency" in k), None)
            if tf_time_series_key:
                tf_df = MarketDataProcessor.process_time_series_data(tf_data, tf_time_series_key, asset_type)
                if tf_df is not None and not tf_df.empty:
                    tf_support = PriceLevelAnalyzer.identify_support_levels(
                        price_data=tf_df,
                        current_price=current_price,
                        latest_atr=latest_atr,
                        interval=tf,
                        include_fibonacci=True,
                        include_psychological=True
                    )
                    tf_resistance = PriceLevelAnalyzer.identify_resistance_levels(
                        price_data=tf_df,
                        current_price=current_price,
                        latest_atr=latest_atr,
                        interval=tf,
                        include_fibonacci=True,
                        include_psychological=True
                    )
                    all_support_levels[tf] = tf_support
                    all_resistance_levels[tf] = tf_resistance
        
        # Consolidate levels using PriceLevelAnalyzer's sophisticated methods
        mtf_support_zones = PriceLevelAnalyzer.consolidate_multi_timeframe_levels(
            all_support_levels, current_price, latest_atr, LevelType.SUPPORT)
        mtf_resistance_zones = PriceLevelAnalyzer.consolidate_multi_timeframe_levels(
            all_resistance_levels, current_price, latest_atr, LevelType.RESISTANCE)
        
        # Format levels for readability
        formatted_support = []
        for level in mtf_support_zones:
            formatted_support.append({
                "price": DataFormatter.format_price(level["price"]),
                "timeframes": level["timeframes"],
                "strength": level["strength"],
                "confidence": level["confidence"],
                "distance_percent": f"{level['distance_percent']:.2f}%"
            })
            
        formatted_resistance = []
        for level in mtf_resistance_zones:
            formatted_resistance.append({
                "price": DataFormatter.format_price(level["price"]),
                "timeframes": level["timeframes"],
                "strength": level["strength"],
                "confidence": level["confidence"],
                "distance_percent": f"{level['distance_percent']:.2f}%"
            })
        
        return {
            "consolidated_support": mtf_support_zones,
            "consolidated_resistance": mtf_resistance_zones,
            "formatted_support": formatted_support,
            "formatted_resistance": formatted_resistance,
            "all_timeframes": {
                "support": all_support_levels,
                "resistance": all_resistance_levels
            },
            "timestamp": datetime.now(UTC).isoformat()
        }
    except ToolError as e:
        # Re-raise ToolError as-is
        raise
    except Exception as e:
        logger.error(f"Error getting price levels for {symbol}: {e}")
        raise ToolError(f"Failed to calculate price levels: {str(e)}")

@mcp.tool()
async def get_key_price_zones(symbol: str, asset_type: str = "stock", timeframe: str = "daily"):
    """
    Get only the most important price zones for a symbol, focusing on high confidence levels.
    
    Args:
        symbol: The stock or crypto symbol
        asset_type: Either "stock" or "crypto"
        timeframe: The primary timeframe for analysis
        
    Returns:
        Dictionary containing only the most significant support and resistance zones
    """
    try:
        # Get all levels first
        all_levels = await get_price_levels(symbol, asset_type, timeframe)
        
        # Filter for only high and very high confidence levels
        key_support = [level for level in all_levels["consolidated_support"] 
                      if level["confidence"] in ["high", "very high"]]
        key_resistance = [level for level in all_levels["consolidated_resistance"] 
                          if level["confidence"] in ["high", "very high"]]
        
        # Sort by distance from current price
        key_support.sort(key=lambda x: abs(x["distance_percent"]))
        key_resistance.sort(key=lambda x: abs(x["distance_percent"]))
        
        # Get top 3 levels for each
        top_support = key_support[:3]
        top_resistance = key_resistance[:3]
        
        return {
            "top_support": top_support,
            "top_resistance": top_resistance,
            "symbol": symbol,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "timestamp": datetime.now(UTC).isoformat()
        }
    except ToolError as e:
        # Re-raise ToolError as-is
        raise
    except Exception as e:
        logger.error(f"Error getting key price zones for {symbol}: {e}")
        raise ToolError(f"Failed to identify key price zones: {str(e)}")

# Registration function now just stores references to app dependencies
def register_price_level_tools(mcp_instance, app_instance, market_manager_instance, http_session_instance, agent_instance):
    """
    Register price level tools with the MCP server.
    This function now primarily stores references to app dependencies for use by the module-level functions.
    """
    global market_manager, http_session
    
    # Store references to app dependencies
    market_manager = market_manager_instance
    http_session = http_session_instance
    
    logger.info("Price level tools registered (now defined at module level)")