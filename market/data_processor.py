import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class MarketDataProcessor:
    """
    A utility class to process market data from various API responses into standardized DataFrames
    and calculate technical indicators.
    """
    
    @staticmethod
    def process_time_series_data(
        data: Dict[str, Any], 
        time_series_key: Optional[str] = None,
        asset_type: str = "stock"
    ) -> Optional[pd.DataFrame]:
        """
        Process market data into a standardized pandas DataFrame.
        
        Args:
            data: API response data containing time series information
            time_series_key: Key to access OHLCV data in the response. If None, attempts to find the key.
            asset_type: Type of asset - "stock" or "crypto" 
            
        Returns:
            Standardized pandas DataFrame with OHLCV data or None if processing fails
        """
        try:
            # If time_series_key not provided, attempt to find it
            if not time_series_key:
                # Use list comprehension to find potential keys
                time_series_keys = [key for key in data.keys() if 
                                   "Time Series" in key or 
                                   "Weekly Time Series" in key or 
                                   "Monthly Time Series" in key or
                                   "Digital Currency" in key]
                
                if time_series_keys:
                    time_series_key = time_series_keys[0]
                else:
                    logger.warning("No time series data found in API response")
                    return None
            
            # Check if data is available
            if not data.get(time_series_key):
                logger.warning(f"Empty time series data for key: {time_series_key}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            
            # Standard column mapping for Alpha Vantage API
            # Both stock and crypto use these column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            
            df.rename(columns=lambda x: column_mapping.get(x, x), inplace=True)
            
            # Convert index to datetime and sort
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Convert all columns to numeric in one pass
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Simple validation of required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns in processed data: {missing_columns}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return None
            
    @staticmethod
    def extract_metadata(df: pd.DataFrame, interval: str = "daily") -> Dict[str, Any]:
        """
        Extract key metadata from a processed price DataFrame.
        Includes comprehensive price metrics and timeframe-specific information.
        
        Args:
            df: Processed pandas DataFrame with OHLCV data
            interval: Time interval of the data (e.g., "daily", "5min")
            
        Returns:
            Dictionary with comprehensive metadata including price, volume, change metrics, etc.
        """
        metadata = {
            "latest_price": None,
            "latest_open": None,
            "latest_high": None,
            "latest_low": None,
            "latest_volume": None,
            "most_recent_date": None,
            "price_change_pct": 0,
            "price_change_value": 0,
            "change_direction": "→",
            "has_data": False,
            "data_points": 0,
            "date_range": None,
            "price_range": None,  # Min and max price in the dataset
            "avg_volume": None,
            "interval": interval,
            "highest_price": None,
            "lowest_price": None,
            "daily_change_pct": None  # For intraday data
        }
        
        try:
            if df is None or df.empty:
                return metadata
                
            metadata["has_data"] = True
            metadata["data_points"] = len(df)
            
            # Record most recent date and prices with appropriate formatting
            is_intraday = any(x in interval for x in ["min", "hour", "h", "m"])
            date_format = "%Y-%m-%d %H:%M:%S" if is_intraday else "%Y-%m-%d"
            
            metadata["most_recent_date"] = df.index[-1].strftime(date_format)
            metadata["date_range"] = {
                "start": df.index[0].strftime(date_format),
                "end": df.index[-1].strftime(date_format)
            }
            
            # Latest OHLCV values
            metadata["latest_price"] = float(df.iloc[-1]['close'])
            metadata["latest_open"] = float(df.iloc[-1]['open'])
            
            if 'high' in df.columns:
                metadata["latest_high"] = float(df.iloc[-1]['high'])
            if 'low' in df.columns:
                metadata["latest_low"] = float(df.iloc[-1]['low'])
            if 'volume' in df.columns:
                metadata["latest_volume"] = float(df.iloc[-1]['volume'])
                metadata["avg_volume"] = float(df['volume'].mean())
            
            # Calculate intraday and overall changes
            if len(df) > 1:
                latest_close = float(df.iloc[-1]['close'])
                prev_close = float(df.iloc[-2]['close'])
                
                # Recent bar change
                metadata["price_change_value"] = latest_close - prev_close
                metadata["price_change_pct"] = ((latest_close - prev_close) / prev_close) * 100
                metadata["change_direction"] = "↑" if metadata["price_change_pct"] >= 0 else "↓"
                
                # Daily (open to close) change for intraday data
                if is_intraday:
                    open_price = float(df.iloc[-1]['open'])
                    metadata["daily_change_pct"] = ((latest_close - open_price) / open_price) * 100
            
            # Price range across entire dataset
            if 'high' in df.columns and 'low' in df.columns:
                metadata["highest_price"] = float(df['high'].max())
                metadata["lowest_price"] = float(df['low'].min())
                metadata["price_range"] = {
                    "min": metadata["lowest_price"],
                    "max": metadata["highest_price"],
                    "pct_diff": ((metadata["highest_price"] - metadata["lowest_price"]) / max(metadata["lowest_price"], 0.0000001)) * 100
                }
            
            # Add additional metrics for volatility assessment
            if len(df) >= 14 and 'close' in df.columns:
                # Calculate recent volatility (standard deviation over last 14 periods)
                metadata["recent_volatility"] = float(df['close'].tail(14).pct_change().std() * 100)  # As percentage
                
                # Calculate average daily range if high and low exist
                if 'high' in df.columns and 'low' in df.columns:
                    daily_ranges = (df['high'] - df['low']) / df['low'] * 100  # As percentage
                    metadata["avg_range_pct"] = float(daily_ranges.mean())
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return metadata
            
    @staticmethod
    def get_time_series_key(data: Dict[str, Any], asset_type: str, interval: str) -> Optional[str]:
        """
        Determine the correct time series key based on asset type and interval.
        Handles all Alpha Vantage response formats for both stocks and cryptocurrencies.
        
        Args:
            data: API response data
            asset_type: "stock" or "crypto"
            interval: Timeframe interval (e.g., "daily", "weekly", "5min")
            
        Returns:
            Time series key string or None if not found
        """
        # First, check if we can find an exact match in the data
        for key in data.keys():
            if "Time Series" in key or "Weekly Time Series" in key or "Monthly Time Series" in key:
                return key
        
        # Normalize interval for consistent handling
        interval_lower = interval.lower().replace(" ", "")
        
        # Define mappings for all known API response formats
        is_crypto = asset_type.lower() == "crypto"
        
        # Extended timeframes (daily, weekly, monthly)
        if interval_lower in ["daily", "day", "1d"]:
            if is_crypto:
                return "Time Series (Digital Currency Daily)"
            else:
                return "Time Series (Daily)"
                
        elif interval_lower in ["weekly", "week", "1w"]:
            if is_crypto:
                return "Time Series (Digital Currency Weekly)"
            else:
                return "Weekly Time Series"
                
        elif interval_lower in ["monthly", "month", "1mo", "1m"]:
            if is_crypto:
                return "Time Series (Digital Currency Monthly)"
            else:
                return "Monthly Time Series"
        
        # Intraday timeframes
        elif any(x in interval_lower for x in ["min", "hour", "h", "m"]):
            # Extract just the numeric part and unit for intraday intervals
            # Handles formats like "5min", "1h", "15m", etc.
            import re
            match = re.search(r'(\d+)([mh]|min|hour)', interval_lower)
            
            if match:
                value, unit = match.groups()
                # Normalize to minutes format expected by API
                if unit in ["h", "hour"]:
                    value = int(value) * 60
                    formatted_interval = f"{value}min"
                else:
                    # Make sure it's in the format "5min" not "5m"
                    formatted_interval = f"{value}min" if unit == "m" else f"{value}{unit}"
                
                if is_crypto:
                    return f"Time Series Crypto ({formatted_interval})"
                else:
                    return f"Time Series ({formatted_interval})"
            else:
                # If we can't parse it, use as-is
                if is_crypto:
                    return f"Time Series Crypto ({interval})"
                else:
                    return f"Time Series ({interval})"
        
        # Fallback - use the format directly if we don't recognize the interval
        if is_crypto:
            return f"Time Series (Digital Currency {interval.capitalize()})"
        else:
            return f"Time Series ({interval})"