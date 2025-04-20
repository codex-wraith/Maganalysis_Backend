import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class DataFormatter:
    """
    A utility class for formatting market data and technical analysis results
    into human-readable formats for display purposes.
    """
    
    @staticmethod
    def sanitize_numeric_field(value: Any) -> Union[float, None]:
        """
        Convert various numeric inputs to float format, handling different formats.
        
        Args:
            value: The value to convert, which may be a string, int, float, or None
            
        Returns:
            Converted float value or None if conversion fails
        """
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove commas and other formatting characters
            cleaned = value.strip().replace(',', '').replace('$', '')
            
            # Handle percentage values
            if '%' in cleaned:
                try:
                    percent_value = cleaned.replace('%', '')
                    return float(percent_value) / 100
                except ValueError:
                    return None
            
            # Handle standard numeric values
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    @staticmethod
    def format_price(price: Optional[float], include_dollar_sign: bool = True, decimal_places: int = 2) -> str:
        """
        Format a price value with commas and the appropriate number of decimal places.
        
        Args:
            price: Price value to format
            include_dollar_sign: Whether to include a dollar sign
            decimal_places: Number of decimal places to include
            
        Returns:
            Formatted price string
        """
        if price is None:
            return "N/A"
        
        # Determine if price is very small and needs more decimals
        if price < 0.01 and price > 0:
            # For very small numbers, use scientific notation or more decimals
            if price < 0.0001:
                formatted = f"{price:.8f}".rstrip('0').rstrip('.') if price > 0 else "0.00"
            else:
                formatted = f"{price:.6f}".rstrip('0').rstrip('.') if price > 0 else "0.00"
        else:
            # Normal number formatting with commas
            formatted = f"{price:,.{decimal_places}f}"
        
        # Add dollar sign if requested
        if include_dollar_sign:
            return f"${formatted}"
        return formatted
    
    @staticmethod
    def format_volume(volume: Optional[float], abbreviate: bool = True) -> str:
        """
        Format volume with appropriate abbreviations (K, M, B).
        
        Args:
            volume: Volume value to format
            abbreviate: Whether to abbreviate large numbers
            
        Returns:
            Formatted volume string
        """
        if volume is None:
            return "N/A"
        
        try:
            volume = float(volume)
        except (ValueError, TypeError):
            return "N/A"
        
        if not abbreviate or volume < 1000:
            return f"{volume:,.0f}"
        
        # Abbreviate large numbers
        if volume >= 1_000_000_000:
            return f"{volume/1_000_000_000:.2f}B"
        elif volume >= 1_000_000:
            return f"{volume/1_000_000:.2f}M"
        elif volume >= 1_000:
            return f"{volume/1_000:.2f}K"
        else:
            return f"{volume:,.0f}"
    
    @staticmethod
    def format_percent(value: Optional[float], include_sign: bool = True, decimal_places: int = 2) -> str:
        """
        Format a decimal value as a percentage with sign.
        
        Args:
            value: Numeric value to format as percentage
            include_sign: Whether to include + for positive values
            decimal_places: Number of decimal places to include
            
        Returns:
            Formatted percentage string
        """
        if value is None:
            return "N/A"
        
        # Format with specified decimal places
        formatted = f"{value:.{decimal_places}f}%"
        
        # Add sign if requested
        if include_sign and value > 0:
            return f"+{formatted}"
        return formatted
    
    @staticmethod
    def format_change_arrow(change: Optional[float]) -> str:
        """
        Return arrow symbol based on a numeric change.
        
        Args:
            change: Numeric change value
            
        Returns:
            Arrow symbol (↑, ↓, or →)
        """
        if change is None:
            return "→"
        
        if change > 0:
            return "↑"
        elif change < 0:
            return "↓"
        else:
            return "→"
    
    @classmethod
    def format_ohlcv_data(cls, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """
        Format OHLCV (Open, High, Low, Close, Volume) data into a readable string.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: The ticker symbol
            interval: Time interval (e.g., "daily", "60min")
            
        Returns:
            Formatted OHLCV data as a string
        """
        if df is None or df.empty:
            return f"No data available for {symbol} ({interval})"
        
        # Get the most recent data point
        latest = df.iloc[-1]
        
        # Calculate change
        if len(df) > 1:
            prev_close = float(df.iloc[-2]['close'])
            change = float(latest['close']) - prev_close
            change_pct = (change / prev_close) * 100
        else:
            change = 0
            change_pct = 0
        
        # Format the data
        date_str = df.index[-1].strftime("%Y-%m-%d %H:%M" if 'min' in interval else "%Y-%m-%d")
        
        result = [
            f"{symbol} ({interval}) - {date_str}",
            f"Open: {cls.format_price(latest['open'])}",
            f"High: {cls.format_price(latest['high'])}",
            f"Low: {cls.format_price(latest['low'])}",
            f"Close: {cls.format_price(latest['close'])} {cls.format_change_arrow(change)} ({cls.format_percent(change_pct)})",
            f"Volume: {cls.format_volume(latest['volume'])}"
        ]
        
        return "\n".join(result)
    
    @classmethod
    def format_intraday_data(cls, data: Dict[str, Any], symbol: str, interval: str) -> str:
        """
        Format intraday data from API response.
        
        Args:
            data: API response data
            symbol: The ticker symbol
            interval: Time interval (e.g., "5min", "60min")
            
        Returns:
            Formatted intraday data as a string
        """
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key or not data.get(time_series_key):
            return f"No intraday data available for {symbol} ({interval})"
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        
        # Rename columns to standardized names
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
        
        # Convert all price and volume columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return cls.format_ohlcv_data(df, symbol, interval)
    
    @classmethod
    def format_crypto_intraday_data(cls, data: Dict[str, Any], symbol: str, market: str, interval: str) -> str:
        """
        Format crypto intraday data from API response.
        
        Args:
            data: API response data
            symbol: The crypto symbol
            market: The market (e.g., "USD")
            interval: Time interval (e.g., "5min", "60min")
            
        Returns:
            Formatted crypto intraday data as a string
        """
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key or not data.get(time_series_key):
            return f"No intraday data available for {symbol}/{market} ({interval})"
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        
        # Rename columns to standardized names for crypto
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
        
        # Convert all price and volume columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return cls.format_ohlcv_data(df, f"{symbol}/{market}", interval)
    
    @classmethod
    def format_technical_indicators(cls, indicators: Dict[str, Any], symbol: str, interval: str) -> str:
        """
        Format technical indicators into a readable string.
        
        Args:
            indicators: Dictionary of indicator results
            symbol: The ticker symbol
            interval: Time interval (e.g., "daily", "60min")
            
        Returns:
            Formatted technical indicators as a string
        """
        result = [f"{symbol} Technical Indicators ({interval})"]
        
        # Format moving averages
        if 'sma' in indicators:
            result.append(f"SMA: {cls.format_price(indicators['sma'])}")
        if 'ema' in indicators:
            result.append(f"EMA: {cls.format_price(indicators['ema'])}")
        
        # Format oscillators
        if 'rsi' in indicators:
            rsi_value = indicators['rsi']
            rsi_interpretation = ""
            if rsi_value is not None:
                if rsi_value > 70:
                    rsi_interpretation = " (Overbought)"
                elif rsi_value < 30:
                    rsi_interpretation = " (Oversold)"
            if rsi_value is not None:
                result.append(f"RSI: {rsi_value:.2f}{rsi_interpretation}")
            else:
                result.append("RSI: N/A")
        
        # Format MACD
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd']
            signal = indicators['macd_signal']
            if macd is not None and signal is not None:
                macd_diff = macd - signal
                bullish = macd_diff > 0
                crossover = abs(macd_diff) < 0.1 * abs(macd)
                macd_interpretation = " (Bullish)" if bullish else " (Bearish)"
                if crossover:
                    macd_interpretation = " (Potential Crossover)"
                result.append(f"MACD: {macd:.3f}, Signal: {signal:.3f}{macd_interpretation}")
            else:
                result.append("MACD: N/A")
        
        # Format Stochastic Oscillator
        if 'stoch_k' in indicators and 'stoch_d' in indicators:
            k = indicators['stoch_k']
            d = indicators['stoch_d']
            if k is not None and d is not None:
                stoch_interpretation = ""
                if k > 80 and d > 80:
                    stoch_interpretation = " (Overbought)"
                elif k < 20 and d < 20:
                    stoch_interpretation = " (Oversold)"
                elif k > d and k - d <= 3:
                    stoch_interpretation = " (Potential Bullish Crossover)"
                elif k < d and d - k <= 3:
                    stoch_interpretation = " (Potential Bearish Crossover)"
                result.append(f"Stochastic %K: {k:.2f}, %D: {d:.2f}{stoch_interpretation}")
            else:
                result.append("Stochastic: N/A")
        
        # Format Bollinger Bands
        if 'bbands' in indicators:
            bbands = indicators['bbands']
            if isinstance(bbands, dict) and all(k in bbands for k in ['upper', 'middle', 'lower']):
                upper = bbands['upper']
                middle = bbands['middle']
                lower = bbands['lower']
                if all(v is not None for v in [upper, middle, lower]):
                    result.append(f"Bollinger Bands:")
                    result.append(f"  Upper: {cls.format_price(upper)}")
                    result.append(f"  Middle: {cls.format_price(middle)}")
                    result.append(f"  Lower: {cls.format_price(lower)}")
                else:
                    result.append("Bollinger Bands: N/A")
            else:
                result.append("Bollinger Bands: N/A")
        
        # Format ATR
        if 'atr' in indicators:
            atr = indicators['atr']
            if atr is not None:
                result.append(f"ATR: {cls.format_price(atr, False)}")
            else:
                result.append("ATR: N/A")
        
        # Format ADX
        if 'adx' in indicators:
            adx = indicators['adx']
            if adx is not None:
                adx_interpretation = ""
                if adx > 25:
                    adx_interpretation = " (Strong Trend)"
                elif adx < 20:
                    adx_interpretation = " (Weak Trend)"
                result.append(f"ADX: {adx:.2f}{adx_interpretation}")
            else:
                result.append("ADX: N/A")
        
        # Format VWAP
        if 'vwap' in indicators:
            vwap = indicators['vwap']
            if vwap is not None:
                result.append(f"VWAP: {cls.format_price(vwap)}")
            else:
                result.append("VWAP: N/A")
        
        return "\n".join(result)
    
    @classmethod
    def format_support_resistance_levels(cls, levels: Dict[str, List], symbol: str) -> str:
        """
        Format support and resistance levels into a readable string.
        
        Args:
            levels: Dictionary with support and resistance levels
            symbol: The ticker symbol
            
        Returns:
            Formatted support and resistance levels as a string
        """
        result = [f"{symbol} Support & Resistance Levels"]
        
        # Format resistance levels
        if 'resistance' in levels and levels['resistance']:
            result.append("\nResistance Levels:")
            for i, level in enumerate(levels['resistance'], 1):
                if isinstance(level, dict) and 'price' in level and 'confidence' in level:
                    # Multi-timeframe format
                    timeframes = level.get('timeframes', [])
                    timeframe_str = ", ".join(timeframes) if timeframes else "unknown"
                    result.append(f"R{i}: {cls.format_price(level['price'])} (Confidence: {level['confidence']}, Timeframes: {timeframe_str})")
                else:
                    # Simple format
                    result.append(f"R{i}: {cls.format_price(level)}")
        else:
            result.append("\nNo resistance levels identified")
        
        # Format support levels
        if 'support' in levels and levels['support']:
            result.append("\nSupport Levels:")
            for i, level in enumerate(levels['support'], 1):
                if isinstance(level, dict) and 'price' in level and 'confidence' in level:
                    # Multi-timeframe format
                    timeframes = level.get('timeframes', [])
                    timeframe_str = ", ".join(timeframes) if timeframes else "unknown"
                    result.append(f"S{i}: {cls.format_price(level['price'])} (Confidence: {level['confidence']}, Timeframes: {timeframe_str})")
                else:
                    # Simple format
                    result.append(f"S{i}: {cls.format_price(level)}")
        else:
            result.append("\nNo support levels identified")
        
        return "\n".join(result)