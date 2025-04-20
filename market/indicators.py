import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from functools import wraps

logger = logging.getLogger(__name__)

def validate_dataframe(func):
    """Decorator to validate DataFrame inputs for indicator calculations"""
    @wraps(func)
    def wrapper(df, *args, **kwargs):
        if df is None or df.empty:
            logger.warning(f"Empty or None DataFrame passed to {func.__name__}")
            return None
            
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"DataFrame missing required columns {missing_columns} for {func.__name__}")
            return None
            
        result = func(df, *args, **kwargs)
        return result
    return wrapper

class TechnicalIndicators:
    """
    A utility class providing methods to calculate various technical indicators.
    All methods handle their own validation and error handling.
    """
    
    @staticmethod
    @validate_dataframe
    def calculate_sma(df: pd.DataFrame, period: int = 20, column: str = 'close') -> Optional[float]:
        """
        Calculate Simple Moving Average
        
        Args:
            df: DataFrame with price data
            period: SMA period
            column: Column to use for calculation
            
        Returns:
            The SMA value or None on error
        """
        try:
            if len(df) < period:
                logger.warning(f"Not enough data for {period}-period SMA calculation")
                return None
                
            # Calculate SMA
            sma = df[column].rolling(window=period).mean().iloc[-1]
            return float(sma)
            
        except Exception as e:
            logger.warning(f"Error calculating SMA: {e}")
            return None
            
    @staticmethod
    @validate_dataframe
    def calculate_ema(df: pd.DataFrame, period: int = 20, column: str = 'close') -> Optional[float]:
        """
        Calculate Exponential Moving Average
        
        Args:
            df: DataFrame with price data
            period: EMA period
            column: Column to use for calculation
            
        Returns:
            The EMA value or None on error
        """
        try:
            if len(df) < period:
                logger.warning(f"Not enough data for {period}-period EMA calculation")
                return None
                
            # Calculate EMA
            ema = df[column].ewm(span=period, adjust=False).mean().iloc[-1]
            return float(ema)
            
        except Exception as e:
            logger.warning(f"Error calculating EMA: {e}")
            return None
            
    @staticmethod
    @validate_dataframe
    def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> Optional[float]:
        """
        Calculate Relative Strength Index using Wilder's smoothing method
        
        Args:
            df: DataFrame with price data
            period: RSI period
            column: Column to use for calculation
            
        Returns:
            The RSI value or None on error
        """
        try:
            if len(df) < period + 1:
                logger.warning(f"Not enough data for {period}-period RSI calculation")
                return None
                
            # Calculate price changes
            delta = df[column].diff()
            
            # Create gain and loss series
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            # First average using simple mean for the initial period
            first_avg_gain = gain.iloc[:period].mean()
            first_avg_loss = loss.iloc[:period].mean()
            
            # Use Wilder's smoothing for subsequent periods (EWM with alpha=1/period)
            # Initialize lists to hold avg_gain and avg_loss values
            avg_gains = [first_avg_gain]
            avg_losses = [first_avg_loss]
            
            # Calculate subsequent values using Wilder's smoothing formula
            for i in range(period, len(gain)):
                avg_gain = ((period - 1) * avg_gains[-1] + gain.iloc[i]) / period
                avg_loss = ((period - 1) * avg_losses[-1] + loss.iloc[i]) / period
                avg_gains.append(avg_gain)
                avg_losses.append(avg_loss)
            
            # Calculate RS and RSI for the latest period
            if avg_losses[-1] == 0:
                rsi = 100  # Prevent division by zero
            else:
                rs = avg_gains[-1] / avg_losses[-1]
                rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return None
            
    @staticmethod
    @validate_dataframe
    def calculate_macd(
        df: pd.DataFrame, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9,
        column: str = 'close'
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate MACD and Signal Line with special handling for small values
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            column: Column to use for calculation
            
        Returns:
            Tuple of (MACD value, Signal value) or (None, None) on error
        """
        try:
            if len(df) < max(fast_period, slow_period, signal_period):
                logger.warning(f"Not enough data for MACD calculation")
                return None, None
                
            # Calculate fast and slow EMAs
            fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
            slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Get final values
            final_macd = float(macd_line.iloc[-1])
            final_signal = float(signal_line.iloc[-1])
            
            # Special handling for small MACD values (relative to price)
            current_price = float(df[column].iloc[-1])
            
            # If MACD values are very small relative to price, scale them for better usability
            if abs(final_macd) < (current_price * 0.0001) and current_price > 100:
                # For higher priced assets, very small MACD values can be hard to interpret
                # Scale both MACD and signal by the same factor to maintain their relationship
                logger.info(f"Scaling small MACD value relative to price (${current_price:.2f})")
                
                # Scale factor: make MACD at least 0.01% of price
                scale_factor = (current_price * 0.0001) / max(abs(final_macd), 0.00001)
                
                # Apply scaling
                final_macd = final_macd * scale_factor
                final_signal = final_signal * scale_factor
            
            return final_macd, final_signal
            
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            return None, None
            
    @staticmethod
    @validate_dataframe
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range
        
        Args:
            df: DataFrame with high, low, close data
            period: ATR period
            
        Returns:
            The ATR value or None on error
        """
        try:
            if len(df) < period:
                logger.warning(f"Not enough data for {period}-period ATR calculation")
                return None
                
            # Calculate the True Range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            tr = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close})
            tr['TR'] = tr.max(axis=1)
            
            # Calculate the ATR
            atr = tr['TR'].rolling(window=period).mean().iloc[-1]
            
            return float(atr)
            
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")
            return None
            
    @staticmethod
    @validate_dataframe
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Calculate Average Directional Index using Wilder's smoothing
        
        Args:
            df: DataFrame with high, low, close data
            period: ADX period
            
        Returns:
            The ADX value or None on error
        """
        try:
            # For small timeframes with many data points, limit the dataset size
            # to avoid timeout issues (particularly important for 5min timeframe)
            max_rows = 500  # Enough for accurate ADX calculation
            if len(df) > max_rows:
                logger.info(f"Using last {max_rows} data points for ADX calculation (from {len(df)} total)")
                df = df.tail(max_rows)
                
            if len(df) < period * 3:
                logger.warning(f"Not enough data for {period}-period ADX calculation")
                return None
                
            # Create copies to avoid modifying original data
            data = df.copy()
            
            # Calculate True Range
            data['high_low'] = data['high'] - data['low']
            data['high_close'] = abs(data['high'] - data['close'].shift(1))
            data['low_close'] = abs(data['low'] - data['close'].shift(1))
            data['tr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            # Calculate Directional Movement
            data['up_move'] = data['high'] - data['high'].shift(1)
            data['down_move'] = data['low'].shift(1) - data['low']
            
            # Calculate Plus and Minus Directional Movement
            data['+dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
            data['-dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)
            
            # Smooth using Wilder's method
            # First calculate simple average for initial period
            for col in ['tr', '+dm', '-dm']:
                data[f'{col}{period}'] = data[col].rolling(window=period).sum()
            
            # Use Wilder's smoothing for subsequent values
            for i in range(period + 1, len(data)):
                for col in ['tr', '+dm', '-dm']:
                    prev_value = data.loc[data.index[i-1], f'{col}{period}']
                    current_value = data.loc[data.index[i], col]
                    data.loc[data.index[i], f'{col}{period}'] = prev_value - (prev_value / period) + current_value
            
            # Calculate Directional Indicators
            data['+di'] = 100 * (data[f'+dm{period}'] / data[f'tr{period}'])
            data['-di'] = 100 * (data[f'-dm{period}'] / data[f'tr{period}'])
            
            # Handle division by zero cases
            data['+di'] = data['+di'].replace([np.inf, -np.inf], np.nan).fillna(0)
            data['-di'] = data['-di'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate DX - preventing division by zero
            denominator = data['+di'] + data['-di']
            data['dx'] = 100 * (abs(data['+di'] - data['-di']) / denominator.where(denominator != 0, 1))
            data['dx'] = data['dx'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate ADX using Wilder's smoothing
            # First ADX is simple average of DX for period, but check for array bounds
            data['adx'] = np.nan
            
            # Ensure index access is within valid range
            if len(data) >= 2*period:
                index = min(2*period-1, len(data.index)-1) 
                data.loc[data.index[index], 'adx'] = data['dx'].iloc[period:2*period].mean()
            
            # Subsequent ADX values use smoothing
            for i in range(2*period, len(data)):
                prev_adx = data.loc[data.index[i-1], 'adx']
                current_dx = data.loc[data.index[i], 'dx']
                data.loc[data.index[i], 'adx'] = ((period - 1) * prev_adx + current_dx) / period
            
            # Return the final ADX value
            adx_value = data['adx'].iloc[-1]
            
            return float(adx_value)
            
        except Exception as e:
            logger.warning(f"Error calculating ADX: {e}")
            return None
            
    @staticmethod
    @validate_dataframe
    def calculate_stochastic(
        df: pd.DataFrame, 
        k_period: int = 14, 
        d_period: int = 3
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            df: DataFrame with high, low, close data
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K value, %D value) or (None, None) on error
        """
        try:
            if len(df) < k_period + d_period:
                logger.warning(f"Not enough data for Stochastic calculation")
                return None, None
                
            # Calculate %K
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            
            k = 100 * ((df['close'] - low_min) / (high_max - low_min))
            
            # Calculate %D
            d = k.rolling(window=d_period).mean()
            
            return float(k.iloc[-1]), float(d.iloc[-1])
            
        except Exception as e:
            logger.warning(f"Error calculating Stochastic Oscillator: {e}")
            return None, None
            
    @staticmethod
    @validate_dataframe
    def calculate_bbands(
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0,
        column: str = 'close'
    ) -> Dict[str, Optional[float]]:
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with price data
            period: Bollinger Bands period
            std_dev: Number of standard deviations
            column: Column to use for calculation
            
        Returns:
            Dictionary with upper, middle, and lower bands or None values on error
        """
        try:
            if len(df) < period:
                logger.warning(f"Not enough data for {period}-period Bollinger Bands calculation")
                return {"upper": None, "middle": None, "lower": None}
                
            # Calculate middle band (SMA)
            middle_band = df[column].rolling(window=period).mean()
            
            # Calculate standard deviation
            std = df[column].rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            return {
                "upper": float(upper_band.iloc[-1]),
                "middle": float(middle_band.iloc[-1]),
                "lower": float(lower_band.iloc[-1])
            }
            
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
            return {"upper": None, "middle": None, "lower": None}
            
    @staticmethod
    @validate_dataframe
    def calculate_vwap(df: pd.DataFrame) -> Optional[float]:
        """
        Calculate Volume Weighted Average Price
        
        Args:
            df: DataFrame with high, low, close, and volume data
            
        Returns:
            The VWAP value or None on error
        """
        try:
            if 'volume' not in df.columns:
                logger.warning("Volume data required for VWAP calculation")
                return None
                
            # Calculate typical price
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Direct calculation matching original implementation
            # Sum of price * volume divided by sum of volume
            sum_pv = (df['typical_price'] * df['volume']).sum()
            sum_volume = df['volume'].sum()
            
            if sum_volume == 0:
                logger.warning("Zero volume sum encountered in VWAP calculation")
                return None
                
            vwap = sum_pv / sum_volume
            
            return float(vwap)
            
        except Exception as e:
            logger.warning(f"Error calculating VWAP: {e}")
            return None
            
            
class TimeframeParameters:
    """
    Utility class for handling timeframe-specific indicator parameters.
    Provides optimized parameter values for different chart timeframes.
    """
    
    @staticmethod
    def get_parameters(interval: str) -> Dict[str, int]:
        """
        Return standardized indicator parameters based on the timeframe.
        This centralized method avoids duplicated parameter settings across analysis functions.
        
        Args:
            interval (str): Time interval for analysis (e.g., "daily", "weekly", "5min")
            
        Returns:
            Dict: Dictionary containing all parameter values for technical indicators
        """
        # Convert interval to lowercase and normalize format
        interval_lower = interval.lower().replace(" ", "")
        
        # Weekly timeframe parameters
        if interval_lower in ["weekly", "week", "1w"]:
            return {
                "sma_period": 20,
                "ema_period": 8,
                "rsi_period": 7,
                "bbands_period": 12,
                "adx_period": 10,
                "atr_period": 10,
                "stoch_k_period": 9,
                "stoch_d_period": 3
            }
        # Monthly timeframe parameters
        elif interval_lower in ["monthly", "month", "1mo"]:
            return {
                "sma_period": 12,
                "ema_period": 6,
                "rsi_period": 6,
                "bbands_period": 6,
                "adx_period": 7,
                "atr_period": 7,
                "stoch_k_period": 5,
                "stoch_d_period": 3
            }
        # Very short intraday timeframes (1min, 5min)
        elif interval_lower in ["1min", "5min", "1m", "5m"]:
            return {
                "sma_period": 20,
                "ema_period": 9,
                "rsi_period": 7,
                "bbands_period": 20,
                "adx_period": 10,
                "atr_period": 8,
                "stoch_k_period": 8,
                "stoch_d_period": 3
            }
        # Medium intraday timeframes (15min, 30min)
        elif interval_lower in ["15min", "30min", "15m", "30m"]:
            return {
                "sma_period": 30,
                "ema_period": 12,
                "rsi_period": 10,
                "bbands_period": 20,
                "adx_period": 12,
                "atr_period": 10,
                "stoch_k_period": 10,
                "stoch_d_period": 3
            }
        # Hourly timeframe
        elif interval_lower in ["60min", "1h", "1hour", "hourly", "60m"]:
            return {
                "sma_period": 40,
                "ema_period": 15,
                "rsi_period": 12,
                "bbands_period": 20,
                "adx_period": 14,
                "atr_period": 12,
                "stoch_k_period": 12,
                "stoch_d_period": 3
            }
        # Daily timeframe
        elif interval_lower in ["daily", "1d", "day"]:
            return {
                "sma_period": 50,
                "ema_period": 21,
                "rsi_period": 14,
                "bbands_period": 20,
                "adx_period": 14,
                "atr_period": 10,
                "stoch_k_period": 9,
                "stoch_d_period": 3
            }
        # Default parameters (daily timeframe)
        else:
            return {
                "sma_period": 50,
                "ema_period": 21,
                "rsi_period": 14,
                "bbands_period": 20,
                "adx_period": 14,
                "atr_period": 10,
                "stoch_k_period": 9,
                "stoch_d_period": 3
            }