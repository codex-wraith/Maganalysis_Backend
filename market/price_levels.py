import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from enum import Enum

logger = logging.getLogger(__name__)

class LevelType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"

class PriceLevelAnalyzer:
    """
    A utility class for identifying support and resistance levels in price data.
    """
    
    @staticmethod
    def _validate_price_data(price_data: pd.DataFrame) -> bool:
        """Validate that price data contains required columns"""
        if price_data is None or price_data.empty:
            logger.warning("Empty or None price data provided")
            return False
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in price_data.columns]
        
        if missing_columns:
            logger.warning(f"Price data missing required columns: {missing_columns}")
            return False
            
        return True
        
    @classmethod
    def _find_swing_points(
        cls, 
        price_data: pd.DataFrame, 
        level_type: LevelType, 
        lookback: int = 3,
        interval: str = "daily"
    ) -> List[Tuple[float, pd.Timestamp, float]]:
        """
        Find swing points (highs or lows) in the price data
        
        Args:
            price_data: DataFrame with OHLCV data
            level_type: LevelType.SUPPORT for swing lows or LevelType.RESISTANCE for swing highs
            lookback: Number of bars to look back/forward for confirmation
            interval: Timeframe interval string (e.g., "5min", "daily")
            
        Returns:
            List of (price, timestamp, volume) tuples for the swing points
        """
        if not cls._validate_price_data(price_data):
            return []
            
        # Adjust lookback based on timeframe
        interval_lower = interval.lower().replace(" ", "")
        
        # Timeframe groups
        very_short_intervals = ["1min", "1m", "1minute"]
        short_intervals = ["5min", "5m", "15min", "15m", "5minute", "15minute"]
        medium_intervals = ["30min", "30m", "60min", "60m", "1hour", "1h", "30minute", "60minute"]
        daily_intervals = ["1day", "1d", "daily", "day"]
        weekly_intervals = ["1week", "1w", "weekly", "week"]
        monthly_intervals = ["1month", "1mo", "monthly", "month"]
        
        # Adaptive settings based on available data
        data_length = len(price_data)
        data_is_limited = data_length < 50  # Consider data limited if fewer than 50 bars
        
        # Adjust lookback parameter based on timeframe and available data
        if any(x in interval_lower for x in very_short_intervals):
            lookback = 1 if data_is_limited else max(lookback, 2)
            max_bars = min(100, data_length)
        elif any(x in interval_lower for x in short_intervals):
            lookback = 1 if data_is_limited else max(lookback, 2)
            max_bars = min(4320, data_length)
        elif any(x in interval_lower for x in medium_intervals):
            lookback = 2 if data_is_limited else max(lookback, 3)
            max_bars = min(720, data_length)
        elif any(x in interval_lower for x in daily_intervals):
            lookback = 2 if data_is_limited else max(lookback, 3)
            max_bars = min(365, data_length)
        elif any(x in interval_lower for x in weekly_intervals):
            lookback = 1 if data_is_limited else max(lookback, 2)
            max_bars = min(75, data_length)
        elif any(x in interval_lower for x in monthly_intervals):
            lookback = 1 if data_is_limited else max(lookback, 2)
            max_bars = min(50, data_length)
        else:
            # Default with adaptive lookback
            lookback = 2 if data_is_limited else lookback
            max_bars = min(150, data_length)
            
        # Ensure lookback is at least 1 and won't exceed available data
        lookback = max(1, min(lookback, data_length // 4))
        
        logger.info(f"Finding swing points on {interval} timeframe with lookback={lookback}")
        
        swing_points = []
        
        # Determine column and comparison function based on level type
        if level_type == LevelType.SUPPORT:
            column = 'low'
            is_extreme = lambda i: all(price_data.iloc[i][column] <= price_data.iloc[j][column] 
                                    for j in range(max(0, i-lookback), i)) and \
                                  all(price_data.iloc[i][column] <= price_data.iloc[j][column] 
                                    for j in range(i+1, min(len(price_data), i+lookback+1)))
        else:  # RESISTANCE
            column = 'high'
            is_extreme = lambda i: all(price_data.iloc[i][column] >= price_data.iloc[j][column] 
                                    for j in range(max(0, i-lookback), i)) and \
                                  all(price_data.iloc[i][column] >= price_data.iloc[j][column] 
                                    for j in range(i+1, min(len(price_data), i+lookback+1)))
        
        # Find swing points
        for i in range(lookback, max_bars - lookback):
            if is_extreme(i):
                price = price_data.iloc[i][column]
                timestamp = price_data.index[i]
                volume = price_data.iloc[i]['volume']
                
                # Add this swing point
                swing_points.append((price, timestamp, volume))
                
        # Find key window high/low points (method 2 from old implementation)
        # This catches additional significant levels that might not be perfect swing points
        if len(swing_points) < 3:  # Only do this if we found few swing points
            # Adapt window size to available data
            window_size = 2 if data_is_limited else 5
            # Ensure window size doesn't exceed data length / 4
            window_size = min(window_size, max(1, data_length // 6))
            for i in range(window_size, max_bars - window_size):
                window_slice = price_data.iloc[i-window_size:i+window_size+1]
                
                if level_type == LevelType.SUPPORT:
                    # Check if this is a local minimum within the window
                    if price_data.iloc[i]['low'] == window_slice['low'].min():
                        # Check if not already included
                        price = price_data.iloc[i]['low']
                        already_captured = any(abs(p - price) / price < 0.01 for p, _, _ in swing_points)
                        
                        if not already_captured:
                            timestamp = price_data.index[i]
                            volume = price_data.iloc[i]['volume']
                            swing_points.append((price, timestamp, volume))
                else:  # RESISTANCE
                    # Check if this is a local maximum within the window
                    if price_data.iloc[i]['high'] == window_slice['high'].max():
                        # Check if not already included
                        price = price_data.iloc[i]['high']
                        already_captured = any(abs(p - price) / price < 0.01 for p, _, _ in swing_points)
                        
                        if not already_captured:
                            timestamp = price_data.index[i]
                            volume = price_data.iloc[i]['volume']
                            swing_points.append((price, timestamp, volume))
        
        return swing_points
        
    @classmethod
    def _calculate_bounce_strength(
        cls, 
        price_data: pd.DataFrame, 
        level_price: float, 
        timestamp: pd.Timestamp, 
        level_type: LevelType, 
        atr: Optional[float] = None
    ) -> float:
        """
        Calculate the bounce strength for a specific price level
        
        Args:
            price_data: DataFrame with OHLCV data
            level_price: The price level to analyze
            timestamp: The timestamp of the swing point
            level_type: Whether this is a support or resistance level
            atr: Optional ATR value for normalizing strength
            
        Returns:
            Bounce strength value (higher is stronger)
        """
        if not cls._validate_price_data(price_data):
            return 0.0
            
        # Get the position of the timestamp in the DataFrame
        try:
            idx = price_data.index.get_loc(timestamp)
        except KeyError:
            logger.warning(f"Timestamp {timestamp} not found in price data")
            return 0.0
            
        # Look at subsequent price action to measure bounce strength
        strength = 0.0
        bars_to_check = min(5, len(price_data) - idx - 1)
        
        if level_type == LevelType.SUPPORT:
            # For support: measure how far price bounced up from the level
            bounce_high = price_data.iloc[idx:idx+bars_to_check+1]['high'].max()
            bounce_pct = (bounce_high - level_price) / level_price * 100
            
            # Normalize by ATR if available
            if atr:
                bounce_pct = bounce_pct / (atr / level_price * 100)
                
            strength = min(bounce_pct * 2, 10)  # Cap at 10
            
        else:  # RESISTANCE
            # For resistance: measure how far price bounced down from the level
            bounce_low = price_data.iloc[idx:idx+bars_to_check+1]['low'].min()
            bounce_pct = (level_price - bounce_low) / level_price * 100
            
            # Normalize by ATR if available
            if atr:
                bounce_pct = bounce_pct / (atr / level_price * 100)
                
            strength = min(bounce_pct * 2, 10)  # Cap at 10
            
        return strength
        
    @classmethod
    def _adjust_importance_for_volume(cls, importance: float, volume: float, avg_volume: float) -> float:
        """
        Adjust level importance based on relative volume
        
        Args:
            importance: Base level importance
            volume: Volume at the swing point
            avg_volume: Average volume in the dataset
            
        Returns:
            Adjusted importance value
        """
        if avg_volume <= 0:
            return importance
            
        volume_ratio = volume / avg_volume
        
        # Volume > 2x average: boost importance by up to 50%
        if volume_ratio > 2:
            return importance * min(1.5, 1 + (volume_ratio - 2) * 0.25)
            
        # Volume < 0.5x average: reduce importance
        if volume_ratio < 0.5:
            return importance * max(0.7, 0.5 + volume_ratio * 0.4)
            
        # Otherwise adjust proportionally
        return importance * (0.8 + volume_ratio * 0.2)
        
    @classmethod
    def _group_nearby_levels(
        cls, 
        levels: List[Tuple[float, float]], 
        price_data: pd.DataFrame, 
        level_type: LevelType,
        atr: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """
        Group nearby price levels into zones
        
        Args:
            levels: List of (price, importance) tuples
            price_data: DataFrame with OHLCV data
            level_type: Whether these are support or resistance levels
            atr: ATR value for adaptive zone width
            
        Returns:
            List of (zone_price, zone_importance) tuples
        """
        if not levels:
            return []
            
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x[0])
        
        # Determine zone grouping distance as a percentage of price
        # If ATR is available, use it for more adaptive grouping
        if atr is not None:
            # Use ATR to determine appropriate grouping distance
            reference_price = sorted_levels[0][0]  # Use first level's price as reference
            grouping_pct = (atr / reference_price) * 1.5  # 1.5x ATR as percentage of price
        else:
            # Default grouping distance (0.5% of price)
            grouping_pct = 0.005
            
        # Initialize zones with the first level
        zones = [(sorted_levels[0][0], sorted_levels[0][1])]
        
        # Group subsequent levels
        for price, importance in sorted_levels[1:]:
            # Find closest zone
            closest_idx = 0
            closest_dist = float('inf')
            
            for i, (zone_price, _) in enumerate(zones):
                dist = abs(price - zone_price) / price
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
                    
            # If close enough to an existing zone, merge
            if closest_dist < grouping_pct:
                zone_price, zone_importance = zones[closest_idx]
                
                # Calculate weighted average price based on importance
                total_importance = zone_importance + importance
                new_price = (zone_price * zone_importance + price * importance) / total_importance
                
                # Replace zone with updated values
                zones[closest_idx] = (new_price, total_importance)
            else:
                # Add as a new zone
                zones.append((price, importance))
                
        return zones
        
    @classmethod
    def _add_fibonacci_levels(
        cls,
        price_data: pd.DataFrame,
        current_price: float,
        level_type: LevelType
    ) -> List[Tuple[float, float]]:
        """
        Add Fibonacci retracement levels based on price range
        
        Args:
            price_data: DataFrame with OHLCV data
            current_price: Current market price
            level_type: Whether to find support or resistance levels
            
        Returns:
            List of (price, importance) tuples for Fibonacci levels
        """
        if price_data.empty:
            return []
            
        # Calculate the price range
        highest_high = price_data['high'].max()
        lowest_low = price_data['low'].min()
        
        if highest_high <= lowest_low:  # Sanity check
            return []
            
        price_range = highest_high - lowest_low
        fib_levels = []
        
        # Standard Fibonacci retracement levels
        if level_type == LevelType.SUPPORT:
            # For support, calculate retracement levels from high to low
            fib_levels = [
                (highest_high - (price_range * 0.236), 0.8),  # 23.6% retracement
                (highest_high - (price_range * 0.382), 1.0),  # 38.2% retracement 
                (highest_high - (price_range * 0.5), 1.2),    # 50.0% retracement
                (highest_high - (price_range * 0.618), 1.5),  # 61.8% retracement
                (highest_high - (price_range * 0.786), 1.2),  # 78.6% retracement
                (lowest_low, 1.0)                             # 100% retracement
            ]
            # Keep only levels below current price
            fib_levels = [(price, importance) for price, importance in fib_levels 
                           if price < current_price]
        else:  # RESISTANCE
            # For resistance, calculate extensions from low to high
            fib_levels = [
                (lowest_low + (price_range * 0.236), 0.8),  # 23.6% retracement
                (lowest_low + (price_range * 0.382), 1.0),  # 38.2% retracement
                (lowest_low + (price_range * 0.5), 1.2),    # 50.0% retracement
                (lowest_low + (price_range * 0.618), 1.5),  # 61.8% retracement
                (lowest_low + (price_range * 0.786), 1.2),  # 78.6% retracement
                (highest_high, 1.0)                         # 100% retracement
            ]
            # Keep only levels above current price
            fib_levels = [(price, importance) for price, importance in fib_levels 
                           if price > current_price]
                
        return fib_levels
        
    @classmethod
    def _add_psychological_levels(
        cls,
        current_price: float, 
        price_data: pd.DataFrame, 
        level_type: LevelType,
        proximity_threshold: float
    ) -> List[Tuple[float, float]]:
        """
        Add psychological price levels, with enhanced handling for penny stocks
        
        Args:
            current_price: Current market price
            price_data: DataFrame with OHLCV data
            level_type: Whether to find support or resistance levels
            proximity_threshold: Distance threshold for grouping levels
            
        Returns:
            List of (price, importance) tuples for psychological levels
        """
        if price_data.empty:
            return []
            
        psychological_levels = []
        is_penny_stock = current_price < 1.0
        is_low_priced = 1.0 <= current_price < 5.0  # Additional category for stocks $1-$5
        
        # Determine price range to analyze based on level type
        if level_type == LevelType.SUPPORT:
            # For support, look at price range from min to current
            price_min = max(0.01, price_data['low'].min() * 0.9)
            price_max = current_price
        else:  # RESISTANCE
            # For resistance, look at price range from current to max
            price_min = current_price
            price_max = max(price_data['high'].max() * 1.1, current_price * 1.5)
        
        # Define levels based on price range and stock type
        if is_penny_stock:
            # Enhanced penny stock level detection with more granular levels
            # Penny stocks often have significant support/resistance at penny increments
            penny_increments = [
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 
                0.75, 0.80, 0.90, 1.00
            ]
            
            # Specific important levels that often have extra significance for penny stocks
            key_penny_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
            
            for level in penny_increments:
                if price_min <= level <= price_max:
                    # Check for historical tests of this level with volume confirmation
                    test_prices = []
                    test_volumes = []
                    
                    # Look for price tests with corresponding volumes
                    for i in range(len(price_data)):
                        if level_type == LevelType.SUPPORT:
                            # For support, check if low price tested this level
                            if abs(price_data.iloc[i]['low'] - level) <= proximity_threshold:
                                test_prices.append(price_data.iloc[i]['low'])
                                test_volumes.append(price_data.iloc[i]['volume'])
                        else:
                            # For resistance, check if high price tested this level
                            if abs(price_data.iloc[i]['high'] - level) <= proximity_threshold:
                                test_prices.append(price_data.iloc[i]['high'])
                                test_volumes.append(price_data.iloc[i]['volume'])
                    
                    # Only include levels with sufficient testing
                    test_count = len(test_prices)
                    if test_count >= 1:
                        # Base importance calculation
                        importance = 0.5 + (test_count * 0.25)
                        
                        # Volume-weighted importance adjustment
                        if test_count > 0 and test_volumes:
                            avg_volume = price_data['volume'].mean()
                            test_volume_ratio = sum(test_volumes) / (test_count * avg_volume)
                            # Boost importance for high volume tests
                            importance *= min(1.5, max(0.7, test_volume_ratio))
                        
                        # Extra importance for key price levels
                        if level in key_penny_levels:
                            importance *= 1.5
                            
                        psychological_levels.append((level, importance))
        
        elif is_low_priced:
            # Special handling for low-priced stocks ($1-$5)
            # These stocks often respect quarter and half-dollar levels
            low_price_increments = [
                1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 
                3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00
            ]
            
            # Key levels that often have extra significance
            key_low_price_levels = [1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00]
            
            for level in low_price_increments:
                if price_min <= level <= price_max:
                    # Check for historical tests with volume
                    test_prices = []
                    test_volumes = []
                    
                    for i in range(len(price_data)):
                        if level_type == LevelType.SUPPORT:
                            if abs(price_data.iloc[i]['low'] - level) <= proximity_threshold:
                                test_prices.append(price_data.iloc[i]['low'])
                                test_volumes.append(price_data.iloc[i]['volume'])
                        else:
                            if abs(price_data.iloc[i]['high'] - level) <= proximity_threshold:
                                test_prices.append(price_data.iloc[i]['high'])
                                test_volumes.append(price_data.iloc[i]['volume'])
                    
                    test_count = len(test_prices)
                    if test_count >= 1:
                        importance = 0.6 + (test_count * 0.2)
                        
                        # Volume-weighted importance
                        if test_count > 0 and test_volumes:
                            avg_volume = price_data['volume'].mean()
                            test_volume_ratio = sum(test_volumes) / (test_count * avg_volume)
                            importance *= min(1.4, max(0.7, test_volume_ratio))
                        
                        # Boost key levels
                        if level in key_low_price_levels:
                            importance *= 1.3
                            
                        psychological_levels.append((level, importance))
        
        else:
            # For regular stocks: round number levels with enhanced importance scoring
            # Find appropriate increment based on price
            if price_max < 20:
                increments = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
                key_levels = [5.0, 10.0, 15.0, 20.0]
            elif price_max < 100:
                increments = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 80.0, 90.0, 100.0]
                key_levels = [25.0, 50.0, 75.0, 100.0]
            elif price_max < 1000:
                increments = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 750.0, 800.0, 900.0, 1000.0]
                key_levels = [100.0, 250.0, 500.0, 750.0, 1000.0]
            else:
                increments = [100.0, 200.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 5000.0, 10000.0]
                key_levels = [1000.0, 2500.0, 5000.0, 10000.0]
                
            for level in increments:
                if price_min <= level <= price_max:
                    # Check for historical tests with volume confirmation
                    test_prices = []
                    test_volumes = []
                    
                    for i in range(len(price_data)):
                        if level_type == LevelType.SUPPORT:
                            # For support, test proximity to lows
                            rel_proximity = abs(price_data.iloc[i]['low'] - level) / level
                            if rel_proximity <= proximity_threshold / level:
                                test_prices.append(price_data.iloc[i]['low'])
                                test_volumes.append(price_data.iloc[i]['volume'])
                        else:
                            # For resistance, test proximity to highs
                            rel_proximity = abs(price_data.iloc[i]['high'] - level) / level
                            if rel_proximity <= proximity_threshold / level:
                                test_prices.append(price_data.iloc[i]['high'])
                                test_volumes.append(price_data.iloc[i]['volume'])
                    
                    test_count = len(test_prices)
                    if test_count >= 1:
                        # Base importance with additional emphasis on number of tests
                        importance = 0.7 + (test_count * 0.25)
                        
                        # Volume-weighted importance
                        if test_count > 0 and test_volumes:
                            avg_volume = price_data['volume'].mean()
                            if avg_volume > 0:
                                test_volume_ratio = sum(test_volumes) / (test_count * avg_volume)
                                importance *= min(1.3, max(0.8, test_volume_ratio))
                        
                        # Additional importance for key psychological levels
                        if level in key_levels:
                            importance *= 1.25
                            
                        psychological_levels.append((level, importance))
                        
        return psychological_levels

    @classmethod
    def identify_price_levels(
        cls,
        price_data: pd.DataFrame,
        level_type: LevelType,
        current_price: float,
        latest_atr: Optional[float] = None,
        interval: str = "daily",
        importance_threshold: float = 2.0
    ) -> List[float]:
        """
        Identify support or resistance levels in the price data based only on real market data
        without synthetic projections or fallbacks.
        
        Args:
            price_data: DataFrame with OHLCV data
            level_type: Whether to find support or resistance levels
            current_price: Current market price
            latest_atr: ATR value for adaptive calculations, if available
            interval: Timeframe interval (e.g., "daily", "weekly", "5min")
            importance_threshold: Minimum importance threshold for returning levels
            
        Returns:
            List of identified price levels
        """
        if not cls._validate_price_data(price_data):
            return []
            
        # Check data size and adjust threshold if necessary
        data_length = len(price_data)
        if data_length < 50:
            # Lower threshold when working with limited data
            importance_threshold = max(1.0, importance_threshold * 0.7)
            logger.info(f"Limited data ({data_length} bars) for {level_type.value} levels on {interval}, lowering threshold to {importance_threshold:.2f}")
        
        logger.info(f"Identifying {level_type.value} levels for {interval} timeframe with {data_length} data points")
        
        # Determine price characteristics
        is_penny_stock = current_price < 1.0
        is_low_priced = 1.0 <= current_price < 5.0
        is_extended = interval in ["daily", "weekly", "monthly"]
        is_short_timeframe = interval in ["1min", "5min", "15min"]
        
        # Find swing points - the primary source of real support/resistance
        swing_points = cls._find_swing_points(price_data, level_type, interval=interval)
        
        # Determine proximity threshold for grouping levels
        # Set thresholds based on price range and volatility
        if is_penny_stock:
            # Enhanced penny stock handling with adaptive thresholds
            if latest_atr is not None:
                # Use ATR for volatility-adjusted thresholds, but ensure minimum spacing
                proximity_threshold = max(0.015, min(0.05, latest_atr * 0.8))
            else:
                # More fine-grained percentage for penny stocks
                if current_price < 0.1:
                    # Extra wide spacing for sub-penny stocks
                    proximity_threshold = max(0.015, current_price * 0.15)
                elif current_price < 0.5:
                    # Wider spacing for very low pennies
                    proximity_threshold = max(0.015, current_price * 0.08)
                else:
                    # Standard penny stock spacing
                    proximity_threshold = max(0.015, current_price * 0.05)
        elif is_low_priced:
            # Low-priced stocks ($1-$5) with intermediate handling
            if latest_atr is not None:
                proximity_threshold = max(0.03, min(0.15, latest_atr * 0.6))
            else:
                proximity_threshold = max(0.03, current_price * 0.025)
        else:
            # Regular stocks with precision based on price magnitude
            if latest_atr is not None:
                # Volatility-based spacing with minimum relative to price
                rel_min = current_price * 0.01  # 1% of price minimum
                proximity_threshold = max(rel_min, latest_atr * 0.5)
            else:
                # Price-based spacing with adjustments for higher priced stocks
                if current_price < 50:
                    proximity_threshold = current_price * 0.015  # 1.5% spacing
                elif current_price < 200:
                    proximity_threshold = current_price * 0.012  # 1.2% spacing
                else:
                    proximity_threshold = current_price * 0.01   # 1.0% spacing
                
        # Initialize levels list
        levels = []
        
        # Process swing points if available - the most reliable source of levels
        if swing_points:
            # Calculate average volume for relative comparison
            avg_volume = price_data['volume'].mean()
            
            # Calculate importance for each swing point with enhanced volume weighting
            for price, timestamp, volume in swing_points:
                # Calculate basic bounce strength
                bounce_strength = cls._calculate_bounce_strength(
                    price_data, price, timestamp, level_type, latest_atr)
                    
                # Adjust importance based on volume with more emphasis for key levels
                importance = cls._adjust_importance_for_volume(bounce_strength, volume, avg_volume)
                
                # Add to levels list
                levels.append((price, importance))
        
        # Add psychological price levels - these are real market phenomena, not synthetic projections
        psych_levels = cls._add_psychological_levels(current_price, price_data, level_type, proximity_threshold)
        levels.extend(psych_levels)
        
        # Add Fibonacci retracement levels calculated from actual price data
        fib_levels = cls._add_fibonacci_levels(price_data, current_price, level_type)
        levels.extend(fib_levels)
        
        # Group nearby levels into zones
        zones = cls._group_nearby_levels(levels, price_data, level_type, latest_atr)
        
        # Apply basic threshold adjustment for different asset types
        # This is to account for natural differences in how different assets form levels
        adjusted_threshold = importance_threshold
        
        if is_penny_stock:
            adjusted_threshold *= 0.85  # 15% lower threshold for penny stocks
        elif is_low_priced:
            adjusted_threshold *= 0.9   # 10% lower threshold for low-priced stocks
        
        # Filter by importance threshold - no fallbacks
        filtered_levels = []
        for price, importance in zones:
            if importance >= adjusted_threshold:
                filtered_levels.append((price, importance))
                        
        # Sort final filtered levels by price
        filtered_levels.sort(key=lambda x: x[0])
        
        # Return just the prices, rounded to appropriate precision
        if is_penny_stock:
            # Higher precision for penny stocks
            return [round(price, 4 if price < 0.1 else 3) for price, _ in filtered_levels]
        else:
            # Standard precision for regular stocks
            return [round(price, 2) for price, _ in filtered_levels]
        
    @classmethod
    def identify_support_levels(
        cls,
        price_data: pd.DataFrame,
        current_price: float,
        latest_atr: Optional[float] = None,
        interval: str = "daily",
        importance_threshold: float = 2.0
    ) -> List[float]:
        """
        Identify support levels including those slightly above current price
        
        Args:
            price_data: DataFrame with OHLCV data
            current_price: Current market price
            latest_atr: ATR value for adaptive calculations, if available
            interval: Timeframe interval (e.g., "daily", "weekly", "5min")
            importance_threshold: Minimum importance threshold for returning levels
            
        Returns:
            List of support levels
        """
        levels = cls.identify_price_levels(
            price_data,
            LevelType.SUPPORT,
            current_price,
            latest_atr,
            interval,
            importance_threshold
        )
        
        # Include support levels slightly above current price (up to 5% for crypto)
        # This matches professional trading practice, especially for volatile crypto markets
        is_crypto = "crypto" in interval.lower() or current_price > 100  # Simple crypto detection
        buffer_pct = 0.05 if is_crypto else 0.03  # 5% for crypto, 3% for stocks
        
        # Return levels that are below current price or within the buffer zone above it
        return [level for level in levels if level < current_price * (1 + buffer_pct)]
        
    @classmethod
    def identify_resistance_levels(
        cls,
        price_data: pd.DataFrame,
        current_price: float,
        latest_atr: Optional[float] = None,
        interval: str = "daily",
        importance_threshold: float = 2.0
    ) -> List[float]:
        """
        Identify resistance levels including those slightly below current price
        
        Args:
            price_data: DataFrame with OHLCV data
            current_price: Current market price
            latest_atr: ATR value for adaptive calculations, if available
            interval: Timeframe interval (e.g., "daily", "weekly", "5min")
            importance_threshold: Minimum importance threshold for returning levels
            
        Returns:
            List of resistance levels
        """
        levels = cls.identify_price_levels(
            price_data,
            LevelType.RESISTANCE,
            current_price,
            latest_atr,
            interval,
            importance_threshold
        )
        
        # Include resistance levels slightly below current price (up to 5% for crypto)
        # This matches professional trading practice, especially for volatile crypto markets
        is_crypto = "crypto" in interval.lower() or current_price > 100  # Simple crypto detection
        buffer_pct = 0.05 if is_crypto else 0.03  # 5% for crypto, 3% for stocks
        
        # Return levels that are above current price or within the buffer zone below it
        return [level for level in levels if level > current_price * (1 - buffer_pct)]
        
    @classmethod
    def consolidate_multi_timeframe_levels(
        cls,
        all_levels: Dict[str, List[float]],
        current_price: float,
        latest_atr: Optional[float] = None,
        level_type: LevelType = LevelType.SUPPORT
    ) -> List[Dict[str, Any]]:
        """
        Consolidate support or resistance levels across multiple timeframes
        
        Args:
            all_levels: Dictionary mapping timeframes to level lists
            current_price: Current market price
            latest_atr: ATR value for adaptive calculations, if available
            level_type: Whether these are support or resistance levels
            
        Returns:
            List of consolidated level dictionaries with price and strength
        """
        if not all_levels:
            return []
            
        # Extract all individual levels
        all_individual_levels = []
        for timeframe, levels in all_levels.items():
            for level in levels:
                all_individual_levels.append((level, timeframe))
                
        # Return empty if no levels
        if not all_individual_levels:
            return []
            
        # Sort by price
        all_individual_levels.sort(key=lambda x: x[0])
        
        # Determine grouping distance based on ATR or price percentage
        if latest_atr is not None:
            grouping_distance = latest_atr * 0.75
        else:
            # Use a percentage of current price
            grouping_distance = current_price * 0.01
            
        # Group nearby levels
        consolidated_zones = []
        current_zone = {
            "price": all_individual_levels[0][0],
            "timeframes": [all_individual_levels[0][1]],
            "count": 1
        }
        
        for level, timeframe in all_individual_levels[1:]:
            if abs(level - current_zone["price"]) <= grouping_distance:
                # Add to current zone
                current_zone["price"] = (current_zone["price"] * current_zone["count"] + level) / (current_zone["count"] + 1)
                current_zone["timeframes"].append(timeframe)
                current_zone["count"] += 1
            else:
                # Finalize current zone and start a new one
                if current_zone and "timeframes" in current_zone and current_zone["timeframes"]:
                    current_zone["strength"] = cls._calculate_zone_strength(current_zone, level_type)
                    consolidated_zones.append(current_zone)
                
                current_zone = {
                    "price": level,
                    "timeframes": [timeframe],
                    "count": 1
                }
                
        # Add the last zone
        if current_zone and "timeframes" in current_zone and current_zone["timeframes"]:
            current_zone["strength"] = cls._calculate_zone_strength(current_zone, level_type)
            consolidated_zones.append(current_zone)
        
        # Sort zones by strength (descending)
        consolidated_zones.sort(key=lambda x: x["strength"], reverse=True)
        
        # Format for return
        result = []
        for zone in consolidated_zones:
            # Filter based on level type (for support below price, for resistance above price)
            if (level_type == LevelType.SUPPORT and zone["price"] >= current_price) or \
               (level_type == LevelType.RESISTANCE and zone["price"] <= current_price):
                continue
                
            result.append({
                "price": round(zone["price"], 2),
                "strength": zone["strength"],
                "timeframes": sorted(set(zone["timeframes"])),
                "confidence": cls._calculate_confidence(zone)
            })
            
        return result
        
    @staticmethod
    def _calculate_zone_strength(zone: Dict[str, Any], level_type: LevelType) -> float:
        """Calculate strength score for a multi-timeframe zone"""
        # Base strength from the number of timeframes that confirm this level
        base_strength = zone["count"] * 1.5
        
        # Boost strength for higher timeframes
        timeframe_boosts = {
            "monthly": 3.0,
            "weekly": 2.0,
            "daily": 1.5,
            "60min": 1.2,
            "30min": 1.1,
            "15min": 1.0,
            "5min": 0.9,
            "1min": 0.8
        }
        
        timeframe_boost = sum(timeframe_boosts.get(tf, 1.0) for tf in zone["timeframes"])
        
        return base_strength * (timeframe_boost / len(zone["timeframes"]))
        
    @staticmethod
    def _calculate_confidence(zone: Dict[str, Any]) -> str:
        """Calculate confidence rating based on zone strength"""
        strength = zone["strength"]
        
        if strength >= 12:
            return "very high"
        elif strength >= 8:
            return "high"
        elif strength >= 5:
            return "medium"
        elif strength >= 3:
            return "low"
        else:
            return "very low"