import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import aiohttp
import asyncio
from aisettings import AISettings
from moralis import evm_api

logger = logging.getLogger(__name__)

class MarketManager:
    def __init__(self, http_session: aiohttp.ClientSession):
        self.last_update = None
        self.current_market_cap: Optional[float] = None
        self.TOTAL_SUPPLY = 47_000_000  # 47 million tokens
        
        # Store the shared HTTP session
        self.http_session = http_session
        
        # Credentials for crypto market data from Moralis
        self.moralis_api_key = os.environ.get('MORALIS_API_KEY')
        self.contract_address = os.environ.get('MAGABNB_CONTRACT_ADDRESS')
        
        if not self.moralis_api_key:
            logger.warning("MORALIS_API_KEY not found in environment")
        if not self.contract_address:
            logger.warning("MAGABNB_CONTRACT_ADDRESS not found in environment")
        
        # Reset price cache on initialization to force a fresh update
        self._price_cache = None
        self._price_cache_time = None
        
        # Flag to avoid repeated API calls when no liquidity exists
        self.no_liquidity = False
        
        # Setup for Alpha Vantage stock data
        self.alphavantage_api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
        if not self.alphavantage_api_key:
            logger.warning("ALPHAVANTAGE_API_KEY not found in environment")
        self.alpha_base_url = "https://www.alphavantage.co/query"
    
    # ----------------------------
    # Crypto market (Moralis) methods
    # ----------------------------
    async def get_current_market_cap(self, force_update: bool = False) -> float:
        """
        Get current market cap.

        With force_update=True, the method fetches the latest token price from Moralis.
        Otherwise, it returns the cached market cap value.
        If no liquidity was previously detected, the API call is skipped (unless forced).
        """
        if self.no_liquidity and not force_update:
            return self.current_market_cap or 0
        if force_update or self._price_cache is None:
            await self._update_market_cap()
        return self.current_market_cap or 0

    async def _update_market_cap(self):
        """Fetch token price from Moralis and calculate market cap, handling errors gracefully."""
        params = {
            "chain": "bsc",
            "address": self.contract_address
        }
        try:
            # Attempt to get token price from Moralis - using ThreadPoolExecutor for sync call in async function
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, 
                    lambda: evm_api.token.get_token_price(
                        api_key=self.moralis_api_key,
                        params=params
                    )
                )
            logger.info(f"Moralis API result: {result}")
            if result and "usdPrice" in result:
                token_price = float(result["usdPrice"])
                self._price_cache = token_price
                self._price_cache_time = datetime.now()
                self.current_market_cap = token_price * self.TOTAL_SUPPLY
                self.last_update = self._price_cache_time
                self.no_liquidity = False  # Liquidity is available
                logger.info(f"Token Price: ${token_price:,.6f}")
            else:
                logger.info("Token is in pre-launch phase or no liquidity available.")
                self.current_market_cap = 0
                self.last_update = datetime.now()
                self.no_liquidity = True
        except Exception as inner_e:
            error_message = str(inner_e)
            if "Insufficient liquidity" in error_message:
                logger.info("Moralis API reports insufficient liquidity in pools to calculate the price. Marking token as pre-launch/no liquidity.")
            else:
                logger.error("Error updating market cap from Moralis API.", exc_info=True)
            self.current_market_cap = 0
            self.last_update = datetime.now()
            self.no_liquidity = True

    async def get_market_status(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Get market cap in a standardized format using crypto data.

        If force_update is True, the market cap is refreshed via an API call.
        """
        market_cap = await self.get_current_market_cap(force_update=force_update)
        return {
            "market_cap": market_cap
        }
    
    # ----------------------------
    # Stock market / broader market (Alpha Vantage) methods
    # ----------------------------
    async def _fetch_alpha_data(self, params: Dict[str, Any], http_session=None) -> Dict:
        """
        Helper function to perform an asynchronous GET request to Alpha Vantage.
        Ensures that the API key and realtime entitlement are added to the request parameters.
        """
        params['apikey'] = self.alphavantage_api_key
        params['entitlement'] = 'realtime'
        
        # Use provided session or the instance's session
        session = http_session or self.http_session
        
        try:
            async with session.get(self.alpha_base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Alpha Vantage API returned status {response.status} for function {params.get('function')}")
                    return {}
                data = await response.json()
                # Don't log the full API response as it can be very large
                logger.info(f"Alpha Vantage API response received for {params.get('function')}")
                
                return data
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage: {e}", exc_info=True)
            return {}

    async def get_top_gainers_losers(self) -> Dict:
        """
        Retrieve the top gainers, losers, and most actively traded stocks from the market.
        Assumes the API returns JSON with keys 'top_gainers', 'top_losers', and 'most_actively_traded'.
        """
        params = {
            "function": "TOP_GAINERS_LOSERS"
        }
        data = await self._fetch_alpha_data(params)
        top_gainers = data.get("top_gainers", [])
        most_active = data.get("most_actively_traded", [])
        return {
            "gainers": top_gainers,
            "most_actively_traded": most_active
        }

    async def get_insider_transactions(self, symbol: str) -> Dict:
        """
        Retrieve insider transactions for a given stock symbol from Alpha Vantage.
        (Not used in the updated full analysis.)
        """
        params = {
            "function": "INSIDER_TRANSACTIONS",
            "symbol": symbol
        }
        data = await self._fetch_alpha_data(params)
        return data

    async def get_intraday_data(
        self,
        symbol: str,
        interval: str = "5min",    # Could be "1min", "5min", "15min", "30min", or "60min"
        adjusted: bool = True,
        extended_hours: bool = True,
        outputsize: str = "full",
        datatype: str = "json",
        month: Optional[str] = None
    ) -> Dict:
        """
        Retrieve intraday OHLCV data (candles) for a specified equity using the
        TIME_SERIES_INTRADAY endpoint. This endpoint covers both pre-market and post-market hours.
        
        The allowed intervals are: "1min", "5min", "15min", "30min", and "60min".
        """
        allowed_intervals = ["1min", "5min", "15min", "30min", "60min"]
        if interval not in allowed_intervals:
            logger.warning(f"Provided interval {interval} is not allowed; defaulting to 5min.")
            interval = "5min"

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "adjusted": "true" if adjusted else "false",
            "extended_hours": "true" if extended_hours else "false",
            "outputsize": outputsize,
            "datatype": datatype,
        }
        if month:
            params["month"] = month

        data = await self._fetch_alpha_data(params)
        return data

    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Dict:
        """
        Retrieve the realtime exchange rate for any pair of digital or physical currencies.
        
        API Parameters:
          - function: "CURRENCY_EXCHANGE_RATE"
          - from_currency: The source currency (e.g., "BTC" or "USD").
          - to_currency: The destination currency (e.g., "USD" or "BTC").
        """
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency
        }
        data = await self._fetch_alpha_data(params)
        return data

    async def get_time_series_daily(self, symbol: str, outputsize: str = "full", datatype: str = "json", http_session: aiohttp.ClientSession = None) -> Dict:
        """
        Retrieve daily time series data for a given stock symbol using the TIME_SERIES_DAILY endpoint.
        
        Using 'full' outputsize to get up to 20 years of historical data (or as much as available).
        This is important for calculating long-term technical indicators like 50-day and 200-day
        moving averages, and for identifying significant support/resistance levels.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "datatype": datatype
        }
        data = await self._fetch_alpha_data(params)
        return data
    
    async def get_time_series_weekly(self, symbol: str, datatype: str = "json") -> Dict:
        """
        Retrieve weekly time series data for a given stock symbol using the TIME_SERIES_WEEKLY endpoint.
        """
        params = {
            "function": "TIME_SERIES_WEEKLY",
            "symbol": symbol,
            "datatype": datatype
        }
        data = await self._fetch_alpha_data(params)
        return data


    async def get_time_series_monthly(self, symbol: str, datatype: str = "json") -> Dict:
        """
        Retrieve monthly time series data for a given stock symbol using the TIME_SERIES_MONTHLY endpoint.
        """
        params = {
            "function": "TIME_SERIES_MONTHLY",
            "symbol": symbol,
            "datatype": datatype
        }
        data = await self._fetch_alpha_data(params)
        return data

    # ----------------------------
    # New Crypto Intraday Endpoint
    # ----------------------------
    async def get_crypto_intraday(
        self,
        symbol: str,
        market: str,
        interval: str = "5min",
        outputsize: str = "full",
        datatype: str = "json"
    ) -> Dict:
        """
        Retrieve intraday time series data for a specified cryptocurrency using the
        CRYPTO_INTRADAY endpoint.
        
        Required parameters:
          - function: "CRYPTO_INTRADAY"
          - symbol: The digital/crypto currency of your choice (e.g., ETH).
          - market: The exchange market of your choice (e.g., USD).
          - interval: Time interval between two consecutive data points (allowed: "1min", "5min", "15min", "30min", "60min").
          - outputsize (optional): "compact" returns only the latest 100 data points; "full" returns the full-length series.
          - datatype (optional): "json" (default) or "csv".
        """
        allowed_intervals = ["1min", "5min", "15min", "30min", "60min"]
        if interval not in allowed_intervals:
            logger.warning(f"Provided interval {interval} is not allowed; defaulting to 5min.")
            interval = "5min"

        params = {
            "function": "CRYPTO_INTRADAY",
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "outputsize": outputsize,
            "datatype": datatype
        }
        data = await self._fetch_alpha_data(params)
        return data
    
    async def get_crypto_daily(
        self,
        symbol: str,
        market: str,
        datatype: str = "json",
        http_session: aiohttp.ClientSession = None
    ) -> Dict:
        """
        Retrieve daily time series data for a specified cryptocurrency using the
        DIGITAL_CURRENCY_DAILY endpoint.
        """
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": market,
            "datatype": datatype
        }
        data = await self._fetch_alpha_data(params)
        return data

    async def get_crypto_weekly(
        self,
        symbol: str,
        market: str,
        datatype: str = "json"
    ) -> Dict:
        """
        Retrieve weekly time series data for a specified cryptocurrency using the
        DIGITAL_CURRENCY_WEEKLY endpoint.
        """
        params = {
            "function": "DIGITAL_CURRENCY_WEEKLY",
            "symbol": symbol,
            "market": market,
            "datatype": datatype
        }
        data = await self._fetch_alpha_data(params)
        return data

    async def get_crypto_monthly(
        self,
        symbol: str,
        market: str,
        datatype: str = "json"
    ) -> Dict:
        """
        Retrieve monthly time series data for a specified cryptocurrency using the
        DIGITAL_CURRENCY_MONTHLY endpoint.
        """
        params = {
            "function": "DIGITAL_CURRENCY_MONTHLY",
            "symbol": symbol,
            "market": market,
            "datatype": datatype
        }
        data = await self._fetch_alpha_data(params)
        return data

      
    async def get_index_quotes(self) -> Dict:
        """
        Retrieves real-time quotes for major market indices using their ETF proxies via GLOBAL_QUOTE endpoint.
        
        ETF proxies:
        - SPY: S&P 500 index
        - DIA: Dow Jones Industrial Average
        - QQQ: Nasdaq Composite
        """
        # Define index ETF proxies
        symbols = ["SPY", "DIA", "QQQ"]
        name_map = {
            "SPY": "S&P 500",
            "DIA": "Dow Jones",
            "QQQ": "Nasdaq"
        }
        
        # Format the data for frontend use
        indices = []
        
        # Make separate API calls for each symbol (more reliable than bulk quotes)
        for symbol in symbols:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol
            }
            
            data = await self._fetch_alpha_data(params)
            
            if data and isinstance(data, dict) and "Global Quote" in data:
                quote = data["Global Quote"]
                price = quote.get("05. price", "0.00")
                change = quote.get("09. change", "0.00")
                change_percent = quote.get("10. change percent", "0.00")
                
                # Only add if we got valid data
                if price and price != "":
                    indices.append({
                        "name": name_map.get(symbol, symbol),
                        "price": price,
                        "change": change,
                        "change_percent": change_percent
                    })
        
        return {"indices": indices}
        
    async def get_news_sentiment(self, ticker: str = None, topics: str = None, time_from: str = None, limit: int = 100, http_session: aiohttp.ClientSession = None) -> Dict:
        """
        Retrieve news sentiment data for a specified ticker and/or topics.
        
        This endpoint provides real-time and historical news articles with sentiment scores
        specifically for stocks and cryptocurrencies or broader market topics.
        
        Parameters:
            ticker (str, optional): The ticker symbol (e.g., "BTC" or "TSLA")
            topics (str, optional): Comma-separated list of topics (e.g., "financial_markets,economy_macro")
            time_from (str, optional): UTC timestamp (e.g., "20220410T000000") for the starting point of time range
            limit (int): Number of items to return (default: 100, max: 1000)
            
        Returns:
            Dict: News articles with sentiment scores
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "sort": "LATEST",  # Get most recent articles first
            "limit": limit
        }
        
        # Add ticker parameter if provided
        if ticker:
            # Format ticker to handle cryptocurrencies
            # Alpha Vantage expects "CRYPTO:BTC" format for cryptocurrencies
            formatted_ticker = ticker
            if ticker in ["BTC", "ETH", "DOGE", "LTC", "ADA", "AVAX", "BNB", "XRP", "SOL", "DOT"]:
                formatted_ticker = f"CRYPTO:{ticker}"
            params["tickers"] = formatted_ticker
            
        # Add topics parameter if provided
        if topics:
            params["topics"] = topics
            
        if time_from:
            params["time_from"] = time_from
            
        data = await self._fetch_alpha_data(params)
        return data