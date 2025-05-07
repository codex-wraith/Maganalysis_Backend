import asyncio
import aiohttp
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
from aiagent import CipherAgent
from social_media_handler import SocialMediaHandler 
from datetime import datetime, UTC
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from hypercorn.config import Config
from hypercorn.asyncio import serve
from aisettings import AISettings
from market.market_manager import MarketManager
from market.indicators import TechnicalIndicators, TimeframeParameters
from market.data_processor import MarketDataProcessor
from mcp_server import mcp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    COMPLETE REPLACEMENT of the previous implementation to use MCP exclusively.
    """
    # Startup
    try:
        # Initialize basic services
        redis_url = os.environ.get('REDISCLOUD_URL')
        if not redis_url:
            logger.warning("REDISCLOUD_URL not found in environment")
            
        together_api_key = os.environ.get('TOGETHER_API_KEY')
        if not together_api_key:
            logger.warning("TOGETHER_API_KEY not found in environment")
            
        # Create a shared aiohttp session with no timeout to allow long-running operations
        shared_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        app.state.http = shared_session
        
        # Initialize simple in-memory cache for Alpha Vantage
        app.state.cache = {}
        
        # Initialize Market Manager for market data access first
        logger.info("Initializing MarketManager...")
        app.state.settings = AISettings()
        app.state.market_manager = MarketManager(shared_session)
        
        # Note: We're skipping MCP initialization to avoid URL validation issues
        logger.info("Skipping MCP server initialization due to URL validation issues")
        
        # Initialize core components with direct implementation
        logger.info("Initializing CipherAgent...")
        app.state.agent = CipherAgent()
        app.state.agent.market_manager = app.state.market_manager
        
        # Keep a reference to mcp for backward compatibility, but don't use it
        app.state.mcp = None
        
        # Initialize SocialMediaHandler directly
        logger.info("Initializing SocialMediaHandler...")
        app.state.social_media_handler = SocialMediaHandler(app.state.agent)
        
        # Initialize simple job store for async tasks
        app.state.jobs = {}
        
        # Initialize Telegram API
        logger.info("Initializing social media APIs...")
        await app.state.social_media_handler.initialize_apis()
        
        # Start Telegram polling if the bot exists
        if app.state.social_media_handler.telegram_bot:
            logger.info("Starting Telegram bot polling...")
            try:
                await app.state.social_media_handler.start_telegram_polling()
                logger.info("Telegram bot polling started successfully")
            except Exception as e:
                logger.error(f"Failed to start Telegram polling: {e}")

        yield

    finally:
        # Shutdown sequence
        logger.info("Starting shutdown sequence...")
        
        if hasattr(app.state, 'social_media_handler'):
            await app.state.social_media_handler.stop_telegram()
        
        # Close the shared aiohttp session
        if hasattr(app.state, 'http') and not app.state.http.closed:
            await app.state.http.close()
        
        logger.info("Shutdown sequence completed")

app = FastAPI(lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    allow_origin_regex=".*",  # Explicitly allow all origins via regex
    max_age=86400  # Cache preflight requests for 24 hours
)

class Message(BaseModel):
    text: str
    platform: str = "telegram"
    user_id: str

@app.post("/chat")
async def chat(message: Message):
    """
    MCP-exclusive chat endpoint replacing previous implementation.
    This implementation only works with the MCP-based agent.
    """
    try:
        if not app.state.agent:
            raise HTTPException(
                status_code=503,
                detail="Chat service not initialized. Please try again later."
            )
            
        # Validate Web requests include a user_id (session ID)
        if message.platform == "web" and not message.user_id:
            raise HTTPException(
                status_code=400,
                detail="Web platform requires a session ID"
            )
            
        # For web platform, always use the Claude model with a lower max token count
        context = None
        if message.platform == "web":
            # Simple context to enforce Claude model and smaller token limit for web
            context = {
                "force_claude": True,
                "max_tokens": 8000
            }
        
        # Use the MCP-based response method (which is now the only implementation)
        response = await app.state.agent.respond(
            message.text, 
            platform=message.platform,
            user_id=message.user_id,
            context=context
        )
        
        # Ensure the response is a simple string for web platform
        if message.platform == 'web' and isinstance(response, dict) and 'content' in response:
            # Handle structured content format
            if isinstance(response['content'], list):
                # Extract text from content blocks
                simple_response = ""
                for block in response['content']:
                    if isinstance(block, dict) and 'text' in block:
                        simple_response += block['text']
                    elif isinstance(block, str):
                        simple_response += block
                response = simple_response
        
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Return a friendly error message instead of an HTTP error
        return {"response": "I'm having trouble with my AI circuits right now. Please try again later."}

@app.get("/health")
@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    try:
        if not hasattr(app.state, 'agent'):
            raise HTTPException(status_code=503, detail="Services not initialized")
        return {
            "status": "ok",
            "timestamp": datetime.now(UTC).isoformat(),
            "agent_status": "ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/market/overview")
async def get_market_overview():
    """Get market overview data including sentiment, top stocks, and indices"""
    try:
        if not app.state.market_manager:
            raise HTTPException(status_code=503, detail="Market manager not initialized")
            
        # Get crypto and general market sentiment data
        mm = app.state.market_manager
        
        crypto_data = []
        market_mood = None
        sentiment_score = None
        article_count = 0
        market_article_count = 0
        
        # Initialize variables to None to indicate no data available
        market_mood = None
        sentiment_score = None
        market_article_count = 0
        btc_mood = None
        btc_score = None
        article_count = 0
        top_stocks_data = {"gainers": [], "most_actively_traded": []}
        indices_data = []
            
        # Get overall market sentiment
        try:
            market_sentiment = await mm.get_news_sentiment(
                topics="financial_markets", 
                limit=50
            )
            
            if market_sentiment and "feed" in market_sentiment:
                market_articles = market_sentiment["feed"]
                market_article_count = len(market_articles)
                
                if market_article_count > 0:
                    market_total_score = 0
                    for article in market_articles:
                        score = article.get("overall_sentiment_score", 0)
                        market_total_score += score
                    
                    market_avg_score = market_total_score / market_article_count
                    sentiment_score = (market_avg_score + 1) * 50
                    
                    if sentiment_score > 65:
                        market_mood = "bullish"
                    elif sentiment_score < 35:
                        market_mood = "bearish"
                    else:
                        market_mood = "neutral"
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {e}")
        
        # Get Bitcoin sentiment
        try:
            btc_sentiment = await mm.get_news_sentiment(ticker="BTC")
            
            if btc_sentiment and "feed" in btc_sentiment:
                articles = btc_sentiment["feed"]
                article_count = len(articles)
                
                if article_count > 0:
                    total_score = 0
                    for article in articles:
                        score = article.get("overall_sentiment_score", 0)
                        total_score += score
                    
                    avg_score = total_score / article_count
                    btc_score = (avg_score + 1) * 50
                    
                    if btc_score > 65:
                        btc_mood = "bullish"
                    elif btc_score < 35:
                        btc_mood = "bearish"
                    else:
                        btc_mood = "neutral"
            
            # Add to crypto data
            crypto_data.append({
                "symbol": "BTC",
                "name": "Bitcoin",
                "price": btc_score,
                "change_percent": btc_mood
            })
        except Exception as e:
            logger.error(f"Error fetching crypto sentiment: {e}")
        
        # Get top stocks data (gainers, losers, most active)
        try:
            top_stocks_data = await mm.get_top_gainers_losers()
            if not top_stocks_data:
                top_stocks_data = {"gainers": [], "most_actively_traded": []}
        except Exception as e:
            logger.error(f"Error fetching top stocks: {e}")
            top_stocks_data = {"gainers": [], "most_actively_traded": []}
            
        # Get major stock indices (S&P 500, DOW, NASDAQ, etc.)
        try:
            index_quotes_result = await mm.get_index_quotes()
            indices_data = index_quotes_result.get("indices", [])
        except Exception as e:
            logger.error(f"Error fetching index quotes: {e}")
            indices_data = []
        
        # Return combined data
        return {
            "market_status": {
                "market_mood": market_mood,
                "sentiment_score": sentiment_score,
                "article_count": market_article_count,
                "source": "financial_markets"
            },
            "crypto_status": {
                "market_mood": btc_mood,
                "sentiment_score": btc_score,
                "article_count": article_count,
                "source": "BTC" 
            },
            "cryptos": crypto_data,
            "top_stocks": top_stocks_data,  # Add top stocks data
            "indices": indices_data,        # Add indices data
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market overview error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving market data")

@app.get("/market/data/{symbol}")
async def get_historical_data(symbol: str, timeframe: str = "1d"):
    """Get historical data for a given symbol and timeframe"""
    try:
        if not app.state.market_manager:
            raise HTTPException(status_code=503, detail="Market manager not initialized")
            
        mm = app.state.market_manager
        http = app.state.http
        
        # Map frontend timeframe to backend interval
        interval_map = {
            "1d": "daily",
            "1h": "60min",
            "15m": "15min",
            "5m": "5min",
            "1m": "1min"
        }
        
        interval = interval_map.get(timeframe, "daily")
        
        # Determine if this is a stock or crypto
        asset_type = "stock"
        if symbol.upper() in ["BTC", "ETH", "XRP", "LTC", "ADA", "DOGE", "USDT", "BNB"]:
            asset_type = "crypto"
            
        try:
            if asset_type == "stock":
                data = await mm.get_time_series_daily(symbol)
                return data
            else:
                data = await mm.get_crypto_daily(symbol, "USD")
                return data
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            raise HTTPException(status_code=500, detail="Error processing market data")
            
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving market data")

@app.get("/market/indicators/{symbol}")
async def get_technical_indicators(symbol: str, timeframe: str = "daily"):
    """Get technical indicators bypassing MCP tools"""
    try:
        if not app.state.market_manager:
            raise HTTPException(status_code=503, detail="Market manager not initialized")
            
        # Detect if this is a crypto symbol
        common_cryptos = ["BTC", "ETH", "USDT", "BNB", "XRP", "ADA", "DOGE", "SOL"]
        is_crypto = symbol.upper() in common_cryptos
        asset_type = "crypto" if is_crypto else "stock"
        
        # Use market manager directly
        mm = app.state.market_manager
        
        price_data = {}
        indicators = {}
        
        # Get time series data based on timeframe and asset type
        raw_data = None
        time_series = None
        
        try:
            if asset_type == "stock":
                if timeframe == "daily":
                    raw_data = await mm.get_time_series_daily(symbol)
                elif timeframe == "weekly":
                    raw_data = await mm.get_time_series_weekly(symbol)
                elif timeframe == "monthly":
                    raw_data = await mm.get_time_series_monthly(symbol)
                else:
                    # Default to intraday for other timeframes
                    raw_data = await mm.get_intraday_data(symbol, interval=timeframe)
            else:
                # For crypto
                if timeframe == "daily":
                    raw_data = await mm.get_crypto_daily(symbol, market="USD")
                elif timeframe == "weekly":
                    raw_data = await mm.get_crypto_weekly(symbol, market="USD")
                elif timeframe == "monthly":
                    raw_data = await mm.get_crypto_monthly(symbol, market="USD")
                else:
                    # Default to intraday for other timeframes
                    raw_data = await mm.get_crypto_intraday(symbol, market="USD", interval=timeframe)
                
            # Process the data
            if raw_data:
                # Extract time series key
                time_series_key = None
                for key in raw_data.keys():
                    if key.startswith("Time Series") or key == "data":
                        time_series_key = key
                        time_series = raw_data[key]
                        break
                
                if time_series:
                    # Get the latest data point
                    latest_date = sorted(time_series.keys(), reverse=True)[0]
                    latest_data = time_series[latest_date]
                    
                    # Extract price data
                    close_key = next((k for k in latest_data.keys() if 'close' in k.lower()), None)
                    open_key = next((k for k in latest_data.keys() if 'open' in k.lower()), None)
                    high_key = next((k for k in latest_data.keys() if 'high' in k.lower()), None)
                    low_key = next((k for k in latest_data.keys() if 'low' in k.lower()), None)
                    volume_key = next((k for k in latest_data.keys() if 'volume' in k.lower()), None)
                    
                    if close_key:
                        current_price = float(latest_data[close_key])
                        price_data["current"] = current_price
                    
                    if open_key:
                        price_data["open"] = float(latest_data[open_key])
                        
                    if high_key:
                        price_data["high"] = float(latest_data[high_key])
                        
                    if low_key:
                        price_data["low"] = float(latest_data[low_key])
                        
                    if volume_key:
                        price_data["volume"] = float(latest_data[volume_key])
                        
                    # Calculate change
                    if len(time_series.keys()) > 1:
                        prev_date = sorted(time_series.keys(), reverse=True)[1]
                        prev_data = time_series[prev_date]
                        if close_key and prev_data.get(close_key):
                            prev_close = float(prev_data[close_key])
                            price_data["change"] = current_price - prev_close
                            price_data["change_percent"] = ((current_price - prev_close) / prev_close) * 100
                    
                    # Process the time series into a DataFrame and calculate indicators
                    df = MarketDataProcessor.process_time_series_data(raw_data, time_series_key, asset_type)
                    if df is not None and not df.empty:
                        # Get timeframe-specific parameters
                        params = TimeframeParameters.get_parameters(timeframe)
                        
                        # Calculate RSI
                        rsi_value = TechnicalIndicators.calculate_rsi(df, period=params["rsi_period"])
                        if rsi_value is not None:
                            indicators["rsi"] = {
                                "value": rsi_value,
                                "signal": "oversold" if rsi_value < 30 else "overbought" if rsi_value > 70 else "neutral"
                            }
                        
                        # Calculate MACD
                        macd_value, signal_value = TechnicalIndicators.calculate_macd(df)
                        if macd_value is not None and signal_value is not None:
                            indicators["macd"] = {
                                "value": macd_value,
                                "signal_line": signal_value,
                                "signal": "bullish" if macd_value > signal_value else "bearish"
                            }
                        
                        # Calculate Stochastic
                        k_value, d_value = TechnicalIndicators.calculate_stochastic(df, k_period=params["stoch_k_period"], d_period=params["stoch_d_period"])
                        if k_value is not None and d_value is not None:
                            indicators["stochastic"] = {
                                "value": k_value,  # %K - renamed to "value" to match frontend expectations
                                "d_value": d_value,  # %D - renamed to "d_value" to match frontend expectations
                                "signal": "oversold" if k_value < 20 else "overbought" if k_value > 80 else "neutral"
                            }
                        
                        # Calculate Bollinger Bands
                        bbands = TechnicalIndicators.calculate_bbands(df, period=params["bbands_period"])
                        if bbands and all(v is not None for v in bbands.values()):
                            indicators["bbands"] = bbands
                        
                        # Calculate SMA & EMA
                        sma_value = TechnicalIndicators.calculate_sma(df, period=params["sma_period"])
                        if sma_value is not None:
                            indicators["sma"] = {
                                "value": sma_value,
                                "signal": "above" if price_data.get("current", 0) > sma_value else "below"
                            }
                        
                        ema_value = TechnicalIndicators.calculate_ema(df, period=params["ema_period"])
                        if ema_value is not None:
                            indicators["ema"] = {
                                "value": ema_value,
                                "signal": "above" if price_data.get("current", 0) > ema_value else "below"
                            }
                        
                        # Calculate ATR & ADX
                        atr_value = TechnicalIndicators.calculate_atr(df, period=params["atr_period"])
                        if atr_value is not None:
                            indicators["atr"] = {"value": atr_value}
                        
                        adx_value = TechnicalIndicators.calculate_adx(df, period=params["adx_period"])
                        if adx_value is not None:
                            indicators["adx"] = {
                                "value": adx_value,
                                "signal": "weak" if adx_value < 20 else "strong" if adx_value > 25 else "moderate"
                            }
                        
                        # Determine trend based on indicators
                        trend = {"direction": "neutral", "strength": "medium"}
                        
                        # Determine trend direction from multiple indicators
                        bullish_signals = 0
                        bearish_signals = 0
                        
                        # RSI
                        if "rsi" in indicators:
                            rsi = indicators["rsi"]["value"]
                            if rsi > 60:
                                bullish_signals += 1
                            elif rsi < 40:
                                bearish_signals += 1
                        
                        # MACD
                        if "macd" in indicators:
                            if indicators["macd"]["signal"] == "bullish":
                                bullish_signals += 1
                            else:
                                bearish_signals += 1
                        
                        # Moving averages
                        if "sma" in indicators and "ema" in indicators:
                            if indicators["sma"]["signal"] == "above" and indicators["ema"]["signal"] == "above":
                                bullish_signals += 1
                            elif indicators["sma"]["signal"] == "below" and indicators["ema"]["signal"] == "below":
                                bearish_signals += 1
                        
                        # Determine overall trend
                        if bullish_signals > bearish_signals + 1:
                            trend["direction"] = "bullish"
                        elif bearish_signals > bullish_signals + 1:
                            trend["direction"] = "bearish"
                        
                        # Determine strength (using ADX if available)
                        if "adx" in indicators:
                            trend["strength"] = indicators["adx"]["signal"]
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            
        # Return the data
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "price_data": price_data,
            "indicators": indicators,
            "trend": trend
        }
    except Exception as e:
        logger.error(f"Technical indicators error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving technical indicators")

@app.get("/market/news-sentiment/{symbol}")
async def get_news_sentiment(symbol: str):
    """Get news sentiment data bypassing MCP tools"""
    try:
        if not app.state.market_manager:
            raise HTTPException(status_code=503, detail="Market manager not initialized")
            
        # Use market manager directly
        mm = app.state.market_manager
        http = app.state.http
        
        sentiment_data = await mm.get_news_sentiment(ticker=symbol)
        
        # Process data into standard format
        if sentiment_data and "feed" in sentiment_data:
            articles = sentiment_data["feed"]
            article_count = len(articles)
            
            sentiment_score = 50  # Neutral default
            sentiment_label = "NEUTRAL"
            most_relevant_article = None
            
            if article_count > 0:
                # Calculate average sentiment
                total_score = 0
                for article in articles:
                    score = article.get("overall_sentiment_score", 0)
                    total_score += score
                
                avg_score = total_score / article_count
                sentiment_score = (avg_score + 1) * 50  # Convert from -1,1 to 0,100
                
                # Determine sentiment label
                if sentiment_score > 65:
                    sentiment_label = "BULLISH"
                elif sentiment_score < 35:
                    sentiment_label = "BEARISH"
                
                # Find most relevant article (highest relevance score)
                most_relevant_article = max(articles, key=lambda x: x.get("relevance_score", 0))
                
                # Clean up article
                if most_relevant_article:
                    most_relevant_article = {
                        "title": most_relevant_article.get("title", ""),
                        "url": most_relevant_article.get("url", ""),
                        "summary": most_relevant_article.get("summary", ""),
                        "published": most_relevant_article.get("time_published", ""),
                        "source": most_relevant_article.get("source", ""),
                        "sentiment_score": most_relevant_article.get("overall_sentiment_score", 0)
                    }
            
            # Extract the original feed (articles)
            original_feed = sentiment_data.get("feed", [])
            
            return {
                "symbol": symbol,
                "sentiment": {
                    "score": sentiment_score,
                    "label": sentiment_label
                },
                "news_count": article_count,
                "most_relevant_article": most_relevant_article,
                "feed": original_feed, # Added feed array here
                "timestamp": datetime.now(UTC).isoformat()
            }
        
        return {
            "symbol": symbol,
            "sentiment": {
                "score": 50,
                "label": "NEUTRAL"
            },
            "news_count": 0,
            "feed": [], # Add empty feed array for consistency
            "timestamp": datetime.now(UTC).isoformat()
        }
            
    except Exception as e:
        logger.error(f"News sentiment error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving news sentiment")

class AssetAnalysisRequest(BaseModel):
    symbol: str
    asset_type: str = "stock"  # "stock" or "crypto"
    market: str = "USD"        # Only used for crypto
    interval: str = "60min"    # 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
    user_id: str
    platform: str = "web"      # Target platform for formatting (web, telegram)

class JobResponse(BaseModel):
    job_id: str
    status: str
    
async def run_analysis_job(app, job_id: str, request: AssetAnalysisRequest):
    """
    Background task to run analysis and store results.
    This implementation completely replaces the previous version and works only with MCP.
    """
    try:
        # Create an MCP context for analysis
        from mcp.server.fastmcp import Context
        
        # Call the analyze_asset method (the MCP-only implementation)
        analysis = await app.state.agent.analyze_asset(
            symbol=request.symbol,
            asset_type=request.asset_type,
            market=request.market,
            interval=request.interval,
            for_user_display=True,  # Format output for direct user display
            platform=request.platform if hasattr(request, 'platform') else 'web'  # Pass platform for formatting
        )
        
        # Store the result in our jobs dictionary
        app.state.jobs[job_id]["status"] = "completed"
        app.state.jobs[job_id]["result"] = analysis
        app.state.jobs[job_id]["completed_at"] = datetime.now(UTC).isoformat()
        
        logger.info(f"Job {job_id} completed successfully")
    except Exception as e:
        # Store the error in our jobs dictionary
        logger.error(f"Job {job_id} failed: {e}")
        app.state.jobs[job_id]["status"] = "failed"
        app.state.jobs[job_id]["error"] = str(e)
        app.state.jobs[job_id]["completed_at"] = datetime.now(UTC).isoformat()

@app.post("/analyze-asset")
async def analyze_asset_endpoint(request: AssetAnalysisRequest):
    """Legacy endpoint - redirects to the job-based solution"""
    try:
        # Create a job and return its ID
        job_id = str(uuid.uuid4())
        app.state.jobs[job_id] = {
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
            "request": request.dict(),
        }
        
        # Start the analysis in the background without awaiting it
        asyncio.create_task(run_analysis_job(app, job_id, request))
        
        # Return the job_id and a message
        return {
            "analysis": f"Analysis for {request.symbol} is processing in the background. Please check back in a few moments.",
            "job_id": job_id
        }
    except Exception as e:
        logger.error(f"Asset analysis job creation error: {e}")
        return {"analysis": f"I'm having trouble analyzing {request.symbol} right now. Please try again later."}

@app.post("/jobs/analyze-asset")
async def create_analysis_job(request: AssetAnalysisRequest):
    """Start an asset analysis job and return immediately with a job ID"""
    try:
        if not app.state.agent:
            raise HTTPException(
                status_code=503,
                detail="Analysis service not initialized. Please try again later."
            )
            
        # Validate Web requests include a user_id (session ID)
        if not request.user_id:
            raise HTTPException(
                status_code=400,
                detail="Request requires a session ID"
            )
            
        # Create a unique job ID
        job_id = str(uuid.uuid4())
        
        # Store job in our jobs dictionary
        app.state.jobs[job_id] = {
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
            "request": request.dict(),
        }
        
        # Start the analysis in the background without awaiting it
        asyncio.create_task(run_analysis_job(app, job_id, request))
        
        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        logger.error(f"Job creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating analysis job: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a job and return the results if available"""
    if job_id not in app.state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = app.state.jobs[job_id]
    
    # Basic job info
    response = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
    }
    
    # Add completed timestamp if available
    if "completed_at" in job:
        response["completed_at"] = job["completed_at"]
    
    # Add result if job is completed
    if job["status"] == "completed" and "result" in job:
        response["result"] = job["result"]
    
    # Add error if job failed
    if job["status"] == "failed" and "error" in job:
        response["error"] = job["error"]
        
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

port = int(os.environ.get("PORT", 8000))

config = Config()
config.bind = [f"0.0.0.0:{port}"]

async def run():
    """Run the web app with background services managed by the lifespan context"""
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(run())