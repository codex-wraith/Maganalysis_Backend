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
from market.indicators import TechnicalIndicators

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
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
        
        # Initialize core components
        logger.info("Initializing CipherAgent...")
        app.state.agent = CipherAgent()
        
        # Initialize SocialMediaHandler directly
        logger.info("Initializing SocialMediaHandler...")
        app.state.social_media_handler = SocialMediaHandler(app.state.agent)
        
        # Initialize Market Manager for market data access
        logger.info("Initializing MarketManager...")
        app.state.settings = AISettings()
        app.state.market_manager = MarketManager(app.state.settings)
        
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
        if shared_session and not shared_session.closed:
            await shared_session.close()
        
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
                "force_claude": True,  # Add a flag to force Claude model
                "max_tokens": 8000     # Use smaller token limit for web
            }
        
        response = await app.state.agent.respond(
            message.text, 
            platform=message.platform,
            user_id=message.user_id,
            context=context
        )
        
        # Ensure the response is a simple string for web platform
        # Claude API might return Content objects sometimes
        if message.platform == 'web' and isinstance(response, dict) and 'content' in response:
            # Handle Claude structured content format
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
async def health_check():
    """Health check endpoint"""
    try:
        if not hasattr(app.state, 'agent'):
            raise HTTPException(status_code=503, detail="Services not initialized")
        return {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "agent_status": "ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/market/overview")
async def get_market_overview():
    """Get market overview data focused on Bitcoin sentiment"""
    try:
        if not hasattr(app.state, 'market_manager'):
            raise HTTPException(status_code=503, detail="Market service not initialized")
            
        # Get crypto and general market sentiment data
        crypto_data = []
        market_mood = None
        sentiment_score = None
        article_count = 0
        market_article_count = 0
        
        # Get overall market sentiment using just the financial_markets topic
        try:
            market_sentiment = await app.state.market_manager.get_news_sentiment(
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
            btc_sentiment = await app.state.market_manager.get_news_sentiment(ticker="BTC")
            btc_mood = None
            btc_score = None
            
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
            
            # Get Ethereum sentiment
            eth_sentiment = await app.state.market_manager.get_news_sentiment("ETH")
            eth_mood = None
            eth_score = None
            
            if eth_sentiment and "feed" in eth_sentiment:
                articles = eth_sentiment["feed"]
                if articles:
                    total_score = sum(article.get("overall_sentiment_score", 0) for article in articles)
                    avg_score = total_score / len(articles)
                    eth_score = (avg_score + 1) * 50
                    
                    if eth_score > 65:
                        eth_mood = "bullish"
                    elif eth_score < 35:
                        eth_mood = "bearish"
                    else:
                        eth_mood = "neutral"
            
            crypto_data.append({
                "symbol": "ETH",
                "name": "Ethereum",
                "price": eth_score,
                "change_percent": eth_mood
            })
            
            # Get Solana sentiment
            sol_sentiment = await app.state.market_manager.get_news_sentiment("SOL")
            sol_mood = None
            sol_score = None
            
            if sol_sentiment and "feed" in sol_sentiment:
                articles = sol_sentiment["feed"]
                if articles:
                    total_score = sum(article.get("overall_sentiment_score", 0) for article in articles)
                    avg_score = total_score / len(articles)
                    sol_score = (avg_score + 1) * 50
                    
                    if sol_score > 65:
                        sol_mood = "bullish"
                    elif sol_score < 35:
                        sol_mood = "bearish"
                    else:
                        sol_mood = "neutral"
            
            crypto_data.append({
                "symbol": "SOL",
                "name": "Solana",
                "price": sol_score,
                "change_percent": sol_mood
            })
            
        except Exception as e:
            logger.error(f"Error fetching crypto sentiment: {e}")
            crypto_data = [
                {"symbol": "BTC", "name": "Bitcoin", "price": None, "change_percent": None},
                {"symbol": "ETH", "name": "Ethereum", "price": None, "change_percent": None},
                {"symbol": "SOL", "name": "Solana", "price": None, "change_percent": None}
            ]
            # If we hadn't already set market_mood from financial_markets, make sure it's None
            if market_mood is None:
                market_mood = None
                sentiment_score = None
                market_article_count = 0
            
            # Set BTC-specific mood to None
            btc_mood = None
            btc_score = None
            article_count = 0
            
        # Get top gainers and losers
        try:
            top_data = await app.state.market_manager.get_top_gainers_losers()
        except Exception as e:
            logger.error(f"Error fetching top gainers/losers: {e}")
            top_data = {"gainers": [], "most_actively_traded": []}
            
        # Get index quotes (S&P 500, Dow Jones, Nasdaq)
        try:
            index_data = await app.state.market_manager.get_index_quotes()
            indices = index_data.get("indices", [])
        except Exception as e:
            logger.error(f"Error fetching index data: {e}")
            indices = []
            
        # Combine data with the crypto and market sentiment we've collected
        response_data = {
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
            "top_stocks": top_data,
            "indices": indices,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        return response_data
    except Exception as e:
        logger.error(f"Market overview error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving market data")

@app.get("/market/indicators/{symbol}")
async def get_technical_indicators(symbol: str, timeframe: str = "daily"):
    """Get technical indicators for a specific symbol (stock or crypto) with optional timeframe"""
    try:
        if not hasattr(app.state, 'market_manager'):
            raise HTTPException(status_code=503, detail="Market service not initialized")
            
        # Fetch data for the symbol (stock or crypto)
        try:
            # Detect if this is a crypto symbol
            common_cryptos = ["BTC", "ETH", "USDT", "BNB", "XRP", "ADA", "DOGE", "SOL"]
            is_crypto = symbol.upper() in common_cryptos
            
            # Normalize timeframe parameter
            valid_timeframes = ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"]
            if timeframe not in valid_timeframes:
                timeframe = "daily"  # Default to daily if invalid timeframe is provided
                
            # Determine if this is an extended timeframe or intraday
            is_extended = timeframe in ["daily", "weekly", "monthly"]
            asset_type = "crypto" if is_crypto else "stock"
            
            # Get data based on asset type and timeframe
            if is_crypto:
                if is_extended:
                    # For extended timeframes, use appropriate endpoint
                    if timeframe == "daily":
                        data = await app.state.market_manager.get_crypto_daily(symbol, "USD")
                    elif timeframe == "weekly":
                        data = await app.state.market_manager.get_crypto_weekly(symbol, "USD")
                    else:  # monthly
                        data = await app.state.market_manager.get_crypto_monthly(symbol, "USD")
                else:
                    # For intraday, use intraday endpoint
                    data = await app.state.market_manager.get_crypto_intraday(symbol, "USD", interval=timeframe)
            else:
                # For stocks
                if is_extended:
                    if timeframe == "daily":
                        data = await app.state.market_manager.get_time_series_daily(symbol)
                    elif timeframe == "weekly":
                        data = await app.state.market_manager.get_time_series_weekly(symbol)
                    else:  # monthly
                        data = await app.state.market_manager.get_time_series_monthly(symbol)
                else:
                    # For intraday, use intraday endpoint
                    data = await app.state.market_manager.get_intraday_data(symbol, interval=timeframe)
            
            # Process the data using the existing market data processor
            from market.data_processor import MarketDataProcessor
            
            # Find the time series key
            time_series_key = next((k for k in data.keys() if "Time Series" in k or "Digital Currency" in k), None)
            
            if not time_series_key or not data.get(time_series_key):
                # Raise an error instead of returning mock data
                raise ValueError(f"No time series data available for symbol {symbol} with timeframe {timeframe}")
            
            # Process the data
            df = MarketDataProcessor.process_time_series_data(data, time_series_key, asset_type)
            if df is None or df.empty:
                raise ValueError("Failed to process time series data")
                
            # Calculate all technical indicators
            rsi = TechnicalIndicators.calculate_rsi(df)
            macd, signal = TechnicalIndicators.calculate_macd(df)
            k_value, d_value = TechnicalIndicators.calculate_stochastic(df)
            bbands = TechnicalIndicators.calculate_bbands(df)
            sma = TechnicalIndicators.calculate_sma(df)
            ema = TechnicalIndicators.calculate_ema(df)
            atr = TechnicalIndicators.calculate_atr(df)
            adx = TechnicalIndicators.calculate_adx(df)
            vwap = TechnicalIndicators.calculate_vwap(df)
            
            # Verify all required indicators were calculated successfully
            if any(x is None for x in [rsi, macd, signal, k_value, d_value, sma, ema, atr, adx]) or \
               any(bbands[x] is None for x in ["upper", "middle", "lower"]):
                raise ValueError(f"Failed to calculate one or more technical indicators for {symbol} with timeframe {timeframe}")
            
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
            rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            macd_signal = "bullish" if macd > signal else "bearish"
            stoch_signal = "oversold" if k_value < 20 else "overbought" if k_value > 80 else "neutral"
            
            # Determine trend signals from moving averages
            sma_signal = "bullish" if current_price > sma else "bearish"
            ema_signal = "bullish" if current_price > ema else "bearish"
            
            # Calculate ADX trend strength signal
            adx_signal = "weak" if adx < 20 else "strong" if adx > 25 else "moderate"
            
            # Calculate overall trend indication
            trend_indicators = [
                1 if sma_signal == "bullish" else -1,
                1 if ema_signal == "bullish" else -1,
                1 if macd_signal == "bullish" else -1,
                1 if rsi > 50 else -1,
                1 if k_value > 50 else -1
            ]
            
            trend_score = sum(trend_indicators)
            overall_signal = "buy" if trend_score >= 3 else "sell" if trend_score <= -3 else "hold"
            
            # Build comprehensive response
            return {
                "symbol": symbol,
                "asset_type": asset_type,
                "indicators": {
                    "rsi": {"value": rsi, "signal": rsi_signal},
                    "macd": {"value": macd, "signal": macd_signal, "signal_line": signal},
                    "stochastic": {"value": k_value, "d_value": d_value, "signal": stoch_signal},
                    "bbands": {
                        "upper": bbands["upper"],
                        "middle": bbands["middle"],
                        "lower": bbands["lower"]
                    },
                    "sma": {"value": sma, "signal": sma_signal},
                    "ema": {"value": ema, "signal": ema_signal},
                    "atr": {"value": atr},
                    "adx": {"value": adx, "signal": adx_signal},
                    "vwap": {"value": vwap}
                },
                "price_data": {
                    "current": current_price,
                    "change": change,
                    "change_percent": change_percent,
                    "high": float(latest_data['high']),
                    "low": float(latest_data['low']),
                    "volume": float(latest_data['volume']) if 'volume' in latest_data else 0
                },
                "analysis": {
                    "trend_score": trend_score,
                    "overall_signal": overall_signal,
                    "strength": abs(trend_score) / 5 * 100  # Convert to percentage
                },
                "timestamp": datetime.now(UTC).isoformat()
            }
                
        except Exception as e:
            logger.error(f"Error fetching technical indicators for {symbol}: {e}")
            # Raise an HTTP exception instead of returning mock data
            raise HTTPException(status_code=500, detail=f"Failed to fetch technical data for {symbol}: {str(e)}")
            
    except Exception as e:
        logger.error(f"Technical indicators error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving technical indicators")

@app.get("/market/news-sentiment/{symbol}")
async def get_news_sentiment(symbol: str):
    """Get news sentiment data for a specific symbol"""
    try:
        if not hasattr(app.state, 'market_manager'):
            raise HTTPException(status_code=503, detail="Market service not initialized")
            
        # Fetch news sentiment from market manager
        try:
            sentiment_data = await app.state.market_manager.get_news_sentiment(symbol)
            if sentiment_data and "feed" in sentiment_data:
                articles = sentiment_data["feed"]
                
                # Calculate overall sentiment
                total_score = 0
                positive = 0
                neutral = 0
                negative = 0
                
                for article in articles:
                    score = article.get("overall_sentiment_score", 0)
                    total_score += score
                    
                    if score > 0.25:
                        positive += 1
                    elif score < -0.25:
                        negative += 1
                    else:
                        neutral += 1
                
                count = len(articles)
                avg_score = total_score / count if count > 0 else 0
                
                # Normalize to 0-100 scale
                normalized_score = (avg_score + 1) * 50
                
                # Calculate percentages
                pos_pct = int((positive / count) * 100) if count > 0 else 0
                neu_pct = int((neutral / count) * 100) if count > 0 else 0
                neg_pct = int((negative / count) * 100) if count > 0 else 0
                
                return {
                    "symbol": symbol,
                    "sentiment": {
                        "score": normalized_score,
                        "positive": pos_pct,
                        "neutral": neu_pct,
                        "negative": neg_pct
                    },
                    "news_count": count,
                    "feed": articles,
                    "timestamp": datetime.now(UTC).isoformat()
                }
        except Exception as e:
            logger.error(f"Error processing news sentiment: {e}")
        
        # Return data indicating no sentiment available if API call failed
        return {
            "symbol": symbol,
            "sentiment": {
                "score": None,
                "positive": None,
                "neutral": None,
                "negative": None
            },
            "news_count": 0,
            "feed": [],
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

class JobResponse(BaseModel):
    job_id: str
    status: str
    
async def run_analysis_job(app, job_id: str, request: AssetAnalysisRequest):
    """Background task to run analysis and store results"""
    try:
        # Call the analyze_asset method directly from the agent
        analysis = await app.state.agent.analyze_asset(
            symbol=request.symbol,
            asset_type=request.asset_type,
            market=request.market,
            interval=request.interval,
            for_user_display=True  # Format output for direct user display
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