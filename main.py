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
        
        # Initialize MCP server first (new order of initialization)
        app.state.mcp = mcp
        logger.info("Initialized MCP server")
        
        # Initialize core components with MCP-exclusive implementation
        logger.info("Initializing MCP-based CipherAgent...")
        app.state.agent = CipherAgent()
        
        # Initialize SocialMediaHandler directly
        logger.info("Initializing SocialMediaHandler...")
        app.state.social_media_handler = SocialMediaHandler(app.state.agent)
        
        # Initialize Market Manager for market data access
        logger.info("Initializing MarketManager...")
        app.state.settings = AISettings()
        app.state.market_manager = MarketManager(app.state.settings)
        app.state.agent.market_manager = app.state.market_manager
        
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
    """Get market overview data using MCP resource"""
    try:
        if not app.state.mcp:
            raise HTTPException(status_code=503, detail="MCP service not initialized")
            
        # Use MCP resource for market overview
        market_overview = await mcp.resources.get("market_overview")
        return market_overview
        
    except Exception as e:
        logger.error(f"Market overview error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving market data")

@app.get("/market/indicators/{symbol}")
async def get_technical_indicators(symbol: str, timeframe: str = "daily"):
    """Get technical indicators using MCP tools"""
    try:
        if not app.state.mcp:
            raise HTTPException(status_code=503, detail="MCP service not initialized")
            
        # Detect if this is a crypto symbol
        common_cryptos = ["BTC", "ETH", "USDT", "BNB", "XRP", "ADA", "DOGE", "SOL"]
        is_crypto = symbol.upper() in common_cryptos
        asset_type = "crypto" if is_crypto else "stock"
        
        # Use MCP tool to get technical indicators
        from market.mcp_tools import get_technical_indicators
        tech_data = await get_technical_indicators(symbol, asset_type, timeframe)
        
        # Check for error
        if "error" in tech_data:
            raise HTTPException(status_code=500, detail=tech_data["error"])
            
        return tech_data
            
    except Exception as e:
        logger.error(f"Technical indicators error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving technical indicators")

@app.get("/market/news-sentiment/{symbol}")
async def get_news_sentiment(symbol: str):
    """Get news sentiment data using MCP tool"""
    try:
        if not app.state.mcp:
            raise HTTPException(status_code=503, detail="MCP service not initialized")
            
        # Use MCP tool to get news sentiment
        from market.mcp_tools import get_news_sentiment
        sentiment_data = await get_news_sentiment(symbol)
        
        return sentiment_data
            
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
        from mcp.client import Context
        
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