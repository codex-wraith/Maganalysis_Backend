from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
import logging
import os
from datetime import datetime, UTC

# Set up logging for MCP
logging.basicConfig(
    level=os.environ.get("MCP_LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
mcp_logger = logging.getLogger("maganalysis.mcp")

# Initialize the MCP server with error handlers
mcp = FastMCP(
    "Maganalysis", 
    error_logger=mcp_logger,
    max_context_tokens=32000,  # Adjust based on model requirements
    max_request_timeout=60000  # 60 seconds max for requests
)

# Import app, http, and market_manager from main
# Import app only when needed, not at module level
def get_app_dependencies():
    from main import app
    http = app.state.http          # shared aiohttp session
    mm = app.state.market_manager  # market manager instance
    return app, http, mm

# Register all MCP tools
from market import mcp_tools, mcp_multi_timeframe, mcp_price_levels
from aiagent import CipherAgent

# Create shared agent instance
agent = CipherAgent()  # shared LLM helper

# Tools are registered by the agent at initialization time

# Export only the mcp instance for use in main.py
__all__ = ["mcp"]

# Register global error handler
@mcp.on_error
async def handle_global_error(error, request_data=None):
    """Global error handler for MCP server"""
    error_id = f"err-{os.urandom(4).hex()}"
    mcp_logger.error(f"MCP Error {error_id}: {error}", exc_info=True)
    
    # Include request context for debugging, but sanitize sensitive info
    if request_data:
        sanitized_data = request_data.copy()
        if 'context' in sanitized_data:
            sanitized_data['context'] = "REDACTED"
        mcp_logger.debug(f"Request data for error {error_id}: {sanitized_data}")
    
    # Return user-friendly error
    return {
        "error": True,
        "error_id": error_id,
        "message": "An unexpected error occurred. Our team has been notified."
    }

# Health probe tool
@mcp.tool("ping")
async def ping():
    return "pong"

@mcp.on_conversation_start
async def provide_memory_context(client_context):
    """
    Provides memory context when a conversation starts.
    
    Args:
        client_context: The MCP client context
    """
    # Extract user information from the context
    user_id = client_context.get("user_id")
    platform = client_context.get("platform", "web")
    
    if user_id:
        try:
            # Get conversation history
            conversation_history = await app.state.agent.message_memory.get_conversation_history(
                platform=platform,
                user_id=user_id,
                limit=15  # Include substantial history for better context
            )
            
            # Format and add the conversation history
            formatted_history = []
            for msg in conversation_history:
                role = "assistant" if msg.get('is_response') else "user"
                formatted_history.append({
                    "role": role,
                    "content": msg.get('text', '')
                })
            
            # Ensure client_context exists on the mcp object
            if not hasattr(mcp, "client_context"):
                mcp.client_context = {}
                
            # Store the history in the client context
            mcp.client_context[user_id] = {
                "conversation_history": formatted_history,
                "platform": platform,
                "last_updated": datetime.now(UTC).isoformat()
            }
            
            # Log access for debugging
            mcp_logger.debug(f"Memory context provided for user {user_id} on {platform}: {len(formatted_history)} messages")
            
        except Exception as e:
            mcp_logger.error(f"Error providing memory context: {e}", exc_info=True)

# Market overview resource
@mcp.resource("market_overview")
async def get_market_overview():
    """Get market overview data focused on market sentiment"""
    # Get crypto and general market sentiment data
    crypto_data = []
    market_mood = None
    sentiment_score = None
    article_count = 0
    market_article_count = 0
    
    # Get overall market sentiment
    try:
        market_sentiment = await mm.get_news_sentiment(
            topics="financial_markets", 
            limit=50,
            http_session=http
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
        mcp_logger.error(f"Error fetching market sentiment: {e}")
    
    # Get Bitcoin sentiment
    try:
        btc_sentiment = await mm.get_news_sentiment(ticker="BTC", http_session=http)
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
    except Exception as e:
        mcp_logger.error(f"Error fetching crypto sentiment: {e}")
    
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
        "timestamp": datetime.now(UTC).isoformat()
    }

@mcp.resource("stock/{symbol}")
async def get_stock_info(symbol: str):
    """Get basic information for a stock symbol"""
    try:
        # Use the market manager to get stock information
        overview = await mm.get_stock_overview(symbol, http_session=http)
        return overview
    except Exception as e:
        raise ToolError(f"Failed to get stock information for {symbol}: {str(e)}")

@mcp.resource("crypto/{symbol}")
async def get_crypto_info(symbol: str):
    """Get basic information for a cryptocurrency"""
    try:
        # Get current price
        data = await mm.get_crypto_daily(symbol, "USD", http_session=http)
        # Extract and return formatted information
        return data
    except Exception as e:
        raise ToolError(f"Failed to get crypto information for {symbol}: {str(e)}")

# Add health probe tool
@mcp.tool("ping")
async def ping():
    return "pong"

# Allow listing tools for debugging
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "list-tools":
        # Create minimal dependencies for tool registration
        import asyncio
        import aiohttp
        from aiagent import CipherAgent
        from market.market_manager import MarketManager
        
        async def init_and_list_tools():
            # Create a temporary session just for the test
            async with aiohttp.ClientSession() as session:
                # Create minimal app state
                class App:
                    class State:
                        def __init__(self):
                            self.http = session
                            self.cache = {}
                            self.market_manager = MarketManager(session)
                    
                    def __init__(self):
                        self.state = self.State()
                
                app_mock = App()
                
                # Create agent
                agent = CipherAgent()
                
                # Register tools 
                from market import mcp_tools, mcp_multi_timeframe, mcp_price_levels
                mcp_tools.register_market_tools(mcp, app_mock, app_mock.state.market_manager, session, agent)
                mcp_multi_timeframe.register_multi_timeframe_tools(mcp, app_mock, app_mock.state.market_manager, session, agent)
                mcp_price_levels.register_price_level_tools(mcp, app_mock, app_mock.state.market_manager, session, agent)
                
                # List tools
                tools = mcp.list_tools()
                print("Available MCP tools:")
                for tool in tools:
                    print(f"- {tool}")
        
        try:
            asyncio.run(init_and_list_tools())
        except Exception as e:
            print(f"Error listing tools: {e}")
            import traceback
            traceback.print_exc()