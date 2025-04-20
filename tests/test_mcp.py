import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from mcp_server import mcp

def test_mcp_initialization():
    """Test that MCP server is initialized correctly"""
    assert mcp is not None
    assert mcp.name == "Maganalysis"

def test_indicator():
    """Test that technical indicators tool is registered"""
    tools = mcp.list_tools()
    assert "get_technical_indicators" in tools
    
def test_market_tools():
    """Test that market tools are registered"""
    tools = mcp.list_tools()
    assert "get_raw_market_data" in tools
    assert "get_market_data" in tools
    assert "get_news_sentiment" in tools
    
def test_price_level_tools():
    """Test that price level tools are registered"""
    tools = mcp.list_tools()
    assert "get_price_levels" in tools
    assert "get_key_price_zones" in tools
    
def test_multi_timeframe_tools():
    """Test that multi-timeframe tool is registered"""
    tools = mcp.list_tools()
    assert "analyze_multi_timeframe" in tools
    
def test_health_probe():
    """Test that health probe is registered"""
    tools = mcp.list_tools()
    assert "ping" in tools