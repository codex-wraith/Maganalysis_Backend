# Fix for ImportError in aiagent.py

## Problem

The application was encountering an `ImportError` when trying to import MCP tools directly from their module files:

```
ImportError: cannot import name 'get_technical_indicators' from 'market.mcp_tools' (/app/market/mcp_tools.py)
```

This occurred because these functions are registered as MCP tools *inside* registration functions and are not available as top-level module imports.

## Solution

The fix involved two main changes:

1. **Removed direct imports of MCP tools**:
   - Removed `get_technical_indicators`, `get_news_sentiment`, `get_raw_market_data` from `market.mcp_tools`
   - Removed `get_price_levels` from `market.mcp_price_levels`
   - Removed `analyze_multi_timeframe` from `market.mcp_multi_timeframe`
   - Kept `add_market_analysis_to_context` from `market.mcp_tools` as it's used as a helper function

2. **Modified all function calls to use the MCP instance**:
   - Changed all direct calls to these functions to use the pattern `mcp.tools.<tool_name>()`
   - Updated function calls to use named parameters for clarity
   - Example: `get_technical_indicators(symbol, asset_type, interval)` → `mcp.tools.get_technical_indicators(symbol=symbol, asset_type=asset_type, timeframe=interval)`

## Files Modified

- `/mnt/c/Users/DefiSorce/Desktop/Maganalysis Project/Maganalysis Backend/aiagent.py`

## Affected Methods

The following methods in `CipherAgent` class were updated:
- `enhance_with_sentiment`
- `analyze_asset`
- `_fetch_real_time_price`
- `_analyze_timeframe`
- `_fetch_market_data`
- `_analyze_multi_timeframe_levels`
- `process_trend_following_strategy`
- `generate_market_strategy`
- `search_for_asset`

## Testing

The changes were tested to ensure:
1. The import structure is correct
2. The MCP tool calls follow the correct pattern with named parameters
3. All necessary functions are called through the MCP instance

These changes ensure that the code now properly uses the MCP tools through the MCP instance rather than trying to directly import them as Python module functions.