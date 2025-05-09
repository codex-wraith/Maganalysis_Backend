# Fix for ImportError in aiagent.py

## Problems

### Problem 1: ImportError for MCP Tools

The application was encountering an `ImportError` when trying to import MCP tools directly from their module files:

```
ImportError: cannot import name 'get_technical_indicators' from 'market.mcp_tools' (/app/market/mcp_tools.py)
```

This occurred because these functions were registered as MCP tools *inside* registration functions and were not available as top-level module imports.

### Problem 2: AttributeError for MCP Tools Access

After implementing the initial fix by switching to `mcp.tools.<tool_name>()` calls, we encountered a different error:

```
AttributeError: 'FastMCP' object has no attribute 'tools'
```

This occurred because the MCP instance in the app doesn't support the `tools` attribute pattern or isn't properly initialized for direct tool calls (`app.state.mcp = None` in main.py).

### Problem 3: Circular Import Issues

After implementing our solution with module-level functions, we encountered circular import errors:

```
ImportError: cannot import name 'app' from partially initialized module 'main' (most likely due to a circular import) (/app/main.py)
```

This occurred because of interdependencies between main.py, mcp_server.py, and the MCP tool modules.

### Problem 4: Incorrect Method Parameters

After deploying the code, we encountered errors related to method parameters:

```
TypeError: MarketManager.get_intraday_data() got an unexpected keyword argument 'http_session'
```

This occurred because the `get_intraday_data` method in the `MarketManager` class doesn't accept an `http_session` parameter, unlike other methods in the same class.

### Problem 5: Unexpected Function Arguments

We also encountered another parameter-related error:

```
TypeError: PriceLevelAnalyzer.identify_support_levels() got an unexpected keyword argument 'include_fibonacci'
```

This occurred because we were passing `include_fibonacci` and `include_psychological` parameters to the `identify_support_levels` and `identify_resistance_levels` methods, but these methods don't accept these parameters.

## Solution

After several iterations, our final solution involved:

1. **Moving function definitions to module level without decorators**: 
   - Moved function implementations from inside registration functions to module level
   - Removed `@mcp.tool()` decorators from the module level to avoid import cycles
   - Added direct imports for dependencies between module files
   - Registered functions with MCP inside registration functions

2. **Creating a global app access mechanism**:
   - Added `get_app()` function to main.py
   - Made app instance accessible globally for module-level functions
   - Added global dependencies (market_manager, http_session) in each module

3. **Using direct function calls in aiagent.py**:
   - Imported functions directly from their module files
   - Changed all calls to use direct function calls
   - Added proper parameter names for clarity

4. **Resolving circular import issues**:
   - Removed direct imports of `mcp`, `app` and `get_app` from module files
   - Added dynamic imports inside functions where needed
   - Used import aliases (`import app as main_app`) to avoid conflicts
   - Replaced global references with function-scoped imports
   - Moved all MCP-related imports to registration functions

5. **Fixing method parameter inconsistencies**:
   - Removed `http_session` parameter from calls to `get_intraday_data`
   - Kept `http_session` parameter for methods that explicitly accept it
   - Relied on the internal HTTP session of MarketManager where needed

6. **Removing unsupported parameters**:
   - Removed `include_fibonacci` and `include_psychological` parameters from calls to `identify_support_levels` and `identify_resistance_levels`
   - Ensured method calls match parameter signatures in the PriceLevelAnalyzer class
   - Fixed all higher timeframe analysis code to use the correct method signatures

## Files Modified

- `/mnt/c/Users/DefiSorce/Desktop/Maganalysis Project/Maganalysis Backend/aiagent.py`
- `/mnt/c/Users/DefiSorce/Desktop/Maganalysis Project/Maganalysis Backend/main.py`
- `/mnt/c/Users/DefiSorce/Desktop/Maganalysis Project/Maganalysis Backend/market/mcp_tools.py`
- `/mnt/c/Users/DefiSorce/Desktop/Maganalysis Project/Maganalysis Backend/market/mcp_price_levels.py`
- `/mnt/c/Users/DefiSorce/Desktop/Maganalysis Project/Maganalysis Backend/market/mcp_multi_timeframe.py`

## Key Changes

### 1. In module files (mcp_tools.py, mcp_price_levels.py, mcp_multi_timeframe.py):

- Moved function implementations outside registration functions to module level
- Added global variables for dependencies (market_manager, http_session)
- Added init_global_dependencies() function to initialize globals
- Updated registration functions to just store dependencies instead of defining functions
- Modified functions to work with direct calling pattern

### 2. In main.py:

- Added global variable to track app instance (`_app_instance`)
- Added get_app() function to provide access to app instance
- Modified lifespan function to set global app instance

### 3. In aiagent.py:

- Updated imports to import functions directly from module files
- Modified all function calls to use direct imports instead of mcp.tools pattern
- Updated function parameter names for clarity (often needed for analyze_multi_timeframe)

## Implementation Details

The strategy we implemented ensures that:

1. MCP tools are still properly registered with MCP inside registration functions
2. Functions are available for direct import and calling as regular Python functions without MCP dependencies
3. App state dependencies are accessible to module-level functions
4. Code maintains its original functionality but with a different invocation pattern
5. Registration functions now handle both dependency storage and MCP registration
6. Circular import issues are resolved by moving all MCP-related imports to registration functions

This approach provides the best of both worlds: MCP registration for external tool access, and direct function calling for internal use, while completely avoiding import cycles.

## Testing

The code should now be able to run without the ImportError or AttributeError previously encountered. All tool functionality should work as before but with the improved calling pattern.