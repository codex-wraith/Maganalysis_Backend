from typing import Optional
from pydantic_settings import BaseSettings

class AISettings(BaseSettings):
    # Model Configuration
    ANTHROPIC_API_KEY: str
    ALPHAVANTAGE_API_KEY: str  # For stock market analysis using Alpha Vantage API
    CHAT_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    SOCIAL_MODEL: str = "claude-3-7-sonnet-20250219"
    MAX_TOKENS: int = 64000
    TEMPERATURE: float = 0

    # Update Intervals
    HEALTH_CHECK_INTERVAL: int = 300  # Health check every 5 minutes

    # Memory Settings
    ADMIN_MEMORY_TTL: Optional[int] = 0  # 0 means no expiration
    USER_MEMORY_TTL: int = 86400         # Regular user memory TTL (24 hours)
    REDIS_TTL: int = 86400               # Redis key expiration (24 hours)
    SHORT_TERM_TTL: int = 3600           # Short-term cache expiration (1 hour)

    # Redis Memory Configuration
    REDIS_HOST: str = "redis-13089.c82.us-east-1-2.ec2.redns.redis-cloud.com"
    REDIS_PORT: int = 13089
    REDIS_USERNAME: str = "default"
    REDIS_PASSWORD: str = "pFevn8lt1oTKqybjYsNjdWwTgIgGisk7"

    RETRY_DELAY: int = 300

    # Market Settings (Crypto)
    MARKET_CAP_UPDATE_INTERVAL: int = 43200 
    TOTAL_SUPPLY: int = 1_000_000_000

    # Optional Stock Market Settings
    STOCK_MARKET_UPDATE_INTERVAL: int = 300  # in seconds

    # New: Toggle to restrict telegram usage to the admin only.
    TELEGRAM_RESTRICT_TO_ADMIN: bool = True

    # New: Toggle to enable/disable Twitter posting.
    TWITTER_ENABLED: bool = False