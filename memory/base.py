from abc import ABC, abstractmethod
from typing import List, Dict, Any
import redis.asyncio as redis
from aisettings import AISettings

class BaseMemoryManager(ABC):
    def __init__(self, settings: AISettings):
        self.settings = settings
        # Use the async Redis client
        self.redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            username=settings.REDIS_USERNAME,
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )
    
    @abstractmethod
    async def add(self, data: Any) -> None:
        pass
    
    @abstractmethod
    async def get(self, query: str, limit: int = 5) -> List[Any]:
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        pass