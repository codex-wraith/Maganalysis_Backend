import json
from typing import List, Any, Dict, Optional
import logging
from memory.base import BaseMemoryManager
from memory.memory_types import MemoryType
from aisettings import AISettings
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

class MessageMemoryManager(BaseMemoryManager):
    def __init__(self, settings: AISettings, character_config: Dict = None):
        super().__init__(settings)
        self.admin_id = (
            character_config['settings']['admin']['telegram_admin_id'] 
            if character_config else None
        )
        self._short_term_cache = {}
        
        logger.info("MessageMemoryManager initialized with Redis and short-term cache")

    def _is_admin(self, user_id: str) -> bool:
        """Check if user is admin."""
        return user_id == self.admin_id

    def _get_timestamp(self) -> float:
        """Get standardized UTC timestamp."""
        return datetime.now(UTC).timestamp()

    async def add(self, data: Any, memory_type: MemoryType = MemoryType.MESSAGE) -> None:
        """Add data to Redis memory with a typed key."""
        try:
            timestamp = self._get_timestamp()
            platform = data.get('platform', 'web')
            user_id = data.get('user_id', 'anonymous')
            
            is_admin = self._is_admin(user_id)
            
            # Handle different memory types
            if memory_type == MemoryType.TWEET:
                # Special handling for tweets - simpler key structure
                key = f"{memory_type.value}:{timestamp}"
            else:
                # Admin messages go to ADMIN type, regular users to MESSAGE type
                memory_type = MemoryType.ADMIN if is_admin else MemoryType.MESSAGE
                key = f"{memory_type.value}:{platform}:{user_id}:{timestamp}"
            
            # Remove non-serializable objects from context
            sanitized_data = data.copy()
            if 'context' in sanitized_data:
                # Create a safe copy of context without non-serializable objects like 'bot'
                if isinstance(sanitized_data['context'], dict):
                    safe_context = {}
                    for k, v in sanitized_data['context'].items():
                        # Skip the bot object which is not JSON serializable
                        if k != 'bot':
                            safe_context[k] = v
                    sanitized_data['context'] = safe_context
            
            value = json.dumps({
                "data": sanitized_data,
                "metadata": {
                    "platform": platform,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "type": memory_type.value,
                    "is_admin": is_admin,
                    "is_response": False
                }
            })
            
            await self.redis.set(key, value)
            logger.info(f"Stored message memory: {key}")
            
        except Exception as e:
            logger.error(f"Error storing message memory: {e}")

    async def get(
        self, 
        memory_type: MemoryType, 
        limit: int = None, 
        user_id: str = None, 
        platform: str = None,
        filter_responses: bool = False
    ) -> List[Dict]:
        """Get messages from memory with optional filtering"""
        try:
            pattern = f"{memory_type.value}:{platform}:{user_id}:*"
            keys = await self.redis.keys(pattern)
            messages = []
            
            for key in keys:
                value = await self.redis.get(key)
                if value:
                    msg = json.loads(value)
                    if filter_responses and msg['metadata'].get('is_response'):
                        continue
                    messages.append(msg['data'])
            
            # Sort and limit
            if messages:
                messages.sort(key=lambda x: x.get('timestamp', 0))
                if limit:
                    return messages[-limit:]
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []

    async def get_relevant_history(
        self, 
        platform: str = None, 
        user_id: str = None
    ) -> List[Any]:
        """
        Return messages relevant to query.
        For admin: returns all ADMIN type memories (includes messages and responses)
        For users: returns their MESSAGE type memories and their RESPONSE type memories
        """
        try:
            messages = []
            is_admin = self._is_admin(user_id)
            
            if is_admin:
                # For admin, get all admin memories with platform filter
                pattern = f"{MemoryType.ADMIN.value}:{platform}:*" if platform else f"{MemoryType.ADMIN.value}:*"
                keys = await self.redis.keys(pattern)
            else:
                # Get regular user messages - explicitly use MESSAGE type
                msg_pattern = f"{MemoryType.MESSAGE.value}:{platform}:{user_id}:*"
                resp_pattern = f"{MemoryType.RESPONSE.value}:{platform}:{user_id}:*"
                
                keys = await self.redis.keys(msg_pattern)
                keys.extend(await self.redis.keys(resp_pattern))
            
            for key in keys:
                value = await self.redis.get(key)
                if value:
                    memory = json.loads(value)
                    messages.append(memory)
            
            # Sort by timestamp (descending)
            messages.sort(
                key=lambda x: x["metadata"]["timestamp"],
                reverse=True
            )
            
            return [m["data"] for m in messages]
            
        except Exception as e:
            logger.error(f"Error getting relevant history: {e}")
            return []

    async def cleanup_expired(self):
        """Remove expired entries from Redis and clean short-term cache."""
        try:
            current_time = self._get_timestamp()
            cleaned_count = 0
            
            for memory_type in MemoryType:
                # Skip PLATFORM type and TWEET type as they don't auto-expire
                if memory_type in [MemoryType.PLATFORM, MemoryType.TWEET]:
                    continue
                    
                pattern = f"{memory_type.value}:*"
                keys = await self.redis.keys(pattern)
                
                for key in keys:
                    try:
                        value = await self.redis.get(key)
                        if value:
                            data = json.loads(value)
                            metadata = data.get('metadata', {})
                            if not metadata and isinstance(data.get('data', {}), dict):
                                metadata = data['data']
                            
                            stored_time = metadata.get('timestamp', 0)
                            if isinstance(stored_time, str):
                                try:
                                    stored_time = datetime.fromisoformat(
                                        stored_time.replace('Z', '+00:00')
                                    ).timestamp()
                                except:
                                    stored_time = 0
                            
                            age = current_time - float(stored_time)
                            ttl = (
                                self.settings.ADMIN_MEMORY_TTL 
                                if metadata.get('is_admin') 
                                else self.settings.USER_MEMORY_TTL
                            )
                            
                            if ttl and age > ttl:
                                await self.redis.delete(key)
                                cleaned_count += 1
                                logger.info(f"Cleaned up expired key: {key} (age: {age:.2f}s)")
                    except Exception as e:
                        logger.warning(f"Error processing key {key}: {e}")
                        continue
                    
            # Clean short-term cache
            for cache_key in list(self._short_term_cache.keys()):
                # Only need user_id for admin check
                user_id = cache_key.split(':')[1]
                is_admin = self._is_admin(user_id)
                
                ttl = 0 if is_admin else self.settings.USER_MEMORY_TTL
                
                if ttl:
                    self._short_term_cache[cache_key] = [
                        msg for msg in self._short_term_cache[cache_key]
                        if (current_time - self._get_msg_timestamp(msg)) < ttl
                    ]
                    
                    if not self._short_term_cache[cache_key]:
                        del self._short_term_cache[cache_key]
            
            # Clean admin memories
            admin_pattern = "admin_memory:*"
            admin_keys = await self.redis.keys(admin_pattern)
            for key in admin_keys:
                value = await self.redis.get(key)
                if value:
                    data = json.loads(value)
                    stored_time = data['metadata'].get('timestamp', 0)
                    if isinstance(stored_time, str):
                        try:
                            stored_time = datetime.fromisoformat(
                                stored_time.replace('Z', '+00:00')
                            ).timestamp()
                        except:
                            stored_time = 0
                    
                    age = current_time - float(stored_time)
                    if self.settings.ADMIN_MEMORY_TTL and age > self.settings.ADMIN_MEMORY_TTL:
                        await self.redis.delete(key)
            
            logger.info(f"Memory cleanup completed. Cleaned {cleaned_count} expired entries")
                        
        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")

    def _get_msg_timestamp(self, msg: Dict) -> float:
        """Helper to get timestamp from a message as float."""
        ts = msg.get('timestamp')
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except:
                return 0
        return float(ts) if ts else 0

    async def get_group_context(self, chat_id: str) -> Dict:
        """Get context for a specific Telegram group."""
        try:
            key = f"{MemoryType.PLATFORM.value}:telegram:group:{chat_id}"
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return {}
        except Exception as e:
            logger.error(f"Error getting group context: {e}")
            return {}

    async def set_group_context(self, chat_id: str, context: Dict) -> None:
        """Set context for a specific Telegram group."""
        try:
            key = f"{MemoryType.PLATFORM.value}:telegram:group:{chat_id}"
            await self.redis.set(key, json.dumps(context))
        except Exception as e:
            logger.error(f"Error setting group context: {e}")

    async def add_to_short_term(
        self, 
        platform: str, 
        user_id: str, 
        message: Dict
    ) -> None:
        """
        Add message to short-term cache.
        """
        try:
            cache_key = f"{platform}:{user_id}"
            if cache_key not in self._short_term_cache:
                self._short_term_cache[cache_key] = []
            
            # Make a safe copy of the message to prevent non-serializable objects
            safe_message = message.copy()
            
            # Handle context that might contain non-serializable objects
            if 'context' in safe_message and isinstance(safe_message['context'], dict):
                # Filter out non-serializable objects like 'bot' from context
                safe_context = {}
                for k, v in safe_message['context'].items():
                    if k != 'bot':  # Skip the Telegram bot object
                        safe_context[k] = v
                safe_message['context'] = safe_context
            
            self._short_term_cache[cache_key].append(safe_message)
            logger.debug(f"Added message to short-term cache for {cache_key}")
        except Exception as e:
            logger.error(f"Error adding to short-term cache: {e}")

    async def get_short_term(self, platform: str, user_id: str) -> List[Dict]:
        """Get short-term cache for specific platform/user."""
        try:
            cache_key = f"{platform}:{user_id}"
            return self._short_term_cache.get(cache_key, [])
        except Exception as e:
            logger.error(f"Error getting short-term cache: {e}")
            return []

    async def update_short_term_response(
        self, 
        platform: str, 
        user_id: str, 
        response: str
    ) -> None:
        """Update the last message in short-term cache with the LLM's response."""
        try:
            cache_key = f"{platform}:{user_id}"
            if cache_key in self._short_term_cache and self._short_term_cache[cache_key]:
                self._short_term_cache[cache_key][-1]['response'] = response
                logger.debug(f"Updated short-term cache response for {cache_key}")
        except Exception as e:
            logger.error(f"Error updating short-term cache response: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Return basic stats about memory usage."""
        try:
            stats = {
                'short_term_cache_size': sum(len(msgs) for msgs in self._short_term_cache.values()),
                'redis_keys': len(await self.redis.keys(f"{MemoryType.MESSAGE.value}:*")),
                'admin_memories': len(await self.redis.keys(f"{MemoryType.ADMIN.value}:*")),
                'platform_contexts': len(await self.redis.keys(f"{MemoryType.PLATFORM.value}:*"))
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

    async def add_response(self, platform: str, user_id: str, response: str) -> None:
        """
        Store the assistant's response in both short-term and long-term memory.
        Responses to admin are stored with ADMIN type, others with RESPONSE type.
        """
        try:
            # Update short-term cache
            await self.update_short_term_response(platform, user_id, response)
            
            # Store in Redis
            timestamp = self._get_timestamp()
            is_admin = self._is_admin(user_id)
            
            # Admin responses go to ADMIN type, regular users to RESPONSE type
            memory_type = MemoryType.ADMIN if is_admin else MemoryType.RESPONSE
            key = f"{memory_type.value}:{platform}:{user_id}:{timestamp}"
            
            value = json.dumps({
                "data": {
                    "text": response,
                    "platform": platform,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "is_response": True
                },
                "metadata": {
                    "type": memory_type.value,
                    "platform": platform,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "is_response": True
                }
            })
            
            await self.redis.set(key, value)
            logger.info(f"Stored response in Redis: {key}")
            
        except Exception as e:
            logger.error(f"Error storing response: {e}")

    async def get_conversation_history(
        self, 
        platform: str, 
        user_id: str, 
        limit: int = None
    ) -> List[Dict]:
        """
        Get deduplicated conversation history from both short-term and long-term memory.
        Includes both user messages and AI responses for a complete conversation.
        """
        try:
            is_admin = self._is_admin(user_id)
            all_messages = []
            
            if is_admin:
                # For admin users, get everything from ADMIN memory
                # (ADMIN memory contains both admin messages and AI responses)
                admin_pattern = f"{MemoryType.ADMIN.value}:{platform}:{user_id}:*"
                keys = await self.redis.keys(admin_pattern)
                
                for key in keys:
                    value = await self.redis.get(key)
                    if value:
                        msg = json.loads(value)
                        all_messages.append(msg['data'])
            else:
                # For regular users, get:
                # 1. User messages from MESSAGE memory
                msg_pattern = f"{MemoryType.MESSAGE.value}:{platform}:{user_id}:*"
                msg_keys = await self.redis.keys(msg_pattern)
                
                for key in msg_keys:
                    value = await self.redis.get(key)
                    if value:
                        msg = json.loads(value)
                        all_messages.append(msg['data'])
                
                # 2. AI responses from RESPONSE memory
                resp_pattern = f"{MemoryType.RESPONSE.value}:{platform}:{user_id}:*"
                resp_keys = await self.redis.keys(resp_pattern)
                
                for key in resp_keys:
                    value = await self.redis.get(key)
                    if value:
                        msg = json.loads(value)
                        all_messages.append(msg['data'])
            
            # Add short-term cache for most recent messages
            short_term = await self.get_short_term(platform, user_id)
            all_messages.extend(short_term)
            
            # Deduplicate while preserving chronological order
            seen = set()
            unique_messages = []
            
            def get_timestamp_safe(msg: Dict) -> float:
                ts = msg.get('timestamp')
                if isinstance(ts, str):
                    try:
                        return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                    except:
                        return 0
                return float(ts) if ts else 0
            
            # Sort everything by timestamp
            all_messages.sort(key=lambda x: get_timestamp_safe(x))

            for msg in all_messages:
                # Create a unique ID that includes whether this is a response
                is_response = msg.get('is_response', False)
                msg_id = f"{is_response}:{msg.get('text', '')}:{get_timestamp_safe(msg)}"
                if msg_id not in seen:
                    seen.add(msg_id)
                    unique_messages.append(msg)
            
            if limit:
                return unique_messages[-limit:]  # Return the last N messages (most recent)
            else:
                return unique_messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    async def get_last_response(self, user_id: str, platform: str) -> Optional[str]:
        """Get the most recent AI response for this user"""
        try:
            # Check if admin and use appropriate memory type
            is_admin = self._is_admin(user_id)
            memory_type = MemoryType.ADMIN if is_admin else MemoryType.RESPONSE
            pattern = f"{memory_type.value}:{platform}:{user_id}:*"
            
            keys = await self.redis.keys(pattern)
            
            if not keys:
                return None
                
            responses = []
            for key in keys:
                value = await self.redis.get(key)
                if value:
                    msg = json.loads(value)
                    # For ADMIN memory, we need to filter to only include responses
                    if memory_type == MemoryType.ADMIN:
                        if msg['metadata'].get('is_response', False):
                            responses.append(msg)
                    else:
                        # For RESPONSE memory, all entries are responses
                        responses.append(msg)
            
            if responses:
                responses.sort(key=lambda x: x['metadata']['timestamp'])
                return responses[-1]['data']['text']
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting last response: {e}")
            return None