from enum import Enum

class MemoryType(Enum):
    MESSAGE = "message_history"        # Regular user messages
    RESPONSE = "message_response"      # AI responses
    ADMIN = "admin_memory"            # Admin messages
    DYNAMIC = "dynamic_knowledge"      # Knowledge learned from admin commands
    PLATFORM = "platform_context"      # Platform-specific settings
    TWEET = "tweet_history"           # Twitter post tracking 