import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List, Union
import re

logger = logging.getLogger(__name__)

class MessageProcessor:
    """
    A utility class for managing LLM response streaming, progress updates, rate limiting,
    and message splitting.
    """
    
    def __init__(self, max_chunk_size: int = 4096, rate_limit_delay: float = 0.5):
        """
        Initialize the message processor
        
        Args:
            max_chunk_size: Maximum size of each message chunk
            rate_limit_delay: Delay between chunks for rate limiting (seconds)
        """
        self.max_chunk_size = max_chunk_size
        self.rate_limit_delay = rate_limit_delay
        
    @staticmethod
    def _clean_message(message: str) -> str:
        """Remove any unwanted characters or formatting from a message"""
        # Remove any control characters except newlines and tabs
        message = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', '', message)
        
        # Collapse multiple newlines into a maximum of two
        message = re.sub(r'\n{3,}', '\n\n', message)
        
        # Trim leading/trailing whitespace
        return message.strip()
        
    def split_message(self, message: str) -> List[str]:
        """
        Split a long message into smaller chunks
        
        Args:
            message: The message to split
            
        Returns:
            List of message chunks
        """
        # Clean the message first
        message = self._clean_message(message)
        
        # If message is already small enough, return it as a single chunk
        if len(message) <= self.max_chunk_size:
            return [message]
            
        chunks = []
        
        # Try to split on paragraph breaks first
        paragraphs = re.split(r'\n\s*\n', message)
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the limit, start a new chunk
            if len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size:
                # If the paragraph itself is too long, split it further
                if len(paragraph) > self.max_chunk_size:
                    # Add the current chunk if not empty
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    
                    # Split the long paragraph by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    sentence_chunk = ""
                    
                    for sentence in sentences:
                        if len(sentence_chunk) + len(sentence) + 1 > self.max_chunk_size:
                            # If the sentence is too long on its own, split it further
                            if len(sentence) > self.max_chunk_size:
                                # Add the current sentence chunk if not empty
                                if sentence_chunk:
                                    chunks.append(sentence_chunk.strip())
                                    sentence_chunk = ""
                                
                                # Split the sentence into chunks
                                while sentence:
                                    chunks.append(sentence[:self.max_chunk_size])
                                    sentence = sentence[self.max_chunk_size:]
                            else:
                                chunks.append(sentence_chunk.strip())
                                sentence_chunk = sentence
                        else:
                            if sentence_chunk:
                                sentence_chunk += " " + sentence
                            else:
                                sentence_chunk = sentence
                    
                    # Add the final sentence chunk if not empty
                    if sentence_chunk:
                        chunks.append(sentence_chunk.strip())
                else:
                    # Add the current chunk and start a new one with this paragraph
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
            else:
                # Add the paragraph to the current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
        
    async def stream_message(
        self, 
        message: str, 
        send_func: Callable[[str, Dict[str, Any]], None],
        extra_args: Dict[str, Any] = None,
        progress_interval: int = 25,
        with_progress_updates: bool = True
    ) -> None:
        """
        Stream a message with progress updates and rate limiting
        
        Args:
            message: The message to stream
            send_func: Function to call for sending each chunk
            extra_args: Additional arguments to pass to the send function
            progress_interval: Percentage interval for progress updates
            with_progress_updates: Whether to include progress updates
        """
        if not message:
            return
            
        # Clean and split the message
        chunks = self.split_message(message)
        total_chunks = len(chunks)
        
        # Nothing to send
        if total_chunks == 0:
            return
            
        # Create default extra_args if none provided
        if extra_args is None:
            extra_args = {}
            
        # Calculate percentage increment for progress updates
        pct_increment = progress_interval
        next_pct = pct_increment
        
        # For single chunk messages, just send it directly
        if total_chunks == 1:
            await send_func(chunks[0], extra_args)
            return
            
        for i, chunk in enumerate(chunks):
            # Calculate progress percentage
            progress_pct = int((i / total_chunks) * 100)
            
            # Add progress indicator if enabled
            if with_progress_updates and progress_pct >= next_pct:
                marker = f"[{progress_pct}%]"
                if not chunk.startswith(marker):
                    chunk = f"{marker} {chunk}"
                next_pct = progress_pct + pct_increment
                
            # Add continuation marker for all chunks except the first
            if i > 0 and not chunk.startswith("[cont") and with_progress_updates:
                chunk = f"[continued] {chunk}"
                
            # Send the chunk
            await send_func(chunk, extra_args)
            
            # Rate limiting delay between chunks
            if i < total_chunks - 1:
                await asyncio.sleep(self.rate_limit_delay)
                
    async def stream_with_updates(
        self,
        content_generator: Callable[[], str],
        update_interval: float,
        send_func: Callable[[str, Dict[str, Any]], None],
        extra_args: Dict[str, Any] = None,
        max_updates: int = 5
    ) -> None:
        """
        Stream updates while waiting for content generation
        
        Args:
            content_generator: Function that generates the content
            update_interval: Time between updates in seconds
            send_func: Function to call for sending updates
            extra_args: Additional arguments to pass to the send function
            max_updates: Maximum number of intermediate updates
        """
        if extra_args is None:
            extra_args = {}
            
        # Send initial progress message
        send_func("⏳ Processing your request...", extra_args)
        
        # Create a task for generating content
        content_task = asyncio.create_task(self._run_generator(content_generator))
        
        # Track updates
        updates_sent = 1
        update_messages = [
            "⏳ Analyzing data...",
            "⏳ Gathering information...",
            "⏳ Almost ready...",
            "⏳ Finishing up...",
            "⏳ Just a moment longer..."
        ]
        
        # Send updates until content is ready
        while not content_task.done() and updates_sent <= max_updates:
            await asyncio.sleep(update_interval)
            
            if not content_task.done():
                update_idx = min(updates_sent, len(update_messages) - 1)
                send_func(update_messages[update_idx], extra_args)
                updates_sent += 1
                
        # Get the content once ready
        content = await content_task
        
        # Stream the actual content
        if content:
            await self.stream_message(content, send_func, extra_args)
            
    async def _run_generator(self, content_generator: Callable[[], str]) -> str:
        """Run a content generator function asynchronously"""
        if asyncio.iscoroutinefunction(content_generator):
            return await content_generator()
        else:
            # Run synchronous function in a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, content_generator)