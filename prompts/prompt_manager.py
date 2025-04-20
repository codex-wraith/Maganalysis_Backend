import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PromptManager:
    def __init__(self):
        """Initialize the prompt manager by loading prompt templates from the JSON file."""
        self._prompts = self._load_prompt_templates()
        
    def _load_prompt_templates(self) -> Dict:
        """Load prompt templates from the JSON file."""
        try:
            with open('prompts/prompts.json', 'r') as f:
                templates = json.load(f)
            return templates
        except Exception as e:
            logger.error(f"Failed to load prompt templates: {e}")
            raise
    
    def get_system_prompt(self, context: str, **kwargs) -> str:
        """
        Get the system prompt for the specified context.
        
        Args:
            context: The context for which to get the system prompt. 
                     Can be "base", "telegram_private", "telegram_group", or "web".
            **kwargs: Parameters to format the prompt template.
            
        Returns:
            The formatted system prompt.
        """
        if context not in self._prompts['system_prompt']:
            raise ValueError(f"Invalid context: {context}. Must be one of: {', '.join(self._prompts['system_prompt'].keys())}")
        
        return self._prompts['system_prompt'][context].format(**kwargs)
    
    def get_admin_context(self, **kwargs) -> str:
        """
        Get the admin context prompt.
        
        Args:
            **kwargs: Parameters to format the prompt template.
            
        Returns:
            The formatted admin context prompt.
        """
        return self._prompts['admin_context']['relationship'].format(**kwargs)
    
    def get_market_context(self, **kwargs) -> str:
        """
        Get the market context prompt.
        
        Args:
            **kwargs: Parameters to format the prompt template.
            
        Returns:
            The formatted market context prompt.
        """
        return self._prompts['market_context']['template'].format(**kwargs)
    
    def get_conversation_break(self, break_type: str) -> str:
        """
        Get the conversation break prompt.
        
        Args:
            break_type: The type of conversation break. Can be "new_session" or "message_separator".
            
        Returns:
            The conversation break prompt.
        """
        if break_type not in self._prompts['conversation_breaks']:
            raise ValueError(f"Invalid break type: {break_type}. Must be one of: {', '.join(self._prompts['conversation_breaks'].keys())}")
        
        return self._prompts['conversation_breaks'][break_type]
    
    def get_intraday_formatting(self, is_crypto: bool = False, **kwargs) -> Dict[str, str]:
        """
        Get the intraday formatting templates.
        
        Args:
            is_crypto: Whether to use crypto-specific formatting.
            **kwargs: Parameters to format the prompt templates.
            
        Returns:
            A dictionary containing the formatted intraday formatting templates.
        """
        section = 'crypto_intraday_formatting' if is_crypto else 'intraday_formatting'
        
        formatted = {}
        for key, template in self._prompts[section].items():
            formatted[key] = template.format(**kwargs)
            
        return formatted
    
    def get_market_analysis_prompt(self, **kwargs) -> str:
        """
        Get the market analysis prompt optimized for production use.

        This version includes only the essential sections:
          - The header section.
          - The sentiment section if the necessary sentiment data is provided.
          - The recommendation section.

        Args:
            **kwargs: Parameters to format the prompt template sections. These should include the dynamic values
                      (such as SYMBOL, PRICE, CHANGE_PCT, etc.) required by the template.

        Returns:
            The fully formatted market analysis prompt.
        """
        template = self._prompts['market_analysis_template']
        
        # Start with the header.
        prompt = template['header']
        
        # Add the sentiment section if both sentiment label and article highlight are provided.
        if 'SENTIMENT_LABEL' in kwargs and 'ARTICLE_HIGHLIGHT' in kwargs:
            prompt += template['sentiment_section']
        
        # Add the recommendation section.
        prompt += template['recommendation_section']
        
        # Return the fully formatted prompt.
        return prompt.format(**kwargs)
    
    def get_top_movers_formatting(self, **kwargs) -> Dict[str, str]:
        """
        Get the top movers formatting templates.
        
        Args:
            **kwargs: Parameters to format the prompt templates.
            
        Returns:
            A dictionary containing the formatted top movers formatting templates.
        """
        formatted = {}
        for key, template in self._prompts['top_movers_formatting'].items():
            if isinstance(template, str):  # Only format strings, not lists or other structures
                formatted[key] = template.format(**kwargs) if kwargs else template
            else:
                formatted[key] = template
                
        return formatted
        
    def get_template_section(self, section_name: str, default: Any = None) -> Any:
        """
        Get a specific section from the prompt templates.
        
        Args:
            section_name: The name of the section to get.
            default: The default value to return if the section doesn't exist.
            
        Returns:
            The requested section or the default value if not found.
        """
        return self._prompts.get(section_name, default)