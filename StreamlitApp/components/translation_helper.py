"""
Translation helper for translating English text to Vietnamese using the Qwen LLM.
"""

import streamlit as st
from typing import Dict, Any, Optional, List
import config


def get_translator():
    """Get or create translator LLM instance."""
    if "translator_llm" not in st.session_state:
        from pipelines.pipeline_v6 import QwenRemoteLLM
        st.session_state.translator_llm = QwenRemoteLLM()
    return st.session_state.translator_llm


@st.cache_data(show_spinner=False, ttl=3600)
def translate_text(text: str, target_lang: str = "vi") -> str:
    """
    Translate text to target language.
    Results are cached to avoid repeated API calls.
    
    Args:
        text: English text to translate
        target_lang: Target language code ("vi" for Vietnamese)
        
    Returns:
        Translated text
    """
    if not text or len(text.strip()) < 5:
        return text
    
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        
        llm = get_translator()
        
        messages = [
            SystemMessage(content="""You are a translator. Translate the following English text to Vietnamese.
Keep technical terms in English if they are commonly used that way in Vietnamese.
Only output the translation, nothing else. Keep the same formatting."""),
            HumanMessage(content=text)
        ]
        
        result = llm._generate(messages)
        if result and result.generations:
            return result.generations[0].message.content
        return text
    except Exception as e:
        return text  # Return original on error


def translate_chunk(chunk: Dict[str, Any], fields: List[str] = None) -> Dict[str, Any]:
    """
    Translate specific fields of a chunk dict to Vietnamese.
    
    Args:
        chunk: The chunk dictionary
        fields: List of fields to translate (default: ["text", "title"])
        
    Returns:
        Chunk with translated fields (prefixed with vi_)
    """
    if fields is None:
        fields = ["text", "title"]
    
    translated = chunk.copy()
    
    for field in fields:
        if field in chunk and chunk[field]:
            vi_field = f"vi_{field}"
            translated[vi_field] = translate_text(chunk[field])
    
    return translated


def translate_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate a graph fact to Vietnamese.
    
    Args:
        fact: Graph fact with source, relation, target
        
    Returns:
        Fact with translated fields
    """
    translated = fact.copy()
    
    # Translate the relation description
    if "relation" in fact:
        translated["vi_relation"] = translate_text(fact["relation"])
    
    return translated


def get_display_text(item: Dict, field: str, lang_mode: str) -> str:
    """
    Get the appropriate text based on language mode.
    
    Args:
        item: The item dict (chunk, fact, etc.)
        field: The field name to get
        lang_mode: "English", "Vietnamese", or "Both"
        
    Returns:
        Text in the appropriate language(s)
    """
    en_text = item.get(field, "")
    vi_field = f"vi_{field}"
    vi_text = item.get(vi_field)
    
    if lang_mode == "English":
        return en_text
    elif lang_mode == "Vietnamese":
        if vi_text:
            return vi_text
        # Translate on demand if not cached
        return translate_text(en_text) if en_text else ""
    else:  # Both
        if vi_text:
            return f"🇬🇧 {en_text}\n\n🇻🇳 {vi_text}"
        vi_translated = translate_text(en_text) if en_text else ""
        if vi_translated and vi_translated != en_text:
            return f"🇬🇧 {en_text}\n\n🇻🇳 {vi_translated}"
        return f"🇬🇧 {en_text}"
