"""
Load stopwords from local files to avoid NLTK dependency
"""
import os
from pathlib import Path
from typing import Set

def load_stopwords(language: str = 'english') -> Set[str]:
    """
    Load stopwords from local file
    
    Args:
        language: Language name (default: 'english')
        
    Returns:
        Set of stopwords
    """
    current_dir = Path(__file__).parent
    stopwords_file = current_dir / "stopwords" / language
    
    if not stopwords_file.exists():
        raise FileNotFoundError(f"Stopwords file not found: {stopwords_file}")
    
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f if line.strip())
    
    return stopwords

# Load English stopwords once at module level
ENGLISH_STOPWORDS = load_stopwords('english')
