import flair
from typing import List, Dict, Set, Tuple
from flair.models import SequenceTagger
from flair.data import Sentence
from flair import torch
import re, logging
from functools import lru_cache

# Setup logging
logging.getLogger("flair").setLevel(logging.CRITICAL)
logging.getLogger("Sentence").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.CRITICAL)
logging.getLogger("SequenceTagger").setLevel(logging.CRITICAL)

# Load the tagger model
tagger = SequenceTagger.load("../resources/taggers/example-universal-pos/best-model.pt")
flair.device = torch.device("cuda:0")

# Define constants for negation vector values
NON_NEG_VAL = 0  # Words outside scope of negation
NEG_CUE_VAL = 1  # Negation cue words
NEG_TOKEN_VAL = 2  # Words within scope of negation

# Define Indonesian negation cues
neg_cues: Set[str] = set([
    "tidak", "tak", "bukan", "takkan", "tiada", "jangan", "belum",
    "tanpa", "pantang", "non", "tdk", "ndak", "gak", "enggak",
    "ngak", "nggak", "ga", "gaada", "ngga", "gk", "kaga", "kagak",
    "nggk", "engga"
])

@lru_cache(maxsize=512)
def pos_tag_flair(text: str) -> Dict[str, List[str]]:
    """Use Flair to tag parts of speech, with caching for performance"""
    sentence = Sentence(text)
    tagger.predict(sentence)
    tokens = []
    tags = []
    result = {}
    for word in sentence:
        tokens.append(word.text)
        tags.append(word.tag)
        
    result['token'] = tokens
    result['tag'] = tags
    return result

def preprocess(text: str) -> str:
    """Clean and normalize text"""
    # Remove special characters except for punctuation we need for splitting
    cleaned_text = re.sub(r'[^\w\s,.!?]', '', text)
    # Normalize whitespace
    cleaned_text = " ".join(cleaned_text.split())
    
    return cleaned_text

def split_text(text: str) -> List[str]:
    """Split text into sentences or clauses for processing"""
    # Simple splitting on common conjunctions and punctuation
    split_pattern = re.compile(r'\b(dan|serta|tetapi|tapi|atau|sebab|karena|jika|kalau|seperti)\b|[?.,!]')
    texts = split_pattern.split(text)
    # Remove empty strings and single spaces
    texts = [t for t in texts if t and t != " "]
    return texts

def baseline_no_negation(text: str) -> Tuple[List[str], List[int]]:
    """
    Baseline 0: No Negation
    Simply returns tokens and a vector of all zeros
    """
    tokens = text.lower().split()
    negation_vector = [0] * len(tokens)
    return tokens, negation_vector

def baseline_rest_of_sentence(text: str) -> Tuple[List[str], List[int]]:
    """
    Baseline 1: Rest of Sentence
    Marks all tokens after a negation cue as being within scope
    """
    tokens = text.lower().split()
    negation_vector = [0] * len(tokens)
    
    for i, token in enumerate(tokens):
        if token in neg_cues:
            negation_vector[i] = NEG_CUE_VAL
            # Mark the rest of the sentence as within scope
            for j in range(i+1, len(tokens)):
                negation_vector[j] = NEG_TOKEN_VAL
    
    return tokens, negation_vector

def baseline_first_sentiment_word(text: str) -> Tuple[List[str], List[int]]:
    """
    Baseline 2: First Sentiment-carrying Word
    Marks only the first word after the negation cue that might carry sentiment
    
    Note: This is a simplified implementation since we don't have a sentiment lexicon.
    We'll assume nouns, verbs, and adjectives potentially carry sentiment.
    """
    tokens = text.lower().split()
    negation_vector = [0] * len(tokens)
    
    # Get POS tags
    pos_data = pos_tag_flair(text)
    
    # Look for negation cues
    for i, token in enumerate(tokens):
        if token in neg_cues:
            negation_vector[i] = NEG_CUE_VAL
            
            # Look for the first sentiment-carrying word (simplification)
            for j in range(i+1, len(tokens)):
                if j < len(pos_data['tag']) and pos_data['tag'][j] in ["NOUN", "VERB", "ADJ"]:
                    negation_vector[j] = NEG_TOKEN_VAL
                    break
    
    return tokens, negation_vector

def baseline_next_following_word(text: str) -> Tuple[List[str], List[int]]:
    """
    Baseline 3: Next Following Word
    Marks only the word immediately following the negation cue
    """
    tokens = text.lower().split()
    negation_vector = [0] * len(tokens)
    
    for i, token in enumerate(tokens):
        if token in neg_cues:
            negation_vector[i] = NEG_CUE_VAL
            # Mark the next word if it exists
            if i+1 < len(tokens):
                negation_vector[i+1] = NEG_TOKEN_VAL
    
    return tokens, negation_vector

def baseline_next_non_adverb(text: str) -> Tuple[List[str], List[int]]:
    """
    Baseline 4: Next Non-Adverb
    Skips adverbs and marks the first non-adverb after the negation cue
    """
    tokens = text.lower().split()
    negation_vector = [0] * len(tokens)
    
    # Get POS tags
    pos_data = pos_tag_flair(text)
    
    for i, token in enumerate(tokens):
        if token in neg_cues:
            negation_vector[i] = NEG_CUE_VAL
            
            # Find the first non-adverb word after negation cue
            for j in range(i+1, len(tokens)):
                if j < len(pos_data['tag']) and pos_data['tag'][j] != "ADV":
                    negation_vector[j] = NEG_TOKEN_VAL
                    break
    
    return tokens, negation_vector

def baseline_fixed_window(text: str, window_size: int = 2) -> Tuple[List[str], List[int]]:
    """
    Baseline 5: Fixed Window Length
    Marks a fixed number of tokens after the negation cue
    """
    tokens = text.lower().split()
    negation_vector = [0] * len(tokens)
    
    for i, token in enumerate(tokens):
        if token in neg_cues:
            negation_vector[i] = NEG_CUE_VAL
            
            # Mark the next N words based on window size
            for j in range(i+1, min(i+1+window_size, len(tokens))):
                negation_vector[j] = NEG_TOKEN_VAL
    
    return tokens, negation_vector

def negation_handling(kalimat: str, baseline_method: int = 0) -> Tuple[List[str], List[int]]:
    """
    Main function to handle negation in Indonesian text using the specified baseline method
    Returns a tuple of (tokens, negation_vector)
    
    Methods:
    0: No Negation
    1: Rest of Sentence
    2: First Sentiment-carrying Word
    3: Next Following Word
    4: Next Non-Adverb
    5: Fixed Window Length (window = 2)
    """
    # Preprocess the text
    text = preprocess(kalimat)
    
    # Apply the selected baseline method
    if baseline_method == 1:
        return baseline_rest_of_sentence(text)
    elif baseline_method == 2:
        return baseline_first_sentiment_word(text)
    elif baseline_method == 3:
        return baseline_next_following_word(text)
    elif baseline_method == 4:
        return baseline_next_non_adverb(text)
    elif baseline_method == 5:
        return baseline_fixed_window(text, window_size=2)
    else:
        # Default to No Negation method
        return baseline_no_negation(text)

def custom_split(text: str) -> List[str]:
    """Split text into tokens"""
    text = re.sub(r'(\d+)/(\d+)', r'\1\2', text)
    text = re.sub(r'[.,!?:;(){}\[\]"\']+', ' ', text)
    words = re.split(r'\s+', text.strip())
    words = [word for word in words if word]

    # Special case handling for "HUD"
    for i, word in enumerate(words):
        if word.upper() == "HUD":
            words[i] = word.lower()

    return words