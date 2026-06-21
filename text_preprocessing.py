# ============================================================================
# text_preprocessing.py
# Text cleaning and preprocessing for NLP analysis
# ============================================================================

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data (first time only)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Preprocess and clean MD&A text for NLP analysis
    """
    
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.custom_stopwords = {
            'shall', 'may', 'would', 'could', 'should',
            'hereby', 'thereby', 'herein', 'therein',
            'hereinafter', 'thereinafter', 'thereof'
        }
        self.stopwords.update(self.custom_stopwords)
    
    def clean_text(self, text):
        """
        Clean raw text
        """
        # Remove headers and footers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\f', '', text)
        
        # Remove tables
        text = re.sub(r'\|.*?\|', '', text)
        text = re.sub(r'\+.*?\+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline
        """
        # Step 1: Clean
        text = self.clean_text(text)
        
        # Step 2: Lowercase
        text = text.lower()
        
        # Step 3: Tokenize sentences
        sentences = sent_tokenize(text)
        
        # Step 4: Process each sentence
        processed = []
        for sentence in sentences:
            # Tokenize words
            words = word_tokenize(sentence)
            
            # Remove stopwords and short words
            words = [
                word for word in words 
                if word not in self.stopwords 
                and len(word) > 2
                and word.isalpha()
            ]
            
            if words:
                processed.append(" ".join(words))
        
        return processed
    
    def segment_text(self, text, segment_type='sentence'):
        """
        Segment text into units
        """
        if segment_type == 'sentence':
            return sent_tokenize(text)
        elif segment_type == 'paragraph':
            return text.split('\n\n')
        elif segment_type == 'document':
            return [text]
        else:
            raise ValueError("segment_type must be 'sentence', 'paragraph', or 'document'")