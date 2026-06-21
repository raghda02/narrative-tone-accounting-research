# ============================================================================
# chatgpt_classification.py
# Tone classification using ChatGPT API
# ============================================================================

import os
import json
import re
from typing import List, Dict

class ChatGPTClassifier:
    """
    Perform tone classification using ChatGPT API
    """
    
    def __init__(self, api_key=None, model="gpt-4"):
        """
        Initialize the classifier
        
        Parameters:
        -----------
        api_key : str
            OpenAI API key (can also be set via OPENAI_API_KEY environment variable)
        model : str
            OpenAI model to use (default: gpt-4)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model
        
        if not self.api_key:
            print("Warning: No API key provided. Using fallback classification.")
            self._fallback_mode = True
        else:
            self._fallback_mode = False
            try:
                import openai
                openai.api_key = self.api_key
                self.openai = openai
            except ImportError:
                print("OpenAI package not installed. Using fallback.")
                self._fallback_mode = True
    
    def classify_tone(self, text: str) -> Dict:
        """
        Classify tone of a single text
        
        Returns:
        --------
        Dict with keys: 'category', 'confidence', 'tone_score'
        """
        if self._fallback_mode:
            return self._fallback_classification(text)
        
        prompt = """
        You are a financial analyst specialized in corporate disclosures.
        Analyze the following MD&A text and classify its overall tone into one of three categories:
        - Positive (favorable outlook, confidence, growth expectations)
        - Neutral (balanced, factual, no clear sentiment)
        - Negative (caution, concern, pessimistic outlook)

        Provide only the category label and confidence score (0-100%).
        Text: {text}

        Response format: Category: [category], Confidence: [score]%
        """
        
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in analyzing corporate disclosure tone."},
                    {"role": "user", "content": prompt.format(text=text[:3000])}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            result = response.choices[0].message.content
            
            # Parse response
            category_match = re.search(r'Category:\s*(\w+)', result, re.IGNORECASE)
            confidence_match = re.search(r'Confidence:\s*([\d.]+)%', result)
            
            category = category_match.group(1).lower() if category_match else 'neutral'
            confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.5
            
            return {
                'category': category,
                'confidence': confidence,
                'tone_score': 1 if category == 'positive' else (-1 if category == 'negative' else 0)
            }
            
        except Exception as e:
            print(f"Error classifying tone: {e}")
            return self._fallback_classification(text)
    
    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """
        Classify tone for multiple texts
        """
        results = []
        for text in texts:
            results.append(self.classify_tone(text))
        return results
    
    def _fallback_classification(self, text: str) -> Dict:
        """
        Fallback when API is not available
        """
        # Simple keyword-based classification
        text_lower = text.lower()
        
        positive_words = ['strong', 'growth', 'increase', 'improve', 'optimistic', 'confident', 'record', 'success']
        negative_words = ['challenge', 'risk', 'uncertain', 'decline', 'difficulty', 'caution', 'pressure', 'concern']
        
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        if pos_count > neg_count:
            category = 'positive'
            confidence = 0.7
        elif neg_count > pos_count:
            category = 'negative'
            confidence = 0.7
        else:
            category = 'neutral'
            confidence = 0.6
        
        return {
            'category': category,
            'confidence': confidence,
            'tone_score': 1 if category == 'positive' else (-1 if category == 'negative' else 0)
        }

# Usage example
if __name__ == "__main__":
    # Test with sample text
    classifier = ChatGPTClassifier()
    result = classifier.classify_tone("Our company achieved record revenues this quarter")
    print(result)