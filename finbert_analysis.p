# ============================================================================
# finbert_analysis.py
# Sentiment analysis using FinBERT transformer model
# ============================================================================

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

class FinBERTAnalyzer:
    """
    Perform sentiment analysis using FinBERT model
    """
    
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer
        )
        self.labels = ['negative', 'neutral', 'positive']
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a single text
        """
        result = self.classifier(text)[0]
        return {
            'label': result['label'],
            'score': result['score']
        }
    
    def analyze_batch(self, texts, batch_size=32):
        """
        Analyze sentiment of multiple texts in batch
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = self.classifier(batch)
            results.extend(batch_results)
        return results
    
    def compute_tone_score(self, texts):
        """
        Compute tone score for a collection of texts
        """
        if not texts:
            return 0
        
        results = self.analyze_batch(texts)
        tone_scores = []
        
        for result in results:
            label = result['label']
            score = result['score']
            
            if label == 'positive':
                tone_scores.append(score)
            elif label == 'negative':
                tone_scores.append(-score)
            else:  # neutral
                tone_scores.append(0)
        
        return np.mean(tone_scores) if tone_scores else 0
    
    def compute_tone_probabilities(self, texts):
        """
        Compute probability scores for each sentiment class
        """
        results = self.analyze_batch(texts)
        probs = []
        
        for result in results:
            label = result['label']
            score = result['score']
            
            if label == 'positive':
                probs.append({'negative': 0, 'neutral': 0, 'positive': score})
            elif label == 'negative':
                probs.append({'negative': score, 'neutral': 0, 'positive': 0})
            else:
                probs.append({'negative': 0, 'neutral': score, 'positive': 0})
        
        return pd.DataFrame(probs)

# Usage example
if __name__ == "__main__":
    # Test with sample text
    sample_texts = [
        "Our company has achieved record revenues this quarter",
        "We are facing significant challenges in the current market",
        "The company continues to execute its strategic plan"
    ]
    
    analyzer = FinBERTAnalyzer()
    tone_score = analyzer.compute_tone_score(sample_texts)
    print(f"Tone Score: {tone_score:.4f}")
    
    probabilities = analyzer.compute_tone_probabilities(sample_texts)
    print(probabilities)