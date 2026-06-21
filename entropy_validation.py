# ============================================================================
# entropy_validation.py
# Entropy-based instrument validation
# ============================================================================

import numpy as np
import pandas as pd
from scipy.stats import entropy

class EntropyValidator:
    """
    Validate instrumental variables using entropy-based measures
    """
    
    def __init__(self):
        pass
    
    def compute_entropy(self, series):
        """
        Compute Shannon entropy for a categorical series
        """
        counts = series.value_counts(normalize=True)
        return entropy(counts, base=2)
    
    def compute_entropy_continuous(self, series, bins=10):
        """
        Compute Shannon entropy for continuous variable using binning
        """
        hist, _ = np.histogram(series, bins=bins, density=True)
        hist = hist / hist.sum()  # Normalize to probabilities
        return entropy(hist, base=2)
    
    def validate_instrument(self, instrument_series, tone_series):
        """
        Perform comprehensive instrument validation
        """
        # Remove NaN values
        valid_indices = ~(instrument_series.isna() | tone_series.isna())
        instrument = instrument_series[valid_indices]
        tone = tone_series[valid_indices]
        
        if len(instrument) < 10:
            return {'error': 'Insufficient data for validation'}
        
        # Compute entropy
        h_instrument = self.compute_entropy_continuous(instrument)
        h_tone = self.compute_entropy_continuous(tone)
        
        # Compute correlation
        correlation = np.corrcoef(instrument, tone)[0, 1]
        
        # Compute coefficient of variation
        cv = np.std(instrument) / np.mean(instrument) if np.mean(instrument) != 0 else 0
        
        return {
            'entropy_instrument': h_instrument,
            'entropy_tone': h_tone,
            'correlation': correlation,
            'coefficient_of_variation': cv,
            'interpretation': self.interpret_entropy(h_instrument)
        }
    
    def interpret_entropy(self, h):
        """
        Interpret entropy value based on framework
        """
        if h > 0.8:
            return "High - Strong exogenous variation"
        elif h >= 0.4:
            return "Medium - Acceptable with caution"
        else:
            return "Low - Weak instrument risk"
    
    def compute_industry_tone_entropy(self, data, industry_col, tone_col, year_col):
        """
        Compute entropy of industry-year tone distribution
        """
        results = []
        for (industry, year), group in data.groupby([industry_col, year_col]):
            if len(group) > 1:
                h = self.compute_entropy_continuous(group[tone_col])
                results.append({
                    'industry': industry,
                    'year': year,
                    'entropy': h,
                    'n_firms': len(group)
                })
        return pd.DataFrame(results)

# Usage example
if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    instrument = np.random.normal(0, 1, 100)
    tone = 0.5 * instrument + np.random.normal(0, 0.5, 100)
    
    validator = EntropyValidator()
    result = validator.validate_instrument(
        pd.Series(instrument),
        pd.Series(tone)
    )
    print("Validation Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")