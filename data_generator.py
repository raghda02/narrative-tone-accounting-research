# ============================================================================
# data_generator.py
# Generate sample data for testing and immediate execution
# ============================================================================

import numpy as np
import pandas as pd
import random
import os

def generate_financial_data(n_firms=50, n_years=10, seed=42):
    """
    Generate simulated financial data mimicking Compustat database structure
    
    Parameters:
    -----------
    n_firms : int
        Number of firms to generate
    n_years : int
        Number of years per firm
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame: Simulated financial data
    """
    np.random.seed(seed)
    random.seed(seed)
    
    firms = [f'Firm_{i:04d}' for i in range(1, n_firms + 1)]
    years = list(range(2025 - n_years + 1, 2025 + 1))
    
    data = []
    
    for firm in firms:
        # Firm-specific fixed characteristics
        base_size = np.random.uniform(10, 100)  # Base assets in billions
        base_leverage = np.random.uniform(0.1, 0.6)
        base_roa = np.random.uniform(-0.05, 0.15)
        
        for year in years:
            # Annual variation with some randomness
            size = base_size * (1.05 ** (year - 2025 + n_years)) * np.random.normal(1, 0.02)
            leverage = base_leverage + np.random.normal(0, 0.02)
            roa = base_roa + np.random.normal(0, 0.01)
            
            # Derived financial metrics
            total_assets = size * 1000  # Convert to millions
            revenue = total_assets * np.random.uniform(0.5, 1.2)
            net_income = revenue * roa
            operating_cf = net_income * np.random.uniform(0.7, 1.3)
            total_debt = total_assets * np.clip(leverage, 0, 0.8)
            ppe = total_assets * np.random.uniform(0.2, 0.5)
            receivables = revenue * np.random.uniform(0.05, 0.15)
            market_cap = total_assets * np.random.uniform(1.0, 3.0)
            
            data.append({
                'firm_id': firm,
                'year': year,
                'total_assets': total_assets,
                'revenue': revenue,
                'net_income': net_income,
                'operating_cf': operating_cf,
                'total_debt': total_debt,
                'ppe': ppe,
                'receivables': receivables,
                'market_cap': market_cap,
                'leverage': np.clip(leverage, 0, 0.8),
                'roa': roa,
                'mtb': market_cap / total_assets,
                'industry': np.random.choice(['Tech', 'Healthcare', 'Energy', 'Consumer', 'Industrial'], 
                                            p=[0.3, 0.2, 0.2, 0.15, 0.15]),
                'big4': np.random.choice([0, 1], p=[0.15, 0.85])
            })
    
    df = pd.DataFrame(data)
    df['cfo'] = df['operating_cf'] / df['total_assets']
    
    return df

def generate_textual_features(n_firms=50, n_years=10, seed=42):
    """
    Generate simulated textual features mimicking NLP analysis results
    
    Returns:
    --------
    pd.DataFrame: Textual features (tone, uncertainty, readability)
    """
    np.random.seed(seed + 1)
    
    firms = [f'Firm_{i:04d}' for i in range(1, n_firms + 1)]
    years = list(range(2025 - n_years + 1, 2025 + 1))
    
    data = []
    for firm in firms:
        # Firm-specific base tone
        base_tone = np.random.uniform(-0.2, 0.5)
        
        for year in years:
            # Annual variation in tone
            tone = base_tone + np.random.normal(0, 0.05)
            chatgpt_tone = tone + np.random.normal(0, 0.03)
            
            # Uncertainty (higher when tone is lower)
            uncertainty = 0.015 + (0.5 - tone) * 0.02 + np.random.normal(0, 0.005)
            uncertainty = np.clip(uncertainty, 0.005, 0.08)
            
            # Readability (Fog Index)
            readability = 16 + (1 - tone) * 5 + np.random.normal(0, 1)
            readability = np.clip(readability, 10, 30)
            
            # Dictionary-based tone (benchmark)
            dict_tone = tone * 0.7 + np.random.normal(0, 0.03)
            
            data.append({
                'firm_id': firm,
                'year': year,
                'tone_finbert': np.clip(tone, -0.3, 0.6),
                'tone_chatgpt': np.clip(chatgpt_tone, -0.3, 0.6),
                'tone_dictionary': np.clip(dict_tone, -0.3, 0.6),
                'uncertainty': uncertainty,
                'readability_fog': readability,
                'mdna_length': np.random.uniform(5000, 20000)
            })
    
    return pd.DataFrame(data)

def generate_full_sample_data(n_firms=50, n_years=10, output_dir="data/"):
    """
    Generate and save complete sample dataset
    
    Parameters:
    -----------
    n_firms : int
        Number of firms
    n_years : int
        Number of years
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    pd.DataFrame: Complete merged dataset
    """
    
    print("Generating financial data...")
    financial_df = generate_financial_data(n_firms, n_years)
    
    print("Generating textual features...")
    textual_df = generate_textual_features(n_firms, n_years)
    
    print("Merging data...")
    merged_df = pd.merge(financial_df, textual_df, on=['firm_id', 'year'], how='inner')
    
    # Compute peer tone (instrumental variable)
    merged_df['peer_tone'] = merged_df.groupby(['industry', 'year'])['tone_finbert'].transform(
        lambda x: (x.sum() - x) / (x.count() - 1) if x.count() > 1 else np.nan
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    merged_df.to_csv(f"{output_dir}/financial_data.csv", index=False)
    merged_df.to_csv(f"{output_dir}/sample_data.csv", index=False)
    
    print(f"✅ Data generated successfully!")
    print(f"   - Total rows: {len(merged_df)}")
    print(f"   - Unique firms: {merged_df['firm_id'].nunique()}")
    print(f"   - Years covered: {merged_df['year'].min()} - {merged_df['year'].max()}")
    print(f"   - Files saved in: {output_dir}")
    
    return merged_df

def generate_mda_text(firm_id, year, tone, n_sentences=20, seed=None):
    """
    Generate synthetic MD&A text reflecting the specified tone
    
    Parameters:
    -----------
    firm_id : str
        Firm identifier
    year : int
        Fiscal year
    tone : float
        Tone value (-1 to 1)
    n_sentences : int
        Number of sentences to generate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    str: Synthetic MD&A text
    """
    if seed:
        random.seed(seed)
    
    positive_phrases = [
        "Our performance has been strong",
        "We are optimistic about future growth",
        "Revenue increased significantly",
        "We have successfully expanded",
        "Our strategy is delivering results",
        "We achieved record profits",
        "We are confident in our position",
        "Positive momentum continues",
        "We have exceeded expectations",
        "We are investing for the future",
        "We are on track to achieve our targets",
        "Our cash flows are robust"
    ]
    
    negative_phrases = [
        "We faced significant challenges",
        "Market conditions remain uncertain",
        "We experienced headwinds",
        "We are cautious about the outlook",
        "We have encountered difficulties",
        "We are monitoring risks",
        "We have seen pressure on margins",
        "Competition has intensified",
        "We are managing cost increases",
        "There is uncertainty in our outlook"
    ]
    
    neutral_phrases = [
        "We continue to monitor the situation",
        "The company remains focused",
        "We are executing our plan",
        "We are following our strategy",
        "We are making progress",
        "We are adapting to the environment",
        "We are committed to our goals",
        "We are evaluating opportunities"
    ]
    
    # Determine sentence selection based on tone
    sentences = []
    tone_ratio = 0.6 + abs(tone) * 0.2
    
    for i in range(n_sentences):
        rand = random.random()
        
        if rand < tone_ratio:
            if tone > 0.1:
                phrase = random.choice(positive_phrases)
            elif tone < -0.1:
                phrase = random.choice(negative_phrases)
            else:
                phrase = random.choice(neutral_phrases)
        else:
            phrase = random.choice(neutral_phrases + positive_phrases + negative_phrases)
        
        # Add some numerical details for realism
        if random.random() > 0.7:
            phrase += f" with growth of {random.randint(2, 15)}%"
        
        if random.random() > 0.8:
            phrase += f" in fiscal {year - random.randint(0, 2)}"
        
        sentences.append(phrase)
    
    # Introduction and conclusion
    if tone > 0.1:
        intro = f"In {year}, {firm_id} delivered strong results across all segments."
    elif tone < -0.1:
        intro = f"In {year}, {firm_id} faced significant challenges in a difficult market."
    else:
        intro = f"In {year}, {firm_id} maintained stable operations."
    
    conclusion = "We remain committed to creating long-term value for our shareholders."
    
    full_text = intro + " " + ". ".join(sentences) + ". " + conclusion
    
    return full_text

if __name__ == "__main__":
    # Generate sample data
    data = generate_full_sample_data(n_firms=50, n_years=10)
    
    print("\n📊 Sample Data Preview:")
    print(data[['firm_id', 'year', 'tone_finbert', 'tone_chatgpt', 'roa']].head())
    
    print("\n📊 Summary Statistics:")
    print(data[['tone_finbert', 'tone_chatgpt', 'tone_dictionary', 'uncertainty', 'roa']].describe())