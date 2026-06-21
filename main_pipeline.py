# ============================================================================
# main_pipeline.py
# Complete workflow execution
# ============================================================================

import os
import sys
import pandas as pd
import numpy as np

# Import modules
from data_generator import generate_full_sample_data
from text_extraction import TextExtractor
from text_preprocessing import TextPreprocessor
from finbert_analysis import FinBERTAnalyzer
from chatgpt_classification import ChatGPTClassifier
from entropy_validation import EntropyValidator
from frq_calculation import FRQCalculator
from econometric_analysis import PanelAnalysis

def run_full_pipeline(config):
    """
    Execute the complete NLP and econometric pipeline
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - output_dir: str, directory to save results
        - n_firms: int, number of firms for sample data
        - n_years: int, number of years
        - use_sample_data: bool, whether to generate sample data
    """
    
    print("=" * 70)
    print("NARRATIVE DISCLOSURE TONE AND FINANCIAL REPORTING QUALITY")
    print("AI-Based Textual Analysis Pipeline")
    print("=" * 70)
    
    # Step 1: Load or generate data
    print("\n" + "-" * 50)
    print("STEP 1: Data Loading")
    print("-" * 50)
    
    output_dir = config.get('output_dir', 'data/')
    os.makedirs(output_dir, exist_ok=True)
    
    if config.get('use_sample_data', True):
        print("Generating sample data...")
        data = generate_full_sample_data(
            n_firms=config.get('n_firms', 50),
            n_years=config.get('n_years', 10),
            output_dir=output_dir
        )
    else:
        print("Loading data from files...")
        data = pd.read_csv(f"{output_dir}/financial_data.csv")
    
    print(f"Data loaded: {len(data)} rows, {data['firm_id'].nunique()} firms")
    
    # Step 2: Prepare text data
    print("\n" + "-" * 50)
    print("STEP 2: Text Extraction and Preprocessing")
    print("-" * 50)
    
    preprocessor = TextPreprocessor()
    
    # For sample data, we simulate MD&A text
    if config.get('use_sample_data', True):
        print("Using simulated MD&A text from sample data...")
        # Create synthetic text based on tone
        texts = []
        for _, row in data.iterrows():
            text = f"""
            MANAGEMENT'S DISCUSSION AND ANALYSIS
            For the fiscal year ended {row['year']}
            
            Firm {row['firm_id']} operates in the {row['industry']} industry.
            We have achieved a return on assets of {row['roa']:.2%}.
            Our leverage ratio is {row['leverage']:.2%}.
            """
            if row['tone_finbert'] > 0.1:
                text += " We are optimistic about our future growth prospects."
            elif row['tone_finbert'] < -0.1:
                text += " We are cautious about market conditions."
            else:
                text += " We continue to execute our strategic plan."
            
            texts.append(text)
        
        data['mda_text'] = texts
    
    # Step 3: Tone analysis
    print("\n" + "-" * 50)
    print("STEP 3: Tone Analysis")
    print("-" * 50)
    
    # FinBERT analysis
    print("Running FinBERT sentiment analysis...")
    finbert = FinBERTAnalyzer()
    finbert_scores = []
    
    for text in data['mda_text'].iloc[:100]:  # Limit for speed
        try:
            score = finbert.compute_tone_score([text])
            finbert_scores.append(score)
        except:
            finbert_scores.append(np.nan)
    
    # ChatGPT analysis (if available)
    print("Running ChatGPT tone classification...")
    chatgpt = ChatGPTClassifier()
    chatgpt_results = []
    
    for text in data['mda_text'].iloc[:50]:  # Limit for speed
        try:
            result = chatgpt.classify_tone(text)
            chatgpt_results.append(result['tone_score'])
        except:
            chatgpt_results.append(0)
    
    # Step 4: Entropy validation
    print("\n" + "-" * 50)
    print("STEP 4: Entropy Validation")
    print("-" * 50)
    
    validator = EntropyValidator()
    validation_results = {}
    
    if 'peer_tone' in data.columns:
        validation = validator.validate_instrument(
            data['peer_tone'].dropna()[:100],
            data['tone_finbert'].dropna()[:100]
        )
        validation_results = validation
        print(f"Validation results: {validation}")
    
    # Step 5: Econometric analysis
    print("\n" + "-" * 50)
    print("STEP 5: Econometric Analysis")
    print("-" * 50)
    
    panel = PanelAnalysis(data)
    panel.prepare_panel()
    
    # Select variables
    y_var = 'roa'
    x_vars = ['tone_finbert', 'leverage']
    if 'mtb' in data.columns:
        x_vars.append('mtb')
    
    # Fixed effects
    print("Running fixed effects regression...")
    fe_model = panel.run_fixed_effects(y_var, x_vars)
    if fe_model:
        print(f"Fixed Effects R²: {fe_model.rsquared:.4f}")
        print(f"Tone coefficient: {fe_model.params.get('tone_finbert', 0):.4f}")
    
    # IV-2SLS
    if 'peer_tone' in data.columns:
        print("Running IV-2SLS regression...")
        iv_model = panel.run_iv_2sls(y_var, x_vars, 'peer_tone')
        if iv_model:
            print(f"First-stage F-statistic: {iv_model['f_stat']:.2f}")
            print(f"Tone coefficient (IV): {iv_model['stage2'].params.get('tone_finbert', 0):.4f}")
    
    # Step 6: Save results
    print("\n" + "-" * 50)
    print("STEP 6: Saving Results")
    print("-" * 50)
    
    results_dir = config.get('results_dir', 'results/')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save summary
    summary = {
        'Total firms': data['firm_id'].nunique(),
        'Total observations': len(data),
        'Years covered': f"{data['year'].min()} - {data['year'].max()}",
        'FinBERT tone mean': data['tone_finbert'].mean(),
        'FinBERT tone std': data['tone_finbert'].std(),
        'ROA mean': data['roa'].mean(),
        'ROA std': data['roa'].std()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{results_dir}/summary_stats.csv", index=False)
    
    print(f"Results saved to: {results_dir}/summary_stats.csv")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return data, fe_model, validation_results

# Usage example
if __name__ == "__main__":
    config = {
        'output_dir': 'data/',
        'results_dir': 'results/',
        'n_firms': 50,
        'n_years': 10,
        'use_sample_data': True
    }
    
    data, model, validation = run_full_pipeline(config)