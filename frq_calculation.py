# ============================================================================
# frq_calculation.py
# Financial Reporting Quality (FRQ) calculations
# ============================================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm

class FRQCalculator:
    """
    Calculate financial reporting quality metrics
    """
    
    def __init__(self, data):
        self.data = data.copy()
    
    def calculate_accruals_quality(self, firm_id_col='firm_id', year_col='year'):
        """
        Calculate accruals quality using Dechow and Dichev (2002) model
        """
        results = []
        
        for firm_id in self.data[firm_id_col].unique():
            firm_data = self.data[self.data[firm_id_col] == firm_id].sort_values(year_col)
            
            if len(firm_data) < 6:
                continue
            
            # Calculate required fields
            firm_data['tca'] = firm_data['net_income'] - firm_data['operating_cf']
            firm_data['tca'] = firm_data['tca'] / firm_data['total_assets']
            
            firm_data['cfo'] = firm_data['operating_cf'] / firm_data['total_assets']
            firm_data['cfo_lag1'] = firm_data['cfo'].shift(1)
            firm_data['cfo_lead1'] = firm_data['cfo'].shift(-1)
            
            firm_data['delta_rev'] = firm_data['revenue'].diff() / firm_data['total_assets'].shift(1)
            firm_data['ppe'] = firm_data['ppe'] / firm_data['total_assets']
            
            # Rolling window estimation
            for i in range(4, len(firm_data) - 4):
                window = firm_data.iloc[i-4:i+4].dropna()
                
                if len(window) < 6:
                    continue
                
                X = window[['cfo_lag1', 'cfo', 'cfo_lead1', 'delta_rev', 'ppe']]
                X = sm.add_constant(X)
                y = window['tca']
                
                try:
                    model = sm.OLS(y, X).fit()
                    residuals = model.resid
                    
                    results.append({
                        'firm_id': firm_id,
                        'year': window.index[-1],
                        'accruals_quality': -np.std(residuals),
                        'n_obs': len(window)
                    })
                except:
                    continue
        
        return pd.DataFrame(results)
    
    def calculate_discretionary_accruals(self, firm_id_col='firm_id', year_col='year'):
        """
        Calculate discretionary accruals using Modified Jones Model
        """
        results = []
        
        for firm_id in self.data[firm_id_col].unique():
            firm_data = self.data[self.data[firm_id_col] == firm_id].sort_values(year_col)
            
            if len(firm_data) < 6:
                continue
            
            # Calculate total accruals
            firm_data['total_accruals'] = (firm_data['net_income'] - firm_data['operating_cf'])
            firm_data['total_accruals'] = firm_data['total_accruals'] / firm_data['total_assets']
            
            firm_data['assets_lag'] = firm_data['total_assets'].shift(1)
            firm_data['delta_rev'] = firm_data['revenue'].diff()
            firm_data['delta_rec'] = firm_data['receivables'].diff()
            
            # Estimate by industry-year
            for (industry, year), group in firm_data.groupby(['industry', year_col]):
                if len(group) < 8:
                    continue
                
                X = pd.DataFrame({
                    'const': 1,
                    'inv_assets': 1 / group['assets_lag'],
                    'delta_rev_adj': (group['delta_rev'] - group['delta_rec']) / group['assets_lag'],
                    'ppe': group['ppe'] / group['assets_lag']
                })
                y = group['total_accruals'] / group['assets_lag']
                
                try:
                    model = sm.OLS(y, X).fit()
                    
                    for idx in group.index:
                        row = group.loc[idx]
                        nondiscretionary = model.predict({
                            'const': 1,
                            'inv_assets': 1 / row['assets_lag'],
                            'delta_rev_adj': (row['delta_rev'] - row['delta_rec']) / row['assets_lag'],
                            'ppe': row['ppe'] / row['assets_lag']
                        })[0]
                        
                        discretionary = (row['total_accruals'] / row['assets_lag']) - nondiscretionary
                        
                        results.append({
                            'firm_id': firm_id,
                            'year': year,
                            'discretionary_accruals': discretionary,
                            'abs_discretionary_accruals': abs(discretionary)
                        })
                except:
                    continue
        
        return pd.DataFrame(results)

# Usage example
if __name__ == "__main__":
    # Test with sample data
    from data_generator import generate_financial_data
    data = generate_financial_data(n_firms=10, n_years=5)
    frq_calc = FRQCalculator(data)
    
    accruals = frq_calc.calculate_accruals_quality()
    print("Accruals Quality Results:")
    print(accruals.head())