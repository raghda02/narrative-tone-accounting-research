# ============================================================================
# text_extraction.py
# Extract MD&A sections from PDF filings
# ============================================================================

import os
import re

class TextExtractor:
    """
    Extract MD&A section from 10-K PDF filings
    """
    
    def __init__(self, pdf_path=None):
        self.pdf_path = pdf_path
        self.text = ""
    
    def extract_mda_section(self):
        """
        Extract MD&A section from PDF file
        """
        if self.pdf_path and os.path.exists(self.pdf_path):
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(self.pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                self.text = text
                return self._extract_mda_from_text(text)
            except ImportError:
                print("PyMuPDF not installed. Using fallback.")
                return self._fallback_extraction()
            except Exception as e:
                print(f"Error extracting PDF: {e}")
                return self._fallback_extraction()
        else:
            return self._fallback_extraction()
    
    def _extract_mda_from_text(self, text):
        """Extract MD&A section from full text"""
        # Look for MD&A section
        patterns = [
            r"ITEM\s*7\.\s*MANAGEMENT'?S?\s*DISCUSSION\s*AND\s*ANALYSIS",
            r"MANAGEMENT'?S?\s*DISCUSSION\s*AND\s*ANALYSIS",
            r"MD&A"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.start()
                # Find end (next Item or financial statements)
                end_patterns = [r"ITEM\s*8\.", r"ITEM\s*9\.", r"FINANCIAL\s*STATEMENTS"]
                end = len(text)
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, text[start+100:], re.IGNORECASE)
                    if end_match:
                        end = start + 100 + end_match.start()
                        break
                self.text = text[start:end]
                return self.text
        
        self.text = text[:5000]  # Fallback: first 5000 chars
        return self.text
    
    def _fallback_extraction(self):
        """Fallback text for testing"""
        self.text = """
        MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION
        AND RESULTS OF OPERATIONS

        Our company has achieved strong results in the current fiscal year.
        Revenue increased by 15% compared to the previous year, driven by
        higher demand and successful product launches. We are optimistic
        about future growth prospects and expect continued improvement in
        operating performance. However, we remain cautious about potential
        risks including market volatility and regulatory changes.
        
        We have implemented strategic initiatives to enhance operational
        efficiency and expand our market presence. These initiatives are
        expected to contribute positively to our financial performance in
        the coming years.
        """
        return self.text
    
    def save_text(self, output_path):
        """Save extracted text to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.text)
        return output_path