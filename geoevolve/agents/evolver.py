import os
import json
import re
import time
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class CodeEvolver:
    """
    LLM-based code mutation and evolution engine.
    Generates improved kriging code using Gemini.
    """
    
    def __init__(self, model="gemini-flash-latest", temperature=0.8, max_retries=3):
        """
        Initialize the evolver with Gemini client.
        
        Args:
            model: Model name (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')
            temperature: Sampling temperature (0-1, higher = more creative)
            max_retries: Max retry attempts for LLM calls
        """
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set in .env file")
        
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "max_output_tokens": 2000,
            }
        )
        self.max_retries = max_retries
        self.call_count = 0
        self.total_tokens = 0
    
    def mutate(self, current_code: str, prompt: str) -> str:
        """
        Request LLM to mutate and improve the kriging code.
        
        Args:
            current_code: Current kriging implementation
            prompt: Improvement instructions from analyzer + RAG context
            
        Returns:
            str: Improved Python code string
        """
        
        system_instructions = """You are an expert in geostatistics and Python kriging algorithms.
Your task is to improve kriging code to reduce RMSE (mean squared error) in spatial interpolation.

CRITICAL RULES:
1. Return ONLY valid, executable Python code. No markdown, no explanations.
2. Define a function called evaluate(dataset) that takes a dataset dict and returns float RMSE.
3. The dataset dict has keys 'train' and 'test', each containing (x, y, z) tuples.
4. You can only import: numpy, pandas, OrdinaryKriging, KNeighborsRegressor, math
5. Do not add new imports or modify the function signature.
6. Make MINIMAL but meaningful changes to improve the algorithm.
7. Test your code mentally - ensure it runs without errors.

Output format: PURE PYTHON CODE ONLY."""
        
        user_message = f"""{system_instructions}

Current kriging code:
```python
{current_code}
```

Improvement instruction:
{prompt}

Generate improved code that addresses the weakness. Return ONLY the Python code."""
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(user_message)
                time.sleep(2)  # Avoid rate limits
                
                self.call_count += 1
                # Gemini doesn't always provide token counts in the same way as OpenAI, 
                # but we can try to estimate or get it if available
                # For simplicity in MVP, we'll just track calls
                
                code = response.text.strip()
                
                # Clean up markdown code blocks if present
                code = self._extract_code(code)
                
                return code
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        return current_code  # Fallback to current code if all retries fail
    
    def _extract_code(self, text: str) -> str:
        """
        Extract Python code from markdown or mixed text.
        
        Args:
            text: Raw text potentially containing markdown code blocks
            
        Returns:
            str: Extracted Python code
        """
        # Try to extract from markdown code block
        match = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no markdown block, return as-is (already pure Python)
        return text.strip()
    
    def get_stats(self):
        """Return API usage statistics"""
        return {
            'calls': self.call_count,
            'model': self.model_name
        }


if __name__ == "__main__":
    # Test the evolver
    test_code = """
import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error

def evaluate(dataset):
    x_train, y_train, z_train = dataset['train']
    x_test, y_test, z_test = dataset['test']
    
    ok = OrdinaryKriging(x_train, y_train, z_train, variogram_model='spherical', verbose=False, enable_plotting=False)
    z_pred, ss = ok.execute('points', x_test, y_test)
    
    rmse = np.sqrt(mean_squared_error(z_test, z_pred))
    return rmse
"""
    
    prompt = "Try using a Gaussian variogram model instead of spherical."
    
    # Note: Requires GOOGLE_API_KEY in .env
    try:
        evolver = CodeEvolver(model="gemini-1.5-flash", temperature=0.8)
        improved_code = evolver.mutate(test_code, prompt)
        print("Improved code:")
        print(improved_code)
        print(f"\nStats: {evolver.get_stats()}")
    except Exception as e:
        print(f"Test failed: {e}")
