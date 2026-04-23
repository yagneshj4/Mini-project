import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import re

load_dotenv()

class CodeEvolver:
    """
    LLM-based code mutation and evolution engine.
    Generates improved kriging code using GPT-4o.
    """
    
    def __init__(self, model="gpt-4o", temperature=0.8, max_retries=3):
        """
        Initialize the evolver with OpenAI client.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'gpt-4o-mini')
            temperature: Sampling temperature (0-1, higher = more creative)
            max_retries: Max retry attempts for LLM calls
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
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
        
        system_message = """You are an expert in geostatistics and Python kriging algorithms.
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
        
        user_message = f"""Current kriging code:
```python
{current_code}
```

Improvement instruction:
{prompt}

Generate improved code that addresses the weakness. Return ONLY the Python code."""
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature,
                    max_tokens=2000,
                    top_p=0.95
                )
                
                self.call_count += 1
                self.total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
                
                code = response.choices[0].message.content.strip()
                
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
            'total_tokens': self.total_tokens,
            'avg_tokens_per_call': self.total_tokens / self.call_count if self.call_count > 0 else 0
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
    
    evolver = CodeEvolver(model="gpt-4o-mini", temperature=0.8)
    improved_code = evolver.mutate(test_code, prompt)
    
    print("Improved code:")
    print(improved_code)
    print(f"\nStats: {evolver.get_stats()}")
