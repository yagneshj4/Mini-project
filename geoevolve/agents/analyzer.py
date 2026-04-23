import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class CodeAnalyzer:
    """
    LLM-based code analyzer that identifies weaknesses in kriging algorithms.
    Returns structured feedback for improvement and RAG query terms.
    """
    
    def __init__(self, model="gpt-4o-mini", temperature=0.3, max_retries=3):
        """
        Initialize the analyzer.
        
        Args:
            model: Model name (gpt-4o-mini is faster/cheaper for analysis)
            temperature: Lower for more consistent analysis
            max_retries: Max retry attempts
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
    
    def analyze(self, code: str, rmse: float) -> dict:
        """
        Analyze kriging code and identify improvement opportunities.
        
        Args:
            code: Current Python kriging code
            rmse: Current RMSE score
            
        Returns:
            dict: {
                'weakness': str (description of identified issue),
                'query': str (3-5 word search term for RAG),
                'suggestion': str (specific improvement idea)
            }
        """
        
        system_message = """You are an expert geospatial data scientist specializing in kriging and spatial interpolation.
Analyze the provided kriging code and identify ONE key weakness that likely causes high RMSE.
Focus on algorithmic limitations, not minor code style issues.

Return ONLY valid JSON with no additional text."""
        
        user_message = f"""Analyze this kriging code. Current RMSE: {rmse:.4f}

Code:
```python
{code}
```

Identify the main algorithmic weakness. Return JSON:
{{
  "weakness": "specific algorithmic issue preventing lower RMSE",
  "query": "3-5 word search term for kriging knowledge base",
  "suggestion": "specific algorithmic change to try"
}}"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Try to extract JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(0))
                    
                    # Validate keys
                    if 'weakness' in analysis and 'query' in analysis:
                        return analysis
                
                # Fallback structured response
                return {
                    'weakness': 'The variogram model may not be optimal for the spatial structure',
                    'query': 'variogram model selection kriging',
                    'suggestion': 'Try multiple variogram models and select based on cross-validation'
                }
                
            except json.JSONDecodeError:
                print(f"JSON decode attempt {attempt + 1} failed, retrying...")
                if attempt == self.max_retries - 1:
                    # Return safe default
                    return {
                        'weakness': 'The algorithm uses a fixed variogram model without adaptation',
                        'query': 'adaptive kriging variogram selection',
                        'suggestion': 'Implement variogram model selection cross-validation'
                    }
            except Exception as e:
                print(f"Analysis attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return {
                        'weakness': 'The kriging parameters may need tuning for better generalization',
                        'query': 'kriging cross validation parameter optimization',
                        'suggestion': 'Add cross-validation for variogram parameter optimization'
                    }
        
        # Final fallback
        return {
            'weakness': 'The current implementation may benefit from model selection',
            'query': 'kriging model selection AIC BIC',
            'suggestion': 'Implement model selection criteria for variogram choice'
        }


if __name__ == "__main__":
    # Test the analyzer
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
    
    analyzer = CodeAnalyzer(model="gpt-4o-mini")
    analysis = analyzer.analyze(test_code, rmse=0.91)
    
    print("Analysis:")
    print(json.dumps(analysis, indent=2))
