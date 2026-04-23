import os
import json
from dotenv import load_dotenv

load_dotenv()

class PromptBuilder:
    """
    Builds optimized prompts for the LLM evolver by combining:
    - Current code
    - Identified weakness
    - RAG context from knowledge base
    - Improvement suggestions
    """
    
    def build(self, 
              code: str,
              weakness: str,
              rag_context: str = None,
              suggestion: str = None) -> str:
        """
        Build a comprehensive improvement prompt.
        
        Args:
            code: Current kriging implementation
            weakness: Identified algorithmic weakness
            rag_context: Retrieved knowledge base context
            suggestion: Specific improvement suggestion
            
        Returns:
            str: Comprehensive prompt for LLM
        """
        
        prompt_parts = []
        
        # Add weakness analysis
        prompt_parts.append(f"IDENTIFIED WEAKNESS:\n{weakness}")
        
        # Add RAG context if available
        if rag_context and rag_context.strip():
            prompt_parts.append(f"\nRELEVANT GEOSPATIAL THEORY:\n{rag_context}")
        
        # Add suggestion
        if suggestion:
            prompt_parts.append(f"\nSUGGESTED APPROACH:\n{suggestion}")
        
        # Add hardcoded fallback suggestions if RAG is empty
        if not rag_context:
            prompt_parts.append("\nGENERAL KRIGING IMPROVEMENTS:")
            prompt_parts.append("- Consider different variogram models (Gaussian, Exponential, Matern)")
            prompt_parts.append("- Implement cross-validation for parameter selection")
            prompt_parts.append("- Use local kriging for better spatial adaptivity")
            prompt_parts.append("- Add trend removal or detrending preprocessing")
        
        return "\n".join(prompt_parts)
    
    def build_minimal(self, code: str) -> str:
        """
        Build minimal prompt when no analysis available.
        """
        return """Your task: Improve this kriging code to reduce RMSE.
        
Consider these modifications:
- Try a different variogram model
- Adjust kriging parameters
- Implement local kriging with adaptive bandwidth
- Add preprocessing steps

Return only executable Python code."""


if __name__ == "__main__":
    builder = PromptBuilder()
    
    code = "dummy_kriging_code"
    weakness = "Fixed variogram model lacks adaptivity"
    rag_context = "Adaptive kriging uses local neighborhoods..."
    suggestion = "Implement K-nearest neighbor kriging"
    
    prompt = builder.build(code, weakness, rag_context, suggestion)
    print("Generated Prompt:")
    print(prompt)
