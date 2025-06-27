"""
Simple LaTeX Generator - Phase 1: Tokenization Foundation
"""

from typing import List, Dict, Tuple

# TODO: Phase 1 - Implement LaTeX-aware tokenization
# 
# This phase focuses ONLY on tokenization - the foundation everything else builds on.
# We'll implement this step by step to make sure it works correctly.
# 
# Key components for Phase 1:
# 
def latex_tokenize(latex_string: str) -> List[str]:
    """LaTeX-aware tokenization that groups meaningful elements."""
    # Rules:
    # - Keep \ as special token (start of commands)
    # - Group consecutive letters (e.g., 'int' becomes one token)
    # - Group consecutive numbers (e.g., '123' becomes one token)
    # - Keep {, }, _, ^ as special tokens
    # - Keep spaces as tokens
    # - Handle LaTeX commands like \int, \frac, \sin, etc.
    # 
    # Example: \int_{0}^{1} x^2 dx
    # Tokens: ['\', 'int', '_', '{', '0', '}', '^', '{', '1', '}', ' ', 'x', '^', '2', ' ', 'd', 'x']
    # 
    # Implementation approach:
    # 1. Split by backslash to identify commands
    # 2. For each part, group letters, numbers, and special chars
    # 3. Handle nested structures carefully
    # Return list of tokens

    tokens = []
    i = 0
    
    while i < len(latex_string):
        char = latex_string[i]
        
        if char == '\\':
            # Handle backslash - start of command
            tokens.append('\\')
            i += 1
            # Collect command name (letters only)
            command = ''
            while i < len(latex_string) and latex_string[i].isalpha():
                command += latex_string[i]
                i += 1
            if command:
                tokens.append(command)
                
        elif char.isalpha():
            # Treat each letter as a separate token (unless part of a command)
            tokens.append(char)
            i += 1
            
        elif char.isdigit():
            # Collect consecutive digits
            digits = ''
            while i < len(latex_string) and latex_string[i].isdigit():
                digits += latex_string[i]
                i += 1
            tokens.append(digits)
            
        elif char in ['{', '}', '_', '^', ' ']:
            # Special characters as separate tokens
            tokens.append(char)
            i += 1
            
        else:
            # Other characters
            tokens.append(char)
            i += 1
    
    return tokens


def test_tokenization():
    """Test the tokenization function with various LaTeX examples."""
    # TODO: Future improvement - Move this to tests/test_tokenization.py using pytest
    # This will allow for better test organization, fixtures, and automated testing
    test_cases = [
        (r"\int", ['\\', 'int']),
        (r"x^2", ['x', '^', '2']),
        (r"a_i", ['a', '_', 'i']),
        (r"\frac{1}{2}", ['\\', 'frac', '{', '1', '}', '{', '2', '}']),
        (r"\int_{0}^{1} x^2 dx", ['\\', 'int', '_', '{', '0', '}', '^', '{', '1', '}', ' ', 'x', '^', '2', ' ', 'd', 'x']),
        (r"\sin(x)", ['\\', 'sin', '(', 'x', ')']),
        (r"", []),  # Empty string
        (r"123", ['123']),  # Just numbers
    ]
    
    print("Testing LaTeX tokenization:")
    for latex, expected in test_cases:
        result = latex_tokenize(latex)
        status = "✅" if result == expected else "❌"
        print(f"{status} Input: {latex!r}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
        print()


# def build_latex_vocab(latex_strings: List[str], max_vocab_size: int = 200) -> Tuple[Dict[str, int], Dict[int, str]]:
#     """Build vocabulary from LaTeX strings with LaTeX awareness."""
#     # Count all tokens from latex_tokenize
#     # Add special tokens: <START>, <END>, <PAD>, <UNK>
#     # Keep most common LaTeX commands and symbols
#     # Return token_to_id and id_to_token mappings
# 
# def tokenize_to_ids(latex_string: str, token_to_id: Dict[str, int]) -> List[int]:
#     """Convert LaTeX string to token IDs."""
#     # Tokenize the string
#     # Convert tokens to IDs
#     # Handle unknown tokens with <UNK>
#     # Return list of token IDs
# 
# def ids_to_latex(token_ids: List[int], id_to_token: Dict[int, str]) -> str:
#     """Convert token IDs back to LaTeX string."""
#     # Convert IDs to tokens
#     # Remove special tokens (<START>, <END>, <PAD>)
#     # Join tokens back into LaTeX string
#     # Return LaTeX string
# 
# def test_tokenization():
#     """Test the tokenization functions with simple examples."""
#     # Test cases:
#     # - Simple commands: \int, \frac, \sin
#     # - With subscripts/superscripts: x^2, a_i
#     # - Complex expressions: \int_{0}^{1} x^2 dx
#     # - Edge cases: nested braces, special characters
#     # Print results and verify correctness
# 
# # Phase 2+ will be added later:
# # - Simple CNN + LSTM model
# # - Simple training loop  
# # - Simple evaluation
# # - Integration and testing

# Simple test runner for development
if __name__ == "__main__":
    test_tokenization() 