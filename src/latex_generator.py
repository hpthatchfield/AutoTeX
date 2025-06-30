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
def latex_tokenize(latex: str) -> list[str]:
    """
    Tokenize a LaTeX string into meaningful units: commands, groups, operators, numbers, and symbols.

    - Groups LaTeX commands (e.g., \int, \frac)
    - Groups consecutive digits
    - Treats {, }, _, ^, and spaces as separate tokens
    - Flags orphaned backslashes as <INVALID>
    """
    tokens = []
    i = 0
    while i < len(latex):
        c = latex[i]
        if c == '\\':
            tokens.append('\\')
            i += 1
            if i >= len(latex):
                tokens.append('<INVALID>')
                break
            cmd = ''
            while i < len(latex) and latex[i].isalpha():
                cmd += latex[i]
                i += 1
            if cmd:
                tokens.append(cmd)
            else:
                tokens.append('<INVALID>')
                if i < len(latex) and latex[i] == '\\':
                    i += 1
        elif c.isalpha():
            tokens.append(c)
            i += 1
        elif c.isdigit():
            num = ''
            while i < len(latex) and latex[i].isdigit():
                num += latex[i]
                i += 1
            tokens.append(num)
        elif c in '{}_^ ':
            tokens.append(c)
            i += 1
        else:
            tokens.append(c)
            i += 1
    return tokens


def build_latex_vocabulary(latex_strings: list[str]) -> dict[str, int]:
    """Build vocabulary from LaTeX strings with special tokens."""
    # Collect all unique tokens
    all_tokens = set()
    for latex_str in latex_strings:
        tokens = latex_tokenize(latex_str)
        all_tokens.update(tokens)
    
    # Create vocabulary with special tokens first
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    all_tokens_list = special_tokens + sorted(all_tokens)
    token_to_id = {token: i for i, token in enumerate(all_tokens_list)}
    
    return token_to_id

# 
def latex_to_token_ids(latex_string: str, token_to_id: Dict[str, int]) -> List[int]:
    """Convert LaTeX string to token IDs."""
    # Tokenize the string
    tokens = latex_tokenize(latex_string)
    # Convert tokens to IDs
    token_ids = [token_to_id.get(token, token_to_id['<UNK>']) for token in tokens]
    # We handle unknown tokens with <UNK> just to be safe :) 
    # Return list of token IDs
    return token_ids

def ids_to_latex(token_ids: List[int], id_to_token: Dict[int, str]) -> str:
    """Convert token IDs back to LaTeX string."""
    # Convert IDs to tokens
    tokens = [id_to_token.get(id, '<UNK>') for id in token_ids]
    # Remove special tokens (<START>, <END>, <PAD>)
    tokens = [token for token in tokens if token not in ['<START>', '<END>', '<PAD>']]
    # Join tokens back into LaTeX string
    latex_string = ''.join(tokens)
    # Return LaTeX string
    return latex_string



def test_tokenization():
    """Test the complete tokenization pipeline with various LaTeX examples."""
    # TODO: Future improvement - Move this to tests/test_tokenization.py using pytest
    # This will allow for better test organization, fixtures, and automated testing
    
    # Test cases for basic tokenization
    tokenization_test_cases = [
        (r"\int", ['\\', 'int']),
        (r"x^2", ['x', '^', '2']),
        (r"a_i", ['a', '_', 'i']),
        (r"\frac{1}{2}", ['\\', 'frac', '{', '1', '}', '{', '2', '}']),
        (r"\int_{0}^{1} x^2 dx", ['\\', 'int', '_', '{', '0', '}', '^', '{', '1', '}', ' ', 'x', '^', '2', ' ', 'd', 'x']),
        (r"\sin(x)", ['\\', 'sin', '(', 'x', ')']),
        (r"", []),  # Empty string
        (r"123", ['123']),  # Just numbers
        
        # Test orphaned backslash cases
        (r"\5", ['\\', '<INVALID>', '5']),  # Backslash followed by number
        (r"\{", ['\\', '<INVALID>', '{']),  # Backslash followed by brace
        (r"\ ", ['\\', '<INVALID>', ' ']),  # Backslash followed by space
        (r"\\", ['\\', '<INVALID>']),       # Backslash at end of string
        
        # Test semantic differences between braces and operators
        (r"x^{2+3}", ['x', '^', '{', '2', '+', '3', '}']),  # Superscript with expression
        (r"a_{i+1}", ['a', '_', '{', 'i', '+', '1', '}']),  # Subscript with expression
        
        # Test nested structures
        (r"\frac{\int x dx}{2}", ['\\', 'frac', '{', '\\', 'int', ' ', 'x', ' ', 'd', 'x', '}', '{', '2', '}']),
    ]
    
    print("=== Testing Basic LaTeX Tokenization ===")
    for latex, expected in tokenization_test_cases:
        result = latex_tokenize(latex)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"{status} Input: {latex!r}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
        if result != expected:
            print(f"   Mismatch at position: {next((i for i, (e, r) in enumerate(zip(expected, result)) if e != r), 'length')}")
        print()
    
    # Test complete pipeline (LaTeX → tokens → IDs → LaTeX)
    print("=== Testing Complete Tokenization Pipeline ===")
    
    # Create vocabulary from test cases
    test_latex_strings = [
        r"\int", r"x^2", r"\frac{1}{2}", r"\sin(x)", r"123",
        r"\int_{0}^{1} x^2 dx", r"a_i", r"x^{2+3}"
    ]
    
    vocab = build_latex_vocabulary(test_latex_strings)
    id_to_token = {id: token for token, id in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens: {[k for k in vocab.keys() if k.startswith('<')]}")
    print()
    
    # Test round-trip conversion
    for latex_str in test_latex_strings:
        print(f"Testing round-trip: {latex_str!r}")
        
        # Step 1: LaTeX → tokens
        tokens = latex_tokenize(latex_str)
        print(f"  Tokens: {tokens}")
        
        # Step 2: tokens → IDs
        token_ids = latex_to_token_ids(latex_str, vocab)
        print(f"  Token IDs: {token_ids}")
        
        # Step 3: IDs → LaTeX
        reconstructed = ids_to_latex(token_ids, id_to_token)
        print(f"  Reconstructed: {reconstructed!r}")
        
        # Check if round-trip worked
        status = "✓ PASS" if reconstructed == latex_str else "✗ FAIL"
        print(f"  Round-trip: {status}")
        if reconstructed != latex_str:
            print(f"    Original:  {latex_str!r}")
            print(f"    Reconstructed: {reconstructed!r}")
        print()
    
    # Test edge cases
    print("=== Testing Edge Cases ===")
    edge_cases = [
        r"",  # Empty string
        r"\\",  # Just backslash
        r"\5",  # Invalid command
        r"x^{2+3}",  # Complex expression
    ]
    
    for latex_str in edge_cases:
        print(f"Edge case: {latex_str!r}")
        tokens = latex_tokenize(latex_str)
        token_ids = latex_to_token_ids(latex_str, vocab)
        reconstructed = ids_to_latex(token_ids, id_to_token)
        
        print(f"  Tokens: {tokens}")
        print(f"  IDs: {token_ids}")
        print(f"  Reconstructed: {reconstructed!r}")
        print()


# Simple test runner for development
if __name__ == "__main__":
    test_tokenization()