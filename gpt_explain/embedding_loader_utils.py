"""
Utility functions for loading saved embeddings and masked data.
"""

import torch
import numpy as np
import json
import os
from pathlib import Path

def load_center_embeddings(sentence_file_prefix, data_dir="gpt_explain"):
    """
    Load center embeddings for a sentence
    
    Args:
        sentence_file_prefix: Safe filename prefix (e.g., "The_food_was_cheap_fresh_and_tasty")
        data_dir: Directory containing the data files
    
    Returns:
        dict with keys: sentence, tokens, token_ids, embeddings (torch.Tensor)
    """
    # Load metadata
    npz_file = os.path.join(data_dir, f"center_embeddings_{sentence_file_prefix}.npz")
    tensor_file = os.path.join(data_dir, f"center_embeddings_{sentence_file_prefix}_tensor.pt")
    
    if not os.path.exists(npz_file) or not os.path.exists(tensor_file):
        raise FileNotFoundError(f"Files not found: {npz_file} or {tensor_file}")
    
    # Load metadata
    data = dict(np.load(npz_file, allow_pickle=True))
    
    # Load embeddings tensor
    embeddings = torch.load(tensor_file)
    
    data['embeddings'] = embeddings
    return data

def load_masked_data(sentence_file_prefix, baseline="zero", data_dir="gpt_explain"):
    """
    Load masked embeddings for a sentence
    
    Args:
        sentence_file_prefix: Safe filename prefix
        baseline: "zero" or "mean"
        data_dir: Directory containing the data files
    
    Returns:
        dict with keys: sentence, tokens, subsets, masked_embeddings (torch.Tensor)
    """
    # Load metadata
    npz_file = os.path.join(data_dir, f"masked_data_{sentence_file_prefix}_{baseline}.npz")
    tensor_file = os.path.join(data_dir, f"masked_data_{sentence_file_prefix}_{baseline}_tensor.pt")
    
    if not os.path.exists(npz_file) or not os.path.exists(tensor_file):
        raise FileNotFoundError(f"Files not found: {npz_file} or {tensor_file}")
    
    # Load metadata
    data = dict(np.load(npz_file, allow_pickle=True))
    
    # Load masked embeddings tensor
    masked_embeddings = torch.load(tensor_file)
    
    data['masked_embeddings'] = masked_embeddings
    return data

def load_sentence_metadata(sentence_file_prefix, data_dir="gpt_explain"):
    """Load metadata JSON for a sentence"""
    metadata_file = os.path.join(data_dir, f"metadata_{sentence_file_prefix}.json")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        return json.load(f)

def list_available_sentences(data_dir="gpt_explain"):
    """List all available sentence prefixes"""
    metadata_files = list(Path(data_dir).glob("metadata_*.json"))
    prefixes = []
    
    for file in metadata_files:
        prefix = file.stem.replace("metadata_", "")
        prefixes.append(prefix)
    
    return sorted(prefixes)

# Example usage:
if __name__ == "__main__":
    # List available sentences
    sentences = list_available_sentences()
    print(f"Available sentences: {sentences}")
    
    if sentences:
        # Load first sentence as example
        prefix = sentences[0]
        print(f"\nLoading data for: {prefix}")
        
        # Load center embeddings
        center_data = load_center_embeddings(prefix)
        print(f"Center embeddings shape: {center_data['embeddings'].shape}")
        print(f"Tokens: {center_data['tokens']}")
        
        # Load masked data
        masked_data = load_masked_data(prefix, "zero")
        print(f"Masked embeddings shape: {masked_data['masked_embeddings'].shape}")
        print(f"Number of subsets: {len(masked_data['subsets'])}")
        
        # Load metadata
        metadata = load_sentence_metadata(prefix)
        print(f"Metadata: {metadata}")
