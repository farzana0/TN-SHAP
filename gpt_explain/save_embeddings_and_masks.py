#!/usr/bin/env python3
"""
Save center embeddings and injected (masked) data separately for reuse.
This script extracts and saves:
1. Center embeddings: Original token embeddings of the main sentences
2. Injected data: Masked embeddings used for neighborhood sampling

Created: September 25, 2025
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_model_and_tokenizer(model_name="distilgpt2"):
    """Load GPT model and tokenizer"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    return model, tokenizer, device

def get_token_embeddings(text, model, tokenizer, device):
    """Get token embeddings for a given text"""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = tokens["input_ids"].to(device)
    
    with torch.no_grad():
        # Get embeddings directly from the embedding layer
        embeddings = model.get_input_embeddings()(input_ids)  # [1, seq_len, d]
    
    return embeddings.squeeze(0), input_ids.squeeze(0), tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

def create_masked_embeddings(embeddings, mask_positions, baseline="zero"):
    """Create masked embeddings by replacing specified positions with baseline"""
    masked_embeddings = embeddings.clone()
    
    if baseline == "zero":
        baseline_vector = torch.zeros_like(embeddings[0])
    elif baseline == "mean":
        # Use mean across all embeddings in the sequence
        baseline_vector = embeddings.mean(dim=0)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    for pos in mask_positions:
        if pos < embeddings.shape[0]:
            masked_embeddings[pos] = baseline_vector
    
    return masked_embeddings

def generate_all_subsets(n, max_k=2):
    """Generate all possible subsets up to size max_k"""
    from itertools import combinations
    
    subsets = []
    for k in range(1, min(max_k + 1, n + 1)):
        for subset in combinations(range(n), k):
            subsets.append(list(subset))
    
    return subsets

def save_embeddings_and_masks(sentences, output_dir="gpt_explain", model_name="distilgpt2"):
    """
    Save center embeddings and masked data for each sentence
    """
    print(f"Saving embeddings and masks for {len(sentences)} sentences...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    
    for i, sentence in enumerate(sentences):
        print(f"\nProcessing sentence {i+1}: '{sentence}'")
        
        # Get original embeddings
        embeddings, token_ids, token_texts = get_token_embeddings(sentence, model, tokenizer, device)
        
        print(f"  Tokens ({len(token_texts)}): {token_texts}")
        print(f"  Embedding shape: {embeddings.shape}")  # [seq_len, d]
        
        # Create safe filename
        safe_name = sentence.replace(" ", "_").replace(".", "").replace(",", "").replace("!", "").replace("?", "")[:50]
        
        # Save center embeddings (original token embeddings)
        center_data = {
            'sentence': sentence,
            'tokens': np.array(token_texts, dtype=object),
            'token_ids': token_ids.cpu().numpy(),
            'embedding_dim': np.array(embeddings.shape[1]),
            'num_tokens': np.array(embeddings.shape[0]),
            'model_name': model_name
        }
        
        center_file = os.path.join(output_dir, f"center_embeddings_{safe_name}.npz")
        np.savez_compressed(center_file, **center_data)
        # Save embeddings separately for easier loading
        embeddings_file = os.path.join(output_dir, f"center_embeddings_{safe_name}_tensor.pt")
        torch.save(embeddings.cpu(), embeddings_file)
        
        print(f"  Saved center embeddings: {center_file}")
        print(f"  Saved embedding tensor: {embeddings_file}")
        
        # Generate all possible masks (subsets) up to k=2
        n_tokens = embeddings.shape[0]
        all_subsets = generate_all_subsets(n_tokens, max_k=2)
        
        print(f"  Generating {len(all_subsets)} masked versions...")
        
        # Save masked data for both baselines
        for baseline in ["zero", "mean"]:
            masked_data = {
                'sentence': sentence,
                'tokens': np.array(token_texts, dtype=object),
                'token_ids': token_ids.cpu().numpy(),
                'baseline': baseline,
                'num_tokens': np.array(n_tokens),
                'embedding_dim': np.array(embeddings.shape[1]),
                'model_name': model_name,
                'subsets': np.array(all_subsets, dtype=object),
                'num_subsets': np.array(len(all_subsets))
            }
            
            # Create masked embeddings for all subsets
            masked_embeddings_list = []
            for subset in tqdm(all_subsets, desc=f"  Creating {baseline} masks"):
                masked_emb = create_masked_embeddings(embeddings, subset, baseline)
                masked_embeddings_list.append(masked_emb.cpu().numpy())
            
            # Convert to tensor for saving
            masked_embeddings_tensor = torch.stack([torch.from_numpy(emb) for emb in masked_embeddings_list])
            
            # Save masked data
            masked_file = os.path.join(output_dir, f"masked_data_{safe_name}_{baseline}.npz")
            
            # Save metadata only (without embeddings)
            np.savez_compressed(masked_file, **masked_data)
            
            # Save masked embeddings tensor
            masked_tensor_file = os.path.join(output_dir, f"masked_data_{safe_name}_{baseline}_tensor.pt")
            torch.save(masked_embeddings_tensor, masked_tensor_file)
            
            print(f"  Saved {baseline} masked data: {masked_file}")
            print(f"  Saved {baseline} tensor: {masked_tensor_file}")
            print(f"    Shape: {masked_embeddings_tensor.shape}")
        
        # Save metadata JSON for easy inspection
        metadata_file = os.path.join(output_dir, f"metadata_{safe_name}.json")
        metadata = {
            'sentence': sentence,
            'tokens': token_texts,
            'num_tokens': n_tokens,
            'embedding_dim': embeddings.shape[1],
            'num_subsets': len(all_subsets),
            'max_k': 2,
            'subsets_k1': [s for s in all_subsets if len(s) == 1],
            'subsets_k2': [s for s in all_subsets if len(s) == 2],
            'files': {
                'center_embeddings': f"center_embeddings_{safe_name}.npz",
                'center_tensor': f"center_embeddings_{safe_name}_tensor.pt",
                'masked_zero': f"masked_data_{safe_name}_zero.npz",
                'masked_zero_tensor': f"masked_data_{safe_name}_zero_tensor.pt",
                'masked_mean': f"masked_data_{safe_name}_mean.npz",
                'masked_mean_tensor': f"masked_data_{safe_name}_mean_tensor.pt"
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved metadata: {metadata_file}")

def create_embedding_loader_utils():
    """Create utility functions for loading the saved embeddings"""
    
    loader_code = '''"""
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
        print(f"\\nLoading data for: {prefix}")
        
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
'''
    
    # Save the loader utilities
    with open("gpt_explain/embedding_loader_utils.py", 'w') as f:
        f.write(loader_code)
    
    print("Created embedding_loader_utils.py")

def main():
    """Main function to save embeddings and masks for all sentences"""
    
    # Define the sentences to process
    sentences = [
        "The food was cheap, fresh, and tasty.",
        "The test was easy and simple.",
        "The product is not very reliable.",
        "Great, just what I needed!"
    ]
    
    print("="*80)
    print("SAVING GPT EMBEDDINGS AND MASKED DATA")
    print("="*80)
    print(f"Processing {len(sentences)} sentences")
    print(f"Model: distilgpt2")
    print(f"Max subset size: k=2")
    print(f"Baselines: zero, mean")
    print("="*80)
    
    # Save embeddings and masks
    save_embeddings_and_masks(sentences)
    
    # Create utility functions
    create_embedding_loader_utils()
    
    print("\n" + "="*80)
    print("EMBEDDING AND MASK SAVING COMPLETED")
    print("="*80)
    
    # List created files
    print("\nCreated files:")
    for file in sorted(os.listdir("gpt_explain")):
        if any(file.startswith(prefix) for prefix in ["center_embeddings_", "masked_data_", "metadata_"]):
            print(f"  {file}")
    
    print(f"\nUtility file created:")
    print(f"  embedding_loader_utils.py")
    
    print(f"\nTo load data, use:")
    print(f"  from embedding_loader_utils import load_center_embeddings, load_masked_data")

if __name__ == "__main__":
    main()
