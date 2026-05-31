#!/usr/bin/env python3
"""
Synthetic Dataset Creator for GPT Token Embeddings with Masking Logic

This script creates synthetic datasets from GPT token embeddings using:
1. Pretrained DistilGPT-2 model
2. Token-level embeddings as input features
3. Neighborhood sampling with Gaussian noise
4. Masking logic similar to semi_global_100pt (k=1, k=2 subsets only)
5. Target prediction based on next-token probabilities

For each sentence, creates:
- Center embedding (original sentence tokens)
- Masked datasets for k=1 and k=2 subsets
- Noisy neighborhood samples for upsampling
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import combinations
import json
from typing import List, Dict, Tuple, Optional
import argparse
import warnings

# Suppress tokenizer warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class GPTSyntheticDatasetCreator:
    def __init__(self, model_name: str = "distilgpt2", device: str = None):
        """Initialize the GPT synthetic dataset creator."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get embedding matrix
        self.embedding_matrix = self.model.get_input_embeddings().weight.data  # [vocab_size, d]
        self.embed_dim = self.embedding_matrix.shape[1]
        
        # Baseline embeddings
        self.E_zero = torch.zeros_like(self.embedding_matrix[0])
        self.E_mean = self.embedding_matrix.mean(dim=0)
        
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Vocabulary size: {self.embedding_matrix.shape[0]}")

    def tokenize_sentence(self, sentence: str) -> Dict:
        """Tokenize a sentence and return token information."""
        tokens = self.tokenizer(sentence, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)  # Remove batch dimension
        
        # Get token strings for debugging
        token_strings = [self.tokenizer.decode([tid]) for tid in input_ids]
        
        return {
            "sentence": sentence,
            "input_ids": input_ids,
            "token_strings": token_strings,
            "n_tokens": len(input_ids)
        }

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for input token ids."""
        return self.model.get_input_embeddings()(input_ids.to(self.device))

    def get_target_score(self, sentence: str, mask_positions: List[int] = None, 
                        baseline: str = "zero", target_token: str = " great") -> Tuple[float, float]:
        """
        Get target score (probability and logit) for next token prediction.
        
        Args:
            sentence: Input sentence
            mask_positions: Token positions to mask (replace with baseline)
            baseline: "zero" or "mean"
            target_token: Target token to predict
            
        Returns:
            (probability, logit) for the target token
        """
        tokens = self.tokenizer(sentence, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)
        
        # Get embeddings
        embeds = self.model.get_input_embeddings()(input_ids)  # [1, n, d]
        
        # Choose baseline vector
        base_vec = self.E_zero if baseline == "zero" else self.E_mean
        
        # Apply masking
        if mask_positions is not None:
            for pos in mask_positions:
                if pos < embeds.shape[1]:  # Check bounds
                    embeds[0, pos, :] = base_vec
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs_embeds=embeds)
            logits = outputs.logits  # [1, n, vocab_size]
        
        # Score probability of target token at final position
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        probs = torch.softmax(logits[0, -1], dim=-1)
        prob = probs[target_id].item()
        logit = logits[0, -1, target_id].item()
        
        return prob, logit

    def generate_all_subsets(self, n_tokens: int, max_k: int = 2) -> Dict[int, List[Tuple[int, ...]]]:
        """Generate all subsets up to order k for masking."""
        subsets = {0: [()]}  # Empty subset (no masking)
        
        for k in range(1, min(max_k + 1, n_tokens + 1)):
            subsets[k] = list(combinations(range(n_tokens), k))
        
        return subsets

    def add_gaussian_noise(self, embeddings: torch.Tensor, noise_std: float = 0.1, 
                          n_samples: int = 10) -> torch.Tensor:
        """Add Gaussian noise to embeddings for neighborhood sampling."""
        # embeddings: [seq_len, embed_dim]
        noisy_samples = []
        
        for _ in range(n_samples):
            noise = torch.randn_like(embeddings) * noise_std
            noisy_embed = embeddings + noise
            noisy_samples.append(noisy_embed)
        
        return torch.stack(noisy_samples)  # [n_samples, seq_len, embed_dim]

    def create_masked_dataset(self, sentence: str, target_token: str = " great", 
                            noise_std: float = 0.1, n_noise_samples: int = 10,
                            baseline: str = "zero") -> Dict:
        """
        Create a complete masked dataset for a sentence.
        
        Returns:
            Dictionary containing:
            - original embeddings
            - masked datasets for k=1, k=2
            - target scores for each configuration
            - metadata
        """
        print(f"Processing sentence: '{sentence}'")
        
        # Tokenize sentence
        token_info = self.tokenize_sentence(sentence)
        input_ids = token_info["input_ids"]
        n_tokens = token_info["n_tokens"]
        
        print(f"  Tokens ({n_tokens}): {token_info['token_strings']}")
        
        # Get original embeddings
        original_embeds = self.get_embeddings(input_ids)  # [n_tokens, embed_dim]
        
        # Detach from computation graph
        original_embeds = original_embeds.detach()
        
        # Generate subsets for masking (k=0, k=1, k=2)
        subsets = self.generate_all_subsets(n_tokens, max_k=2)
        
        # Storage for results
        dataset = {
            "sentence": sentence,
            "target_token": target_token,
            "token_info": token_info,
            "original_embeddings": original_embeds.cpu().numpy(),
            "baseline": baseline,
            "noise_std": noise_std,
            "n_noise_samples": n_noise_samples,
            "subsets": subsets,
            "data": {}
        }
        
        # Process each order (k=0, k=1, k=2)
        for k in sorted(subsets.keys()):
            print(f"  Processing order k={k} ({len(subsets[k])} subsets)")
            
            order_data = {
                "k": k,
                "subsets": subsets[k],
                "samples": []
            }
            
            # Process each subset of this order
            for subset_idx, subset in enumerate(subsets[k]):
                print(f"    Subset {subset_idx+1}/{len(subsets[k])}: {subset}")
                
                # Get target score for this masking configuration
                prob, logit = self.get_target_score(sentence, list(subset), baseline, target_token)
                
                # Create base sample (original + masking)
                base_embeds = original_embeds.clone()
                base_vec = self.E_zero if baseline == "zero" else self.E_mean
                
                # Apply masking
                for pos in subset:
                    base_embeds[pos] = base_vec
                
                # Detach base embeddings
                base_embeds = base_embeds.detach()
                
                # Add noise samples for neighborhood upsampling
                if n_noise_samples > 0:
                    noisy_embeds = self.add_gaussian_noise(base_embeds, noise_std, n_noise_samples)
                    all_embeds = torch.cat([base_embeds.unsqueeze(0), noisy_embeds])  # [1+n_noise, seq_len, embed_dim]
                else:
                    all_embeds = base_embeds.unsqueeze(0)  # [1, seq_len, embed_dim]
                
                # Create sample data
                sample_data = {
                    "subset": subset,
                    "subset_size": len(subset),
                    "target_prob": prob,
                    "target_logit": logit,
                    "embeddings": all_embeds.cpu().numpy(),  # [n_total_samples, seq_len, embed_dim]
                    "n_samples": all_embeds.shape[0]
                }
                
                order_data["samples"].append(sample_data)
            
            dataset["data"][k] = order_data
        
        return dataset

    def save_dataset(self, dataset: Dict, output_path: str):
        """Save dataset to file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_dataset = convert_numpy(dataset)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_dataset, f, indent=2)
        
        print(f"Dataset saved to: {output_path}")

    def create_summary_stats(self, dataset: Dict) -> Dict:
        """Create summary statistics for a dataset."""
        stats = {
            "sentence": dataset["sentence"],
            "n_tokens": dataset["token_info"]["n_tokens"],
            "token_strings": dataset["token_info"]["token_strings"],
            "embed_dim": self.embed_dim,
            "orders": {}
        }
        
        for k, order_data in dataset["data"].items():
            n_subsets = len(order_data["subsets"])
            total_samples = sum(sample["n_samples"] for sample in order_data["samples"])
            
            # Get target score statistics
            target_probs = [sample["target_prob"] for sample in order_data["samples"]]
            target_logits = [sample["target_logit"] for sample in order_data["samples"]]
            
            stats["orders"][k] = {
                "n_subsets": n_subsets,
                "total_samples": total_samples,
                "target_prob_mean": np.mean(target_probs),
                "target_prob_std": np.std(target_probs),
                "target_logit_mean": np.mean(target_logits),
                "target_logit_std": np.std(target_logits)
            }
        
        return stats


def main():
    # Test sentences
    sentences = [
        "The food was cheap, fresh, and tasty.",
        "The test was easy and simple.", 
        "The product is not very reliable.",
        "Great, just what I needed"
    ]
    
    # Configuration
    target_token = " great"  # What we're trying to predict
    noise_std = 0.1  # Standard deviation for Gaussian noise
    n_noise_samples = 5  # Number of noisy samples per original sample
    baseline = "zero"  # Use zero baseline for masking
    
    # Create output directory
output_dir = "./gpt_explain"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize creator
    creator = GPTSyntheticDatasetCreator()
    
    # Process each sentence
    all_stats = {}
    
    for i, sentence in enumerate(sentences):
        print(f"\n{'='*60}")
        print(f"Processing sentence {i+1}/{len(sentences)}")
        print(f"{'='*60}")
        
        # Create dataset
        dataset = creator.create_masked_dataset(
            sentence=sentence,
            target_token=target_token,
            noise_std=noise_std,
            n_noise_samples=n_noise_samples,
            baseline=baseline
        )
        
        # Save dataset
        filename = f"sentence_{i+1}_dataset.json"
        output_path = os.path.join(output_dir, filename)
        creator.save_dataset(dataset, output_path)
        
        # Create and store stats
        stats = creator.create_summary_stats(dataset)
        all_stats[f"sentence_{i+1}"] = stats
        
        print(f"Summary for sentence {i+1}:")
        print(f"  Tokens: {stats['n_tokens']}")
        for k, order_stats in stats["orders"].items():
            print(f"  Order k={k}: {order_stats['n_subsets']} subsets, {order_stats['total_samples']} total samples")
            print(f"    Target prob: {order_stats['target_prob_mean']:.4f} ± {order_stats['target_prob_std']:.4f}")
    
    # Save combined statistics
    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ALL DATASETS CREATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Statistics saved to: {stats_path}")
    print(f"Individual dataset files:")
    for i in range(len(sentences)):
        print(f"  sentence_{i+1}_dataset.json")


if __name__ == "__main__":
    main()
