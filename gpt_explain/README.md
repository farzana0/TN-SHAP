# GPT Synthetic Dataset with Token Embedding Masking

This directory contains synthetic datasets created from GPT token embeddings using a masking approach similar to the `semi_global_100pt` methodology, adapted for natural language processing.

## Overview

The synthetic dataset generator creates neighborhood-sampled data from pretrained GPT models (DistilGPT-2) by:

1. **Token-level Analysis**: Each token in a sentence becomes an input feature
2. **Embedding Extraction**: Uses the pretrained embedding matrix to convert tokens to dense vectors
3. **Subset Masking**: Applies systematic masking to subsets of tokens (k=1, k=2 orders only)
4. **Neighborhood Sampling**: Adds Gaussian noise around each configuration for upsampling
5. **Target Prediction**: Uses next-token probability as the prediction target

## Test Sentences

The datasets are created for these 4 sentences:
1. **"The food was cheap, fresh, and tasty."**
2. **"The test was easy and simple."**
3. **"The product is not very reliable."** 
4. **"Great, just what I needed"**

Each sentence is analyzed to predict the likelihood of the next token being " great".

## Dataset Structure

### File Organization
```
gpt_explain/
├── create_gpt_synthetic_dataset.py     # Main dataset creation script
├── save_embeddings_and_masks.py        # Script to save reusable embeddings
├── embedding_loader_utils.py           # Helper functions to load embeddings
├── test_saved_embeddings.py            # Verification script for embeddings
├── sentence_1_dataset.json             # Dataset for sentence 1
├── sentence_2_dataset.json             # Dataset for sentence 2
├── sentence_3_dataset.json             # Dataset for sentence 3
├── sentence_4_dataset.json             # Dataset for sentence 4
├── dataset_statistics.json             # Combined statistics
├── README.md                           # This file
└── gpt_explain/                        # Subdirectory with reusable embeddings
    ├── center_embeddings_*.npz         # Center embedding metadata
    ├── center_embeddings_*_tensor.pt   # Center embeddings [tokens × 768]
    ├── masked_data_*_zero.npz          # Zero baseline metadata
    ├── masked_data_*_zero_tensor.pt    # Zero masked embeddings [subsets × tokens × 768]
    ├── masked_data_*_mean.npz          # Mean baseline metadata
    ├── masked_data_*_mean_tensor.pt    # Mean masked embeddings [subsets × tokens × 768]
    └── metadata_*.json                 # Complete metadata for each sentence
```

### Dataset Schema

Each `sentence_X_dataset.json` contains:

```json
{
  "sentence": "The food was cheap, fresh, and tasty.",
  "target_token": " great",
  "token_info": {
    "sentence": "The food was cheap, fresh, and tasty.",
    "input_ids": [...],           // Token IDs
    "token_strings": [...],       // Human-readable tokens
    "n_tokens": 8                 // Number of tokens
  },
  "original_embeddings": [...],   // [n_tokens, embed_dim] original embeddings
  "baseline": "zero",             // Masking baseline used
  "noise_std": 0.1,              // Gaussian noise standard deviation
  "n_noise_samples": 5,          // Number of noise samples per configuration
  "subsets": {                    // All subsets for each order
    "0": [[]],                    // k=0: no masking
    "1": [[0], [1], [2], ...],   // k=1: single token masking
    "2": [[0,1], [0,2], ...]     // k=2: pairs of tokens masking
  },
  "data": {                       // Actual dataset samples
    "0": { ... },                 // k=0 order data
    "1": { ... },                 // k=1 order data  
    "2": { ... }                  // k=2 order data
  }
}
```

### Order Data Structure

Each order `k` contains:
```json
{
  "k": 1,                        // Order (subset size)
  "subsets": [[0], [1], ...],    // All subsets of this size
  "samples": [                   // One entry per subset
    {
      "subset": [0],             // Which tokens are masked
      "subset_size": 1,          // Size of masked subset
      "target_prob": 0.0234,     // Probability of target token
      "target_logit": -3.45,     // Logit score for target token
      "embeddings": [...],       // [n_samples, n_tokens, embed_dim]
      "n_samples": 6             // 1 original + 5 noisy samples
    },
    ...
  ]
}
```

### Embedding Array Structure

The `embeddings` arrays have shape `[n_samples, n_tokens, embed_dim]` where:
- **n_samples**: 1 (original) + n_noise_samples (typically 5) = 6 total
- **n_tokens**: Number of tokens in the sentence  
- **embed_dim**: Embedding dimension (768 for DistilGPT-2)

For masked positions, the embedding is replaced with the baseline vector (zero or mean).

## Key Features

### 1. Systematic Masking
- **Order 0 (k=0)**: No masking - original sentence
- **Order 1 (k=1)**: Mask individual tokens one at a time  
- **Order 2 (k=2)**: Mask all pairs of tokens

### 2. Neighborhood Sampling
- Each masked configuration is augmented with Gaussian noise
- Default: 5 noisy samples per original sample
- Noise standard deviation: 0.1 (configurable)

### 3. Consistent Target
- All configurations predict the same target token (" great")
- Allows comparison of how different maskings affect prediction
- Uses both probability and logit scores

### 4. Baseline Masking
- Masked tokens replaced with learned baseline
- **Zero baseline**: Replace with zero vector
- **Mean baseline**: Replace with mean of embedding matrix

## 🔄 Using Reusable Embeddings

For efficient reuse in multiple experiments, center embeddings and masked data are saved separately:

### Loading Center Embeddings

```python
from embedding_loader_utils import load_center_embeddings

# Load original token embeddings
center_data = load_center_embeddings('The_food_was_cheap_fresh_and_tasty')
embeddings = center_data['embeddings']  # Shape: [10, 768]
tokens = center_data['tokens']          # Token strings
sentence = center_data['sentence']      # Original sentence
```

### Loading Masked Data

```python
from embedding_loader_utils import load_masked_data

# Load masked embeddings with zero baseline
masked_data = load_masked_data('The_food_was_cheap_fresh_and_tasty', 'zero')
masked_embeddings = masked_data['masked_embeddings']  # Shape: [55, 10, 768]
subsets = masked_data['subsets']                      # Subset definitions

# Access specific masked version
subset_idx = 0  # First subset (masks token 0)
masked_version = masked_embeddings[subset_idx]  # Shape: [10, 768]
masked_positions = subsets[subset_idx]          # [0] - positions that were masked
```

### Available Sentence Prefixes
- `The_food_was_cheap_fresh_and_tasty` (10 tokens, 55 subsets)
- `The_test_was_easy_and_simple` (7 tokens, 28 subsets)
- `The_product_is_not_very_reliable` (7 tokens, 28 subsets)
- `Great_just_what_I_needed` (7 tokens, 28 subsets)

## 🔧 Usage Examples

### Loading a Dataset
```python
import json
import numpy as np

# Load dataset
with open('sentence_1_dataset.json', 'r') as f:
    dataset = json.load(f)

# Access original embeddings
original_embeds = np.array(dataset['original_embeddings'])  # [n_tokens, embed_dim]

# Access k=1 order data  
k1_data = dataset['data']['1']
first_sample = k1_data['samples'][0]
sample_embeddings = np.array(first_sample['embeddings'])  # [n_samples, n_tokens, embed_dim]

print(f"Sentence: {dataset['sentence']}")
print(f"Tokens: {dataset['token_info']['token_strings']}")
print(f"Masked subset: {first_sample['subset']}")
print(f"Target probability: {first_sample['target_prob']:.4f}")
```

### Training Data Preparation
```python
# Collect all samples for training
X, y = [], []

for k in ['0', '1', '2']:  # For each order
    for sample in dataset['data'][k]['samples']:
        embeddings = np.array(sample['embeddings'])  # [n_samples, n_tokens, embed_dim]
        target_prob = sample['target_prob']
        
        # Flatten embeddings for each sample
        for i in range(embeddings.shape[0]):
            X.append(embeddings[i].flatten())  # [n_tokens * embed_dim]
            y.append(target_prob)

X = np.array(X)  # [total_samples, n_tokens * embed_dim]  
y = np.array(y)  # [total_samples]
```

## Statistics Summary

The `dataset_statistics.json` provides overview statistics:
- Number of tokens per sentence
- Subsets count per order
- Total samples generated
- Target probability statistics (mean, std)

Example statistics structure:
```json
{
  "sentence_1": {
    "sentence": "The food was cheap, fresh, and tasty.",
    "n_tokens": 8,
    "token_strings": ["The", " food", " was", ...],
    "embed_dim": 768,
    "orders": {
      "0": {"n_subsets": 1, "total_samples": 6, ...},
      "1": {"n_subsets": 8, "total_samples": 48, ...}, 
      "2": {"n_subsets": 28, "total_samples": 168, ...}
    }
  },
  ...
}
```

## Technical Details

### Model: DistilGPT-2
- **Architecture**: Distilled GPT-2 (faster inference)
- **Embedding dimension**: 768
- **Vocabulary size**: ~50,257 tokens
- **Context length**: 1024 tokens

### Masking Logic  
Based on `semi_global_100pt_end2end.py`:
- Generates all possible subsets up to order k=2
- Applies systematic masking to each subset
- Uses baseline embeddings (zero or mean) for masked positions

### Neighborhood Sampling
- Adds Gaussian noise: `embedding_noisy = embedding_original + N(0, σ²I)`
- Default σ = 0.1 (10% of typical embedding magnitude)
- Increases training data diversity and robustness

## Running the Generator

```bash
cd ./gpt_explain
python create_gpt_synthetic_dataset.py
```

### Configuration Options (edit in script):
- `sentences`: List of input sentences to process
- `target_token`: What token to predict (default: " great")  
- `noise_std`: Gaussian noise standard deviation (default: 0.1)
- `n_noise_samples`: Number of noisy samples per original (default: 5)
- `baseline`: Masking baseline - "zero" or "mean" (default: "zero")

## Applications

This synthetic dataset can be used for:

1. **Shapley Value Approximation**: Training surrogate models for token attribution
2. **Feature Importance**: Understanding which tokens contribute most to predictions
3. **Robustness Analysis**: Testing model behavior under token masking/corruption
4. **Interpretability Research**: Developing explainable NLP methods
5. **Benchmarking**: Comparing different attribution methods on controlled data

The dataset provides ground truth for how masking different token combinations affects the prediction target, enabling quantitative evaluation of explanation methods.
