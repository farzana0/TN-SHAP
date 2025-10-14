#!/usr/bin/env python3
"""
Test script to verify that saved embeddings and masked data can be loaded correctly.
"""

import torch
import numpy as np
import json
from embedding_loader_utils import load_center_embeddings, load_masked_data, load_sentence_metadata, list_available_sentences

def test_loading():
    """Test loading all saved embeddings and masked data"""
    print("="*80)
    print("TESTING SAVED EMBEDDINGS AND MASKED DATA")
    print("="*80)
    
    # List available sentences
    sentences = list_available_sentences()
    print(f"Available sentences: {len(sentences)}")
    for i, sentence in enumerate(sentences):
        print(f"  {i+1}. {sentence}")
    
    print("\n" + "="*80)
    
    # Test each sentence
    for sentence_prefix in sentences:
        print(f"\nTesting: {sentence_prefix}")
        print("-" * 60)
        
        try:
            # Load metadata
            metadata = load_sentence_metadata(sentence_prefix)
            print(f"Original sentence: {metadata['sentence']}")
            print(f"Tokens: {metadata['tokens']}")
            print(f"Number of tokens: {metadata['num_tokens']}")
            print(f"Embedding dimension: {metadata['embedding_dim']}")
            print(f"Number of subsets: {metadata['num_subsets']}")
            
            # Load center embeddings
            center_data = load_center_embeddings(sentence_prefix)
            print(f"Center embeddings shape: {center_data['embeddings'].shape}")
            print(f"Device: {center_data['embeddings'].device}")
            print(f"Data type: {center_data['embeddings'].dtype}")
            
            # Check that tokens match
            assert list(center_data['tokens']) == metadata['tokens'], "Tokens mismatch between center data and metadata!"
            
            # Load masked data for both baselines
            for baseline in ["zero", "mean"]:
                masked_data = load_masked_data(sentence_prefix, baseline)
                print(f"Masked embeddings ({baseline}) shape: {masked_data['masked_embeddings'].shape}")
                print(f"Number of subsets ({baseline}): {len(masked_data['subsets'])}")
                
                # Verify shapes
                expected_shape = (metadata['num_subsets'], metadata['num_tokens'], metadata['embedding_dim'])
                actual_shape = tuple(masked_data['masked_embeddings'].shape)
                assert actual_shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {actual_shape}"
                
                # Check that first few subsets are correct
                subsets = masked_data['subsets']
                print(f"First 5 subsets ({baseline}): {subsets[:5]}")
                
                # Verify that masking was applied correctly
                center_embeddings = center_data['embeddings']
                masked_embeddings = masked_data['masked_embeddings']
                
                # Test first subset (should mask position 0)
                first_subset = subsets[0]
                first_masked = masked_embeddings[0]  # [num_tokens, embedding_dim]
                
                # Check that the masked position is different from original
                if baseline == "zero":
                    # Should be zero vector
                    assert torch.allclose(first_masked[first_subset[0]], torch.zeros(metadata['embedding_dim'])), "Zero masking failed!"
                elif baseline == "mean":
                    # Should be mean vector
                    mean_vec = center_embeddings.mean(dim=0)
                    assert torch.allclose(first_masked[first_subset[0]], mean_vec, atol=1e-6), "Mean masking failed!"
                
                # Check that unmasked positions are the same
                for pos in range(metadata['num_tokens']):
                    if pos not in first_subset:
                        assert torch.allclose(first_masked[pos], center_embeddings[pos], atol=1e-6), f"Unmasked position {pos} changed!"
            
            print(f"✅ All tests passed for {sentence_prefix}")
            
        except Exception as e:
            print(f"❌ Error testing {sentence_prefix}: {e}")
            return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nSummary of saved data:")
    print(f"- {len(sentences)} sentences processed")
    print(f"- Center embeddings: num_tokens × embedding_dim")
    print(f"- Masked embeddings: num_subsets × num_tokens × embedding_dim")
    print(f"- Two baselines: zero and mean")
    print(f"- Subset sizes: k=1 and k=2")
    print(f"- All data verified and ready for use!")
    
    return True

def demonstrate_usage():
    """Demonstrate how to use the saved data"""
    print("\n" + "="*80)
    print("USAGE DEMONSTRATION")
    print("="*80)
    
    sentences = list_available_sentences()
    if not sentences:
        print("No sentences available!")
        return
    
    # Use first sentence as example
    sentence_prefix = sentences[0]
    print(f"Using sentence: {sentence_prefix}")
    
    # Load center embeddings
    center_data = load_center_embeddings(sentence_prefix)
    print(f"\\nCenter embeddings:")
    print(f"  Shape: {center_data['embeddings'].shape}")
    print(f"  Sentence: {center_data['sentence']}")
    print(f"  Tokens: {center_data['tokens']}")
    
    # Load masked data
    masked_data = load_masked_data(sentence_prefix, "zero")
    print(f"\\nMasked embeddings (zero baseline):")
    print(f"  Shape: {masked_data['masked_embeddings'].shape}")
    print(f"  Number of subsets: {len(masked_data['subsets'])}")
    print(f"  Subsets by size:")
    
    subsets = masked_data['subsets']
    k1_subsets = [s for s in subsets if len(s) == 1]
    k2_subsets = [s for s in subsets if len(s) == 2]
    
    print(f"    k=1: {len(k1_subsets)} subsets (e.g., {k1_subsets[:5]})")
    print(f"    k=2: {len(k2_subsets)} subsets (e.g., {k2_subsets[:5]})")
    
    print(f"\\nExample usage in Shapley value computation:")
    print(f"  center_emb = load_center_embeddings('{sentence_prefix}')")
    print(f"  masked_emb = load_masked_data('{sentence_prefix}', 'zero')")
    print(f"  # Use center_emb['embeddings'] as reference")
    print(f"  # Use masked_emb['masked_embeddings'][i] for subset i")
    print(f"  # Subset definition: masked_emb['subsets'][i]")

def main():
    """Main function"""
    success = test_loading()
    if success:
        demonstrate_usage()
    else:
        print("Testing failed! Please check the data files.")

if __name__ == "__main__":
    main()
