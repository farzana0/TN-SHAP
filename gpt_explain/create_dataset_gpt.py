import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Load model ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilgpt2"   # faster than gpt2, works well for demos
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
embedding_matrix = model.get_input_embeddings().weight.data  # [vocab_size, d]

# ---- Baselines ----
E_zero = torch.zeros_like(embedding_matrix[0])             # [d]
E_mean = embedding_matrix.mean(dim=0)                      # [d]

def get_logits_with_mask(prompt, mask_positions=None, baseline="zero", target_token=" great"):
    """
    prompt: str, e.g. "The movie was very good"
    mask_positions: list of token indices (after tokenizer) to replace
    baseline: "zero" or "mean"
    target_token: which next-token logit to score
    """
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)      # [1, n]
    n = input_ids.shape[1]
    
    # Get embeddings
    embeds = model.get_input_embeddings()(input_ids)  # [1, n, d]
    
    # Choose baseline vector
    base_vec = E_zero if baseline == "zero" else E_mean
    
    if mask_positions is not None:
        for pos in mask_positions:
            embeds[0, pos, :] = base_vec
    
    # Forward pass using inputs_embeds
    outputs = model(inputs_embeds=embeds)
    logits = outputs.logits  # [1, n, vocab_size]
    
    # Score probability of chosen target token at final position
    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    probs = torch.softmax(logits[0, -1], dim=-1)
    return probs[target_id].item(), logits[0, -1, target_id].item()

# ---- Example usage ----
prompt = "The movie was very good"
for baseline in ["zero", "mean"]:
    prob, logit = get_logits_with_mask(prompt, mask_positions=[2,3], baseline=baseline, target_token=" great")
    print(f"{baseline} baseline, prob of 'great': {prob:.4f}, logit {logit:.2f}")
