import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. MECHANISTIC INTERPRETABILITY COMPONENTS
# ==========================================

class ActivationHook:
    """
    Hooks directly into the target LLM's residual stream layer
    and caches the forward-pass activation tensor.
    """
    def __init__(self, model, layer_idx):
        self.hook = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)
        self.stored_activation = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.stored_activation = output[0].detach()
        else:
            self.stored_activation = output.detach()

    def remove(self):
        self.hook.remove()


class ActivationVerbalizer(nn.Module):
    """
    AV Engine: Injects raw numerical activations directly into text prompt
    embeddings to generate an explanatory natural language string.
    """
    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.hidden_size = self.model.config.hidden_size
        
        # Freeze target model layers to reduce runtime overhead
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward_generate(self, activation_vector, max_new_tokens=16):
        prompt = "The internal state of the model represents the concept of:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Pull raw tokens embeddings from base matrix
        embeddings = self.model.get_input_embeddings()(input_ids)
        
        # Scale & cast vector to match base embedding precision (BFloat16)
        act_norm = activation_vector / (torch.norm(activation_vector, p=2, dim=-1, keepdim=True) + 1e-8)
        act_unsqueezed = act_norm.view(1, 1, self.hidden_size).to(device=self.device, dtype=embeddings.dtype)
        
        # Splice raw activation into embedding sequence
        injected_embeddings = torch.cat([embeddings, act_unsqueezed], dim=1)
        
        # Pad mask array to account for the newly appended pseudo-token
        ones = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=self.device)
        extended_attention_mask = torch.cat([attention_mask, ones], dim=1)
        
        outputs = self.model.generate(
            inputs_embeds=injected_embeddings,
            attention_mask=extended_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ActivationReconstructor(nn.Module):
    """
    AR Engine: Evaluates the text summary string and projects the final
    token hidden state back into an estimation of the original activation space.
    """
    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.hidden_size = self.model.config.hidden_size
        
        # Dynamic Casting Fix: Initialize projection layer to match model's precision
        self.reconstruction_head = nn.Linear(self.hidden_size, self.hidden_size)
        self.reconstruction_head = self.reconstruction_head.to(device=device, dtype=self.model.dtype)
        
    def forward(self, text_explanation):
        inputs = self.tokenizer(text_explanation, return_tensors="pt").to(self.device)
        outputs = self.model.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        final_token_hidden = outputs.last_hidden_state[:, -1, :]
        reconstructed_vector = self.reconstruction_head(final_token_hidden)
        
        reconstructed_norm = reconstructed_vector / (torch.norm(reconstructed_vector, p=2, dim=-1, keepdim=True) + 1e-8)
        return reconstructed_norm

# ==========================================
# 2. RUNTIME ORCHESTRATION PIPELINE
# ==========================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing pipeline on device: {device}")

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading target model: {model_name}")
    target_tokenizer = AutoTokenizer.from_pretrained(model_name)
    target_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Layer 16 extraction (~2/3 deep into Qwen's residual stack)
    target_layer = 16
    hook = ActivationHook(target_model, target_layer)

    print("Extracting real activations from WikiText validation splits...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    extracted_activations = []
    sample_count = 66  # Match your existing sample cache ceiling
    
    for idx, item in enumerate(dataset):
        if len(item["text"].strip()) < 40 or len(extracted_activations) >= sample_count:
            continue
        inputs = target_tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=128).to(device)
        if inputs["input_ids"].shape[1] < 10:
            continue
        with torch.no_grad():
            target_model(**inputs)
        
        act = hook.stored_activation[:, -1, :].clone().cpu()
        extracted_activations.append(act)
        
    hook.remove()
    activations_tensor = torch.cat(extracted_activations, dim=0)
    print(f"Successfully cached {activations_tensor.shape[0]} activations of dimension {activations_tensor.shape[1]}")

    # Instantiate NLA Architecture
    av = ActivationVerbalizer(model_name, device)
    ar = ActivationReconstructor(model_name, device)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(ar.reconstruction_head.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
    print("Beginning optimization routine for Activation Reconstructor Head...")
    ar.train()
    
    split_idx = int(0.8 * len(activations_tensor))
    train_acts = activations_tensor[:split_idx]
    test_acts = activations_tensor[split_idx:]
    
    train_pairs = []
    for i in range(len(train_acts)):
        act_vec = train_acts[i].to(device)
        explanation = av.forward_generate(act_vec, max_new_tokens=16)
        train_pairs.append((act_vec, explanation))

    for epoch in range(5):
        epoch_losses = []
        for orig_act, exp_text in train_pairs:
            optimizer.zero_grad()
            
            target_norm = orig_act / (torch.norm(orig_act, p=2, dim=-1, keepdim=True) + 1e-8)
            # Mixed Precision Fix: Force alignment with the BFloat16 Reconstructor stream
            target_norm = target_norm.to(device=device, dtype=ar.model.dtype).unsqueeze(0)
            
            recon_norm = ar(exp_text)
            loss = loss_fn(recon_norm, target_norm)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch+1}/5 | Average MSE Loss: {np.mean(epoch_losses):.5f}")

    # ==========================================
    # 3. FIXED QUANTITATIVE EVALUATION PHASE
    # ==========================================
    print("\n--- Phase 5: Quantitative Metric Evaluation ---")
    ar.eval()
    mse_records = []
    cosine_records = []
    
    test_acts_norm = []
    for act in test_acts:
        n_act = act / (torch.norm(act, p=2, dim=-1, keepdim=True) + 1e-8)
        # NumPy Casting Fix: Cast out of BFloat16 to regular Float32 before .numpy()
        test_acts_norm.append(n_act.float().numpy())
        
    test_acts_norm = np.array(test_acts_norm)
    mean_vector = np.mean(test_acts_norm, axis=0)
    
    total_variance = np.sum((test_acts_norm - mean_vector) ** 2)
    unexplained_variance = 0.0

    print("\nQualitative Tracking Window:")
    for idx, orig_act in enumerate(test_acts):
        act_vec = orig_act.to(device)
        explanation = av.forward_generate(act_vec, max_new_tokens=16)
        
        with torch.no_grad():
            recon_norm_tensor = ar(explanation).cpu().squeeze(0)
            # NumPy Casting Fix: Cast out of BFloat16 to regular Float32 before .numpy()
            recon_norm = recon_norm_tensor.float().numpy()
        
        orig_norm = test_acts_norm[idx]
        
        mse = np.mean((orig_norm - recon_norm) ** 2)
        cosine_sim = np.dot(orig_norm, recon_norm) / (np.linalg.norm(orig_norm) * np.linalg.norm(recon_norm) + 1e-8)
        
        mse_records.append(mse)
        cosine_records.append(cosine_sim)
        unexplained_variance += np.sum((orig_norm - recon_norm) ** 2)
        
        if idx < 3:
            print(f"  [Sample {idx+1}] Generated Explanation: \"{explanation.strip()}\" | Cosine Similarity: {cosine_sim:.4f}")

    fve = 1.0 - (unexplained_variance / total_variance)
    print(f"\nFinal Quantitative Summary Metric:")
    print(f"  Mean Cosine Similarity:       {np.mean(cosine_records):.4f}")
    print(f"  Mean Squared Error (MSE):      {np.mean(mse_records):.6f}")
    print(f"  Fraction of Variance Explained (FVE): {fve:.4f}")

if __name__ == "__main__":
    main()