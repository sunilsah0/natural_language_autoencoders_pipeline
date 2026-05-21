import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActivationHook:
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
    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.hidden_size = self.model.config.hidden_size
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward_generate(self, activation_vector, max_new_tokens=32):
        prompt = "The internal state of the model represents the concept of:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        embeddings = self.model.get_input_embeddings()(input_ids)
        
        # Cast activation vector to match embedding dtype (bfloat16)
        act_norm = activation_vector / (torch.norm(activation_vector, p=2, dim=-1, keepdim=True) + 1e-8)
        act_unsqueezed = act_norm.view(1, 1, self.hidden_size).to(device=self.device, dtype=embeddings.dtype)
        
        injected_embeddings = torch.cat([embeddings, act_unsqueezed], dim=1)
        
        # Extend attention mask for the injected vector token
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
    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.hidden_size = self.model.config.hidden_size
        
        # Dynamic casting: Ensure linear layer matches the model's weight type (BFloat16)
        self.reconstruction_head = nn.Linear(self.hidden_size, self.hidden_size)
        self.reconstruction_head = self.reconstruction_head.to(device=device, dtype=self.model.dtype)
        
    def forward(self, text_explanation):
        inputs = self.tokenizer(text_explanation, return_tensors="pt").to(self.device)
        outputs = self.model.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        final_token_hidden = outputs.last_hidden_state[:, -1, :]
        reconstructed_vector = self.reconstruction_head(final_token_hidden)
        
        reconstructed_norm = reconstructed_vector / (torch.norm(reconstructed_vector, p=2, dim=-1, keepdim=True) + 1e-8)
        return reconstructed_norm