import torch
from transformers import GPT2LMHeadModel

def convert_huggingface_to_megatron(hf_checkpoint_path, megatron_checkpoint_path, model_type='gpt2'):
    """
    Convert Hugging Face GPT-2 checkpoint to Megatron-LM checkpoint format.
    Args:
        hf_checkpoint_path (str): Path to the Hugging Face checkpoint (pytorch_model.bin).
        megatron_checkpoint_path (str): Path to save the converted Megatron-LM checkpoint.
        model_type (str): Model type, default is 'gpt2'.
    """
    print("Loading Hugging Face checkpoint...")
    hf_model = GPT2LMHeadModel.from_pretrained(hf_checkpoint_path)
    hf_state_dict = hf_model.state_dict()

    print("Converting checkpoint to Megatron-LM format...")
    megatron_state_dict = {}
    
    # Mapping Hugging Face layers to Megatron-LM layers
    for key, value in hf_state_dict.items():
        # Replace Hugging Face naming with Megatron-LM naming
        new_key = key.replace("transformer.h", "layers").replace("transformer.", "")
        new_key = new_key.replace("ln_f", "final_layernorm").replace("ln_", "input_layernorm.")
        new_key = new_key.replace("attn.c_proj", "attention.output")
        new_key = new_key.replace("attn.q_proj", "attention.query_key_value")  # Combine Q, K, V
        new_key = new_key.replace("mlp.c_fc", "mlp.dense_h_to_4h")
        new_key = new_key.replace("mlp.c_proj", "mlp.dense_4h_to_h")
        megatron_state_dict[new_key] = value

    print(f"Saving converted checkpoint to {megatron_checkpoint_path}...")
    torch.save(megatron_state_dict, megatron_checkpoint_path)
    print("Conversion complete!")

# Example usage
convert_huggingface_to_megatron(
    hf_checkpoint_path="./gpt2_small",
    megatron_checkpoint_path="./megatron_gpt2_small/checkpoint.pth",
)
