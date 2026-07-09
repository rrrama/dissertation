import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LorTaConfig, get_peft_model
import json
from safetensors.torch import load_file as safe_load_file

def test_merge_and_unload():
    # Step 1: Load the pretrained base model and tokenizer
    model_name = "roberta-large"
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 2: Load the LorTaConfig from the adapter_config.json
    adapter_path = "output/model"  # Replace with your actual path
    with open(f"{adapter_path}/adapter_config.json", "r") as f:
        adapter_config_dict = json.load(f)
    lorta_config = LorTaConfig(**adapter_config_dict)

    # Step 3: Initialize the model with get_peft_model
    lorta_model = get_peft_model(base_model, lorta_config)
    lorta_model.eval()

    # Step 4: Load the adapter weights
    adapter_weights = safe_load_file(f"{adapter_path}/adapter_model.safetensors")

    # Manually load the adapter weights
    missing_keys, unexpected_keys = lorta_model.load_state_dict(adapter_weights, strict=False)
    #if missing_keys:
        #print(f"Missing keys when loading adapter weights: {missing_keys}")
    #if unexpected_keys:
        #print(f"Unexpected keys when loading adapter weights: {unexpected_keys}")

    # Prepare some sample input data
    input_text = "This is a test sentence for merging adapters."
    inputs = tokenizer(input_text, return_tensors="pt")

    # Move inputs to the same device as the model
    device = next(lorta_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Step 5: Run the model before merging and record the outputs
    with torch.no_grad(), lorta_model._enable_peft_forward_hooks():
        outputs_before = lorta_model(**inputs)
    logits_before = outputs_before.logits

    # Step 6: Merge and unload the adapters
    lorta_model.merge_and_unload()

    # Step 7: Load the weights of lorta_model into base_model
    base_model.load_state_dict(lorta_model.model.state_dict())

    # Step 8: Run the base model after loading the merged weights
    with torch.no_grad():
        outputs_after_merge = base_model(**inputs)
    logits_after_merge = outputs_after_merge.logits
    #breakpoint()

    # Step 9: Compare the outputs after merging
    if torch.allclose(logits_before, logits_after_merge, atol=1e-6):
        print("Test passed: Outputs are the same after merging and loading into base model.")
    else:
        print("Test failed: Outputs differ after merging and loading into base model.")

    # Print the maximum difference between outputs
    max_difference = (logits_before - logits_after_merge).abs().max().item()
    print(f"Maximum difference between outputs after merging and loading: {max_difference}")

# Run the test
test_merge_and_unload()
