import argparse
from typing import Literal
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM


def merge_models(model_dir, save: bool = False):
    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    merged_model = model.merge_and_unload()

    if save:
        merged_model.save_pretrained(model_dir, safe_serialization=True, max_shard_size="2GB")
    return merged_model


def push_to_hub(hf_model_id, model=None, model_dir=None, mode: Literal["merge", "peft"] = "peft"):
    assert model or model_dir, "Either model or model_dir should be provided"

    if model is not None:
        model.push_to_hub(hf_model_id)
        return

    if mode == "peft":
        # Load PEFT model on CPU
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    model.push_to_hub(hf_model_id)

def main():
    parser = argparse.ArgumentParser(description='Merge LoRA and base model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--push', action='store_true', help='Push the model to the hub')
    parser.add_argument('--hf_model_id', type=str, help='The model id on the hub')
    parser.add_argument('--mode', type=str, default="peft", choices=["peft", "merge"], help='The mode to use for the model')
    parser.add_argument('--save', action='store_true', help='Save the model after merging')
    args = parser.parse_args()

    model = None
    if args.mode == "merge":
        model = merge_models(args.model_dir, save=args.save)
    if not args.push:
        return

    push_to_hub(args.hf_model_id, model, args.model_dir, args.mode)


if __name__ == "__main__":
    main()
