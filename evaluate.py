import argparse
import torch
from peft import AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_text(
    prompts,
    peft_model_id=None,
    model_id=None,
    max_length=200,
    num_beams=5,
    batch_size=1,
    do_sample=False,
    temperature=1.0,
):
    if peft_model_id is not None:
        # Load Model with PEFT adapter
        model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_id, device_map="auto", torch_dtype=torch.float16
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
        except Exception:
            print("Tokenizer not found, using base model tokenizer")
            peft_config = PeftConfig.from_pretrained(peft_model_id)
            tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    else:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Batch prompts
    batched_prompts = [
        prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
    ]

    device = model.device

    for prompt_batch in batched_prompts:
        # Prepare input
        templated_prompts = [
            tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in prompt_batch
        ]
        encoded_inputs = tokenizer.encode_batch(
            templated_prompts, return_tensors="pt", padding=True
        )
        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)

        # Generate text
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=temperature,  # Add the temperature option here
            use_cache=True,
        )

        # Decode output
        for idx, prompt in enumerate(prompt_batch):
            output_text = tokenizer.decode(output_ids[idx], skip_special_tokens=True)
            print(f"Prompt: {prompt}")
            print(f"Output: {output_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts", nargs="+", required=True, help="List of prompt strings"
    )
    parser.add_argument(
        "--peft_model_id", required=False, default=None, help="Path to saved PEFT adapter model (either peft_model_id or model_id is required)"
    )
    parser.add_argument(
        "--model_id", default=None, help="Path to saved PEFT adapter model (either peft_model_id or model_id is required)"
    )
    parser.add_argument(
        "--max_length", type=int, default=200, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--num_beams", type=int, default=5, help="Number of beams for beam search"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for prompt strings"
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="Use sampling for text generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling"
    )
    args = parser.parse_args()

    if args.peft_model_id is None and args.model_id is None:
        raise ValueError("Either peft_model_id or model_id must be provided")

    generate_text(
        args.prompts,
        args.peft_model_id,
        args.model_id,
        args.max_length,
        args.num_beams,
        args.batch_size,
        args.do_sample,
        args.temperature,
    )
