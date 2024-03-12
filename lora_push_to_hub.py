import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftModel, PeftConfig


def main(args):
    model_path = args.model_path
    save_path = args.save_path
    repo_id = args.repo_id
    merge_adapter = args.merge_adapter
    max_shard_size = args.max_shard_size
    tokenizer_path = args.tokenizer_path

    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path is not None else base_model_name
    )

    # base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    # model = PeftModel.from_pretrained(base_model, model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    if merge_adapter:
        print("Merging adapter into the base model")
        model = model.merge_and_unload()
        if save_path is not None:
            print(
                f"Saving merged model to {save_path} and pushing to the Hub at {repo_id}"
            )
            model.save_pretrained(
                save_path,
                max_shard_size=max_shard_size,
                push_to_hub=True,
                repo_id=repo_id,
                private=False,
            )
            tokenizer.save_pretrained(
                save_path, push_to_hub=True, repo_id=repo_id, private=False
            )
            print(
                f"Merged model and tokenizer saved locally to {save_path} and pushed to {repo_id}"
            )
            return

        print("Pushing merged model and tokenizer to the Hub")
        model.push_to_hub(
            repo_id=repo_id, commit_message="Upload merged model", private=False
        )
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message="Upload tokenizer",
            private=False,
        )
        print(f"Merged model and tokenizer pushed to {repo_id}")
    else:
        print("Pushing PEFT adapter and tokenizer to the Hub")
        model.push_to_hub(
            repo_id=repo_id, commit_message="Upload PEFT adapter", private=False
        )
        tokenizer.push_to_hub(
            repo_id=repo_id, commit_message="Upload tokenizer", private=False
        )
        print(f"PEFT adapter and tokenizer pushed to {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload PEFT adapter or merged model to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the directory containing the PEFT adapter files",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the directory containing the tokenizer files. If not provided, the tokenizer will be loaded from the base model of PEFT adaptors on Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face Hub (in the format 'username/repo_name')",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Local path to save the merged model",
    )
    parser.add_argument(
        "--merge_adapter",
        action="store_true",
        help="Merge the adapter into the base model before pushing to the Hub",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Maximum shard size when saving the merged model (default: 2GB)",
    )
    args = parser.parse_args()
    main(args)
