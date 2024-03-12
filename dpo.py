"""
DPO training of <= 7B model on UltraFeedback dataset
Using HF DPOTrainer from TRL
Based on the script from the TRL repo and blog from Phillip Schmidt
https://www.philschmid.de/dpo-align-llms-in-2024-with-trl

This script is designed to be run from the command line, but can also be run interactively in a vscode jupyter session
or exported to a jupyter notebook and run as any .ipynb notebook.

"""
# %%
import argparse
import gc
import os
import time

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer


# %% [markdown] ################################################################
# ## Setup argparse for CLI and interactive jupyter notebook use

# We can set defaults here and then override them with argparse if we're running from the CLI
# below in the `if __name__ == "__main__":` block

def parse_arguments(input_args=None):

    DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    parser = argparse.ArgumentParser(description="DPO Alignment Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add arguments for environment variables
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="The model ID for the tokenizer and model")
    parser.add_argument("--dataset_id", type=str, default="argilla/ultrafeedback-binarized-preferences-cleaned",
                        help="The dataset ID for the training data (must comply with DPO expected format, see code and comments if changing)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="The number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The learning rate for training")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="The DPO beta factor for loss, controls divergence from the reference model, higher is less divergence")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="The output directory for saving the trained model")
    parser.add_argument("--run_name", type=str, default=f"dpo_{DEFAULT_MODEL_ID.replace('/', '-')}_{time.strftime('%Y%m%d%H%M')}",
                        help="The name of the training run")
    parser.add_argument("--merge_adaptors", action="store_true", help="Merge the adapters and save the model")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the model to the Hugging Face model hub")
    parser.add_argument("--hub_repo_id", type=str, default=f"dpo-ultrafeedback-{DEFAULT_MODEL_ID.replace('/', '-')}",
                        help="The Hugging Face model hub repository ID")
    parser.add_argument("--wandb_project", type=str, default="dpo-ultrafeedback",
                        help="The Weights & Biases project name")
    args = parser.parse_args(input_args)
    return args


# %% [markdown] ################################################################
# ## Setup Hugging Face API key and Weights & Biases API key.
# Can use environment variables, .env file or falls back to interactive user input
def setup_api_keys():
    if os.environ.get('HF_TOKEN', None) is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            if os.environ.get('HF_TOKEN', None) is None:
                raise ImportError
        except ImportError:
            import getpass
            hf_token = getpass.getpass('Enter your Hugging Face API key: ')
            os.environ['HF_TOKEN'] = hf_token

    assert os.environ.get('HF_TOKEN', None) is not None, "Hugging Face API key not set"
    login(token=os.environ['HF_TOKEN'], add_to_git_credential=True)

    if os.environ.get('WANDB_API_KEY', None) is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            if os.environ.get('WANDB_API_KEY', None) is None:
                raise ImportError
        except ImportError:
            from getpass import getpass
            wandb_key = getpass('Enter your Weights & Biases API key: ')
            os.environ['WANDB_API_KEY'] = wandb_key

    assert os.environ.get('WANDB_API_KEY', None) is not None, "Weights & Biases API key not set"


# %% [markdown] ################################################################
# ## Setup dataset
#
# Expected format for the base dataset:
#
# ```json
# {"chosen": "<prompt + good response>", "rejected": "<prompt + worse response>" }
# {"chosen": "<prompt + good response>", "rejected": "<prompt + worse response>" }
# {"chosen": "<prompt + good response>", "rejected": "<prompt + worse response>" }
# ```
#
# Each "chosen" and "rejected" pair is a conversation of 2 or more turns in the standard format: [{"role": role, "content": content}, ...]
# The "chosen" response is the better of the two.
# If more than 2 turns are present, the last must be a response and is the only entry that can be different between "chosen" and "rejected".
#
# The base model used is not instruction tuned, but we want to have we will use the ChatML formatting to impose this structure,
# e.g. `<|im_start|>user\nINSTRUCTION\n<|im_end|>\n<|im_start|>assistant\n...`.

# %% [markdown] ################################################################
# We'll use the UltraFeedback dataset. There was an issue with the original dataset, so we'll use the fixed version that
# cleaned up ~1000 examples that the auto-LLM-eval had incorrectly assessed which response was better.
#
# This dataset was made by Argilla: [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned?row=0)
#
# HF DPOTrainer from TRL expects inputs as a triplet of (prompt, chosen, rejected). We need to pick out just the final response from
# each of the chosen and rejected conversation pairs. We then assign all prior conversation turns to be the `prompt`.
#
# All turns in the prompt and responses are then formated as ChatML and tokenized before being passed to the model.
#
# Take approx 20% of the dataset and split 11000 and 2750 for training and validation respectively.
# %% [markdown] ################################################################

# Have to modify the chat template:
# - to allow for non alternating user/assistant roles since we have to extract the final assistant message from
#   the conversation and apply template
# - to remove the bos_token from each message if there is one before the user prompt/first message since
#   `trl` `DPOTrainer` will add it to the whole conversation, otherwise we'll end up with 2.
#   see `trl/trainer/dpo_trainer.py#L655`
# - to add a space before the assistant message to separate it from the user's end of instruction symbol
#
# NB with mistral, there's only the BOS token at the start of the whole conversation, and it's highly model specific.
# Gemma is the same but LLama has one for each user message.
#
# HF chat template for Mistral has an error where it's missing a space befor the assistant message to
# separate it from the user's end of instruction symbol. This isn't necessary for the final turn since
# it's a user message and the LM's response will correctly tokenize. It's an issue when using chat templates to combine
# multiple turns in strings then tokenize. Without the space it can lead to tokenization combining and confusing the end
# of the instruction symbol (e.g. [/INST] for Mistral) with the start of the assistant's response, thereby worsening performance
# as the instrcution symbol is effectively corrupted or you end up with e.g. 'Hello' rather than '_Hello' as the first word in
# the assistant's response
#
# MAKE SURE to check this for other models
#
# Original for mistral:
#   MISTRAL_ORIG_CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
MISTRAL_FIXED_CHAT_TEMPLATE = "{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

MISTRAL_FINAL_TURN_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only assistant roles are supported when modified template for the final turn!') }}{% endif %}{% endfor %}"

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

# %%
# %%
###############################################################################
# Extract the final assistant response from each conversation

def extract_final_assistant_message(messages):
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    raise ValueError("No assistant message found")

# this version is more general and useful for other tasks
def extract_final_assistant_message_index(messages):
    for i, message in enumerate(reversed(messages)):
        if message["role"] == "assistant":
            return len(messages) - i - 1
    raise ValueError("No assistant message found")

# %%
# Basic tests
def test_extract_final_assistant_message_index():
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you."},
        {"role": "user", "content": "That's good to hear."},
        {"role": "assistant", "content": "Yes, it is."}
    ]

    # print(extract_final_assistant_message_index(messages))
    assert extract_final_assistant_message_index(messages) == 3
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you."},
        {"role": "user", "content": "That's good to hear."},
    ]
    # print(extract_final_assistant_message_index(messages))
    assert extract_final_assistant_message_index(messages) == 1

    try:
        extract_final_assistant_message_index([])
        assert False, "No exception raised"
    except ValueError:
        pass


# %% [markdown] ################################################################
# ## Process the dataset for the required format:
# Map the function to create triplets of (prompt, chosen, rejected) for each
# conversation in the dataset.
###############################################################################

# %%
# System message if there isn't one in the conversation and the model uses them (not all do).
# Assuming here that we're starting from an instruction tuned model already,
# since we're not going to be doing FFT first

# modified from
# https://github.com/huggingface/alignment-handbook/blob/87cc800498b17432cfb7f5acb5e9a79f15c867fc/src/alignment/data.py#L27C1-L38C62
def maybe_insert_system_message(messages, tokenizer, default_system_message=DEFAULT_SYSTEM_MESSAGE):
    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    if messages[0]["role"] == "system":
        if "system" not in chat_template:
            raise ValueError("Model uses system messages, but system message found in conversation")
        return

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template:
        messages.insert(0, {"role": "system", "content": default_system_message})

# %%
def create_triplets(example, tokenizer, final_turn_chat_template=None, default_system_message=DEFAULT_SYSTEM_MESSAGE):

    chosen_messages = example["chosen"]
    rejected_messages = example["rejected"]

    # Functions will raise an error if there isn't the expected structure
    # with at least 2 messages and it can find a final assistant message, so let it raise, don't handle
    # we want to know
    chosen_index = extract_final_assistant_message_index(chosen_messages)
    rejected_index = extract_final_assistant_message_index(rejected_messages)
    # there should only be one conversation turn difference that's an assistant message
    if chosen_messages[:chosen_index] != rejected_messages[:rejected_index]:
        raise ValueError("More than one assistant message is different between chosen and rejected responses. This is not expected."
                         "\nMust be able to extract a single assistant message that is different and keep the rest for the prompt.")

    prompt_messages = chosen_messages[:chosen_index]
    # final different assistant messages
    chosen_messages = [chosen_messages[chosen_index]]
    rejected_messages =[rejected_messages[rejected_index]]

    maybe_insert_system_message(prompt_messages, tokenizer, default_system_message=default_system_message)

    if final_turn_chat_template is None:
        # some templates require alternating user/assistant roles which won't be satisfied here since we're only
        # extracting the final assistant message. We'll use a different template for the final turn.
        final_turn_chat_template = tokenizer.chat_template
    return {
        "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False),
        "chosen": tokenizer.apply_chat_template(chosen_messages, chat_template=final_turn_chat_template, tokenize=False),
        "rejected": tokenizer.apply_chat_template(rejected_messages, chat_template=final_turn_chat_template, tokenize=False),
    }


# %% [markdown] ################################################################
# ## Train the model

def train(args: argparse.Namespace):

    # %% [markdown] ################################################################
    # ## Setup logging
    # %%

    model_id = args.model_id
    dataset_id = args.dataset_id
    run_name = args.run_name
    output_dir = os.path.join(args.output_dir, run_name)

    # check for existance of previous run with the same name so we tell the trainer to resume
    resume_from_checkpoint = output_dir if os.path.exists(output_dir) else None
    print("Model ID:", model_id)
    print("Dataset ID:", dataset_id)
    print("Run name:", run_name)
    print("Output directory:", output_dir)
    print("Resume from checkpoint:", "True" if resume_from_checkpoint else "False")



    if os.environ.get('WANDB_PROJECT', None) is None:
        os.environ['WANDB_PROJECT'] = "dpo-ultrafeedback"

    wandb.init(
        project=os.environ['WANDB_PROJECT'],
        name=run_name,
        group=f"{model_id.replace('/', '-')}",
        resume="allow",
    )


    # %%
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # flash_attention_2 requires padding on the left
    tokenizer.truncation_side = "left" # avoid cutting of the last generation
    tokenizer.chat_template = MISTRAL_FIXED_CHAT_TEMPLATE


    # %% [markdown] ################################################################
    # ## Load, format and save the dataset

    # check if the datasets are already saved on disk, if so load them, if not save them
    if os.path.exists("train_dataset.json") and os.path.exists("test_dataset.json"):
        print("Loading datasets from disk...")
        train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
        eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
    else:
        dataset = load_dataset(dataset_id, split="train").shuffle().select(range(13750))

        # map triplet creation function to our splits of the dataset
        dataset = dataset.map(
            create_triplets,
            remove_columns=dataset.features,
            fn_kwargs={
                "tokenizer": tokenizer,
                "final_turn_chat_template": MISTRAL_FINAL_TURN_CHAT_TEMPLATE,
            },
        )

        # split 11,000 and 2,750 for training and validation assuming 13,750 examples
        dataset = dataset.train_test_split(test_size=int(0.2 * len(dataset)))

        # save to disk as json
        dataset["train"].to_json("train_dataset.json", orient="records")
        dataset["test"].to_json("test_dataset.json", orient="records")
        # this is a bit wasteful, but load from disk to make sure it's the same behaviour
        # as if we had loaded from disk in the first place
        train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
        eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")

    print("\nMapped dataset formatting for training, validation with chat_templating for prompt, chosen, rejected")
    print(train_dataset[0]["prompt"])
    print(train_dataset[0]["chosen"])
    print(train_dataset[0]["rejected"])
    print(f"\n\nTraining dataset: {train_dataset}\nValidation dataset: {eval_dataset}")


    # %% [markdown] ################################################################
    # ## Train the model
    # Use the DPOTrainer from TRL to train the model on the dataset to align to the
    # kind of responses from the chosen examples.
    #
    # DPO typically needs a reference model to keep frozen and compare against the trained model
    # to ensure we're not going too far out of distribution from the original instruction tuned model
    # and the base model it was trained from.
    # This means having 2 models in memory. We can avoid this here because we'll be are not going to be
    # doing a full fine-tune. Instead, we'll use PEFT to train QLoRA adapters.
    #
    # This has the added benefit that the original model is unchanged (until the end if we merge the adapters)
    # so we can avoid having 2 models in memory. the DPOTrainer is written to handle this case iff doing (Q)LoRA
    # training and we can leave the `reference_model` argument as None.
    # %%

    # setup quantisation via bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)


    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )


    # %% [markdown] ################################################################
    # flash_attention_2 requires padding on the left and needs to know a fixed size length for the input.
    # The Alignment Handbook truncates samples from the left, meaning that the final portion of chosen and
    # rejected responses are kept, i.e. it will remove some of the prompt rather than the responses, which is preferrable.
    # Code below instead filters samples longer than a threshold max percentile calculated over the dataset.
    # Comment this section back in to recalculate and filter the dataset. Or simply use constants that may result in
    # the trunctation from the left.
    # %%

    #### COMMENT IN TO RECALCULATE MAX LENGTHS ####
    # from numpy import percentile

    # target_percentile = 97
    # # lets find the p95 length of the prompt
    # prompt_length = int(percentile([len(tokenizer(x)["input_ids"]) for x in train_dataset["prompt"]], target_percentile))
    # max_seq_length_chosen = int(percentile([len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) for x in train_dataset], target_percentile))
    # max_seq_length_rejected = int(percentile([len(tokenizer(x["prompt"] + x["rejected"])["input_ids"]) for x in train_dataset], target_percentile))
    # max_seq_length = max(max_seq_length_chosen, max_seq_length_rejected)

    # filter datasets to remove samples that are too long
    # train_dataset = train_dataset.filter(lambda x: len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= max_seq_length)
    # eval_dataset = eval_dataset.filter(lambda x: len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= max_seq_length)
    # print(f"len(train_dataset): {len(train_dataset)}")
    # print(f"len(eval_dataset): {len(eval_dataset)}")

    # # Up the lengths to next multiple of 2, why 2? Don't know
    # prompt_length = ((prompt_length + 1) // 2) * 2
    # max_seq_length = ((max_seq_length + 1) // 2) * 2
    # print(f"{target_percentile} prompt length: {prompt_length}")
    # print(f"{target_percentile} prompt + chosen length: {max_seq_length}")

    #! experiment with changing this for the values calculated above
    prompt_length = 1024
    max_seq_length = 1512


    # %% [markdown] ################################################################
    # ## Training hypers
    # DPO requires 10-100x lower LR than standard fine-tuning
    # Here we would take the FFT LR of 2e-4 and might expect 2e-6 to 2e-5 for DPO based on QLoRA paper
    # But, bump it up to 5e-5 (just 4x smaller) to start with and see how it manages
    # If batch size is a problem or we want to increase it, we could reduce the max_seq_length...keep as is for now.
    # %%
    # %% [markdown] ################################################################
    # DPO beta factor as an important hyperparameter, controls divergence from the reference model
    # higher is less divergence and it's typically 0.1 - 0.5. Don't want to stray too far from the well trained base model
    # %%

    # using config based on QLoRA paper, Sebastian Raschka's blog and Phillip Schmidt's blog
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


    trainer_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        dataloader_num_workers=4,
        per_device_train_batch_size=8,  # lower to 1 to fit on 16 or even 24GB GPU if necessary
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,  # increase if per_device_train_batch_size is lowered
        gradient_checkpointing=True,  # save memory
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=700,
        bf16=True,
        tf32=True,  # mixed precision, needed for accumulation
        push_to_hub=False,  # we'll do this manually
        report_to="wandb",
        run_name=run_name,
    )

    dpo_args = {
        "beta": args.dpo_beta,  # DPO beta factor for loss, controls divergence from the reference model, higher is less divergence
        "loss_type": "sigmoid"
    }

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Using (Q)LoRA we don't need additional copy for the reference
        peft_config=peft_config,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_prompt_length=prompt_length,
        beta=dpo_args["beta"],
        loss_type=dpo_args["loss_type"],
    )


    # %%
    # ## Train!
    # Key metric on the logs is the `loss` of course
    # but also the DPO metric of the margin between chosen and rejected, that we want to see increase.
    # Key knobs to tweak are the learning rate and beta factor
    # Likely optimisations for speed via `per_device_[train/eval]_batch_size` and `gradient_accumulation_steps`

    print("\n\nTraining...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()

    print("\n\nTraining complete\n")


    # %%
    # Merge the adapters and save the model. Alternatively use the script in the repo:
    #   `lora_merge_push.py` or `lora_push_to_hub.py` to do this
    if args.merge_adaptors:

        print("Merging LoRA and base model and saving...")
        # try to save some VRAM as this can require 2x if all on GPU...not thouroughly tested to see if it's
        # completely necessary in latest transformers, but this works either way
        # Merge LoRA and base model and save
        del model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        # Load PEFT model on CPU
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        # Merge LoRA and base model and save
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
        model = merged_model

    if args.push_to_hub:
        print("Pushing to Hugging Face model hub...")
        # push to hub
        model.push_to_hub(args.hub_repo_id, use_temp_dir=True)
        tokenizer.push_to_hub(args.hub_repo_id, use_temp_dir=True)


# %%

if __name__ == "__main__":
    args = parse_arguments()
    setup_api_keys()

    train(args)

    print("DPO training and model saving complete")
    exit(0)


# %% [markdown] ################################################################
# Alternative to running via the CLI.
# If in an interactive jupyter session, run this cell instead, having run all the above
# %%
# uncomment to run
# Defaults to sensible defaults, same as the CLI but change as needed
# args = parse_arguments(["--merge_adaptors", "--push_to_hub"])
# setup_keys()
# train(args)
