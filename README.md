# DPO Alignment Mistral-7b

This repository contains the code for aligning Language Learning Models (LLMs) with the DPO (Differential Preference Optimization) method using the TRL (Transformers Reinforcement Learning) library. The base model used for this project is [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) but can work with most HF models.

The training for DPO is written in a single script that can be run as-is, or interactively as an interactive jupyter session in vscode (using `# %%` cell syntax). As such, rather than using `argparse` or similar, environment variables can be used for configuration, or interactively edit options.


## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/teome/dpo-lora-alignment.git
```

2. Install dependencies

```bash
pip install -r requirements-dev.txt
```

or use poetry if you prefer

Flash attention is used to speed up training and inference. It can be problematic to install. If it fails during the install from requirements-dev.txt, just remove it, rerun the install then separately run:

```bash
pip install flash-attn --no-build-isolation
```

See [flash attention docs](https://github.com/Dao-AILab/flash-attention) for more info.

## Training

The training script, `dpo.py` is written to be runnable either from the command line with arguments, or as a jupyter notebook via vscode interacive jupyter session (using `# %%` cell syntax). It can also be exported to a standard jupyter notebook.

### CLI usage

```
DPO Alignment Script

options:
  -h, --help            show this help message and exit
  --model_id MODEL_ID   The model ID for the tokenizer and model (default: mistralai/Mistral-7B-Instruct-v0.2)
  --dataset_id DATASET_ID
                        The dataset ID for the training data (must comply with DPO expected format, see code and comments if changing) (default: argilla/ultrafeedback-binarized-preferences-cleaned)
  --num_train_epochs NUM_TRAIN_EPOCHS
                        The number of training epochs (default: 1)
  --learning_rate LEARNING_RATE
                        The learning rate for training (default: 5e-05)
  --dpo_beta DPO_BETA   The DPO beta factor for loss, controls divergence from the reference model, higher is less divergence (default: 0.1)
  --output_dir OUTPUT_DIR
                        The output directory for saving the trained model (default: ./outputs)
  --run_name RUN_NAME   The name of the training run (default: dpo_mistralai-Mistral-7B-Instruct-v0.2_<autogen_current_time_stamp>)
  --merge_adaptors      Merge the adapters and save the model (default: False)
  --push_to_hub         Push the model to the Hugging Face model hub (default: False)
  --hub_repo_id HUB_REPO_ID
                        The Hugging Face model hub repository ID (default: dpo-ultrafeedback-mistralai-Mistral-7B-Instruct-v0.2)
  --wandb_project WANDB_PROJECT
                        The Weights & Biases project name (default: dpo-ultrafeedback)
```

The DPO training continues from the instruction-tuned `Mistral-7B-Instruct-v0.2`. To get the best results, it's generally advised to first instruction fine-tune on the same dataset you'll use for DPO. This isn't included here to keep it DPO focussed.

The dataset used is the fixed and cleaned UltraFeedback dataset [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned). This dataset was built using LLM-as-judge style labelling of response pairs to create each dataset entry consisting of (prompt, chosen-response, rejected-response). The original dataset contained errors in labelling, so here we use the version released by Argilla. See their [dataset page](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) for more info


Set the following environment variables to enable `wandb` logging and push to `Hugging Face Hub`. You can either put these in a `.env` file or `export ...` env variables in the shell.

```
HF_TOKEN=...
WANDB_API_KEY=...
```

If these can't be found, the script falls back to interactive prompt the user to enter them, so make sure they exist if you're running without access to a prompt or it will just hang.

The model checkpoints are saved in `<output_dir>/<run_name>` as set from the CLI or interactively. By default, the `<run_name>` is generated automatically from the model name and a time-suffix if but can be set manually.

The resulting model (and it's tokenizer) are optionally pushed to Hugging Face Hub.
It's also optional to first merge the LoRA adaptors into the base model first. This is recommended if you plan to use the model for inference in anything other than HF, or even with HF it's just easier to deal with a single entity and not worry about adaptors. The downside of course is that you'll have to deal with a much larger model rather than just the smaller adaptors for saving locally or to the hub.


## Evaluation

Evaluation is achieved by

1. Vibes: just run the `evaluate.py` script that has command line arguments to take prompts and sampling parameters to get a feel for the performance.
2. [MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md)

[MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) from LMSYS can be used to evaluate instruction and multi-turn chats. It does this by using the LLM-as-judge method. In particular, using GPT-4-Turbo (or other OpenAI compatible endpoint) to either directly rate a response, or provide a comparison between responses to decide which is preferred on the bases if conciceness, cohenrence, information etc.

There are lots of issues with using a particular model to judge other models' responses but it's surprisingly consistent with human judgement and a useful alternative to more specialised and task specific benchmarks that suffer from model overtraining and distance from real-world usage. It's far from perfect, but it's useful.

Since we're using GPT-4 (`gpt-4-turbo-preview`) we'll need to setup the API key

```bash
export OPENAI_API_KEY=sk-....
```

First clone the repo, install and change into the `llm_judge` directory

```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"
cd fastchat/llm_judge
```

### Using FastChat Hugging Face models directly

_Copied from the FastChat docs_:

Step 1. Generate model answers to MT-bench questions

```bash
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]
```
Arguments:

[MODEL-PATH] is the path to the weights, which can be a local folder or a Hugging Face repo ID.
[MODEL-ID] is a name you give to the model.
e.g.,

```bash
python gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5
```

The answers will be saved to data/mt_bench/model_answer/[MODEL-ID].jsonl.

To make sure FastChat loads the correct prompt template, see the supported models and how to add a new model here.

You can also specify --num-gpus-per-model for model parallelism (needed for large 65B models) and --num-gpus-total to parallelize answer generation with multiple GPUs.


Judgement can be individual scoring for the models' responses, as comparisons with a baseline model, or as a set of pairwise comparisons between all models provided or more. We'll go with pairwise comparison between our starting and DPO tuned model.

Step 2, Option 3: Run GPT-4 judge with all pair comparisons
Another option is to run pairwise comparisons on all possible pairs. This could be more expensive when #models increases, but it gives you a more comprehensive information.

```bash
python gen_judgment.py --mode pairwise-all --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
python show_result.py --mode pairwise-all
```

### Using the vLLM backend

To speed up generation of the outputs from our models, it's also possible to use vLLM as the inference backend, rather than standard Hugging Face transformers inference. See the [docs for FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md)

```bash
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.vllm_worker --model-path [MODEL-PATH]
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
python gen_api_answer.py --model [MODEL-NAME] --openai-api-base http://localhost:8000/v1 --parallel 50
```

Where `--model-path` can be local or a HF repo and `--parallel` is the number of concurrent API calls to the backend, which will attempt to batch appropriately.


## Results for pairwise comparison using GPT-4

| model                                   | win | loss | tie | win_rate | loss_rate | win_rate_adjusted |
|-----------------------------------------|-----|------|-----|----------|-----------|-------------------|
| mistral-7b-instruct-v0.2-ultrafeedback | 61  | 20   | 79  | 0.38125  | 0.12500   | 0.628125          |
| mistral-7b-instruct-v0.2               | 20  | 61   | 79  | 0.12500  | 0.38125   | 0.371875          |
