# =================================
# # Inference with Activations
"""
This script contains the code to run a full inference on language models (both small and large) while extracting the activations from the
middle-last layers of the last prompt token. The rationale behind choosing this is that usually the reasoning of the model is usually more
or less complete at the last token, and the middle-last layers are usually the ones that contain more information as it is just before the
response generation at this stage.

USAGE:

"""
# =================================

# 1. Imports
#-------------
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_torch_dtype(dtype_name: str):
    """
    What it does: This function is used to convert a string to the actual pytorch data type.
    Rationale: When you load 7B/8B models, using full float32 can be expensive. Therefore, we would want to revert to bfloat16 or float16.
    This allows us to change the precision from the command line itself.
    """
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"ERROR: Unsupported dtype: {dtype_name}")

def format_chat_prompt(tokenizer, user_prompt: str):
    """
    What it does: Takes a raw prompt from the dataset and formats it into the model's expected chat format.
    Rationale: Different models may have different ways of handling chat prompts, so we need to ensure the prompt is formatted correctly for each model.
    """
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
        )
    return f"User: {user_prompt}\nAssistant:"

@torch.no_grad()
def get_prompt_activations(model, input_ids, attention_mask):
    """
    what it does: This is the core activation extraction function.
    Rationale: to extract activaions 
    """
    # running the model on the prompt (ie: Inference code)
    outputs = model( # model - is the loaded LLM
        input_ids = input_ids, # the tokenized prompt
        attention_mask = attention_mask, # tells us which tokens are real and which are not (eg: padding etc) [1,1,0,0,1]. 1- real token; 0 - padding
        output_hidden_states = True, #asking the model to return the hidden stae represntations as well, as it usually only returns the logits
        use_cache = False,
    )

    """
    NOTE: each hidden state is of the dimensions: [batch_size, sequence_length, hidden_dim], while logis are of the dimensions [batch_size, sequence_lenth, vocab_size]
    Therefore, hidden_states are the actual internal respresentations of the model (ie: what the model understands) while the logits are
    the model's predictions of the next token based on its understanding (ie: hidden states).

   
    Emebdding layer = hidden_state[0] ([1, 78, 4096])
    layer1 = hidden_state[1] ([1, 78, 4096])
    ...
    layer32 = hidden_state[32] ([1, 78, 4096])

    Therefore, if a model has 32 transformer layers then it has length = 33 (0-32).
    """
    hidden_states = outputs.hidden_states

    # finding the index of the last token, since it is zero indexed we subtract 1
    final_token_idx = int(attention_mask[0].sum().item()) - 1 # .item() converts a single-value pytorch tensor to normal python number

    layer_vectors = []
    for layer_hidden in hidden_states:
        vec = layer_hidden[0, final_token_idx, :].detach().cpu() # take the first batch item, for the last token index, take all the hidden dimensions
        layer_vectors.append(vec)

        """
        Example: 
        prompt_length = 12 tokens (seq length)
        hidden_dim = 4096
        num_transformer_layers = 28

        therefore:
        # of hdden states = 29 (including the embedding layer)
        dimension of each hidden state = [1, 12, 4096]
        final_token_idx = 11
        """

    activations = torch.stack(layer_vectors, dim=0) # turns into a list of vectors into a tensor

    return activations

def generate_model_response(model, tokenizer, input_ids, attention_mask, max_new_tokens: int):

    generated_ids = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask, 
        max_new_tokens = max_new_tokens,
        do_sample = False, # means greedy decoding is used - chooses the highest probability amongst the list of possible next tokens 
        pad_token_id = tokenizer.eos_token_id,
    )

    prompt_len = input_ids.shape[1] # finds out the length of the prompt so that it can be removed from the response
    response_ids = generated_ids[0, prompt_len:] # extracts only the model's response and not the prompt.

    # converts the response tokens into text
    response_text = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    ).strip()

    return response_text, generated_ids.detach().cpu()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", 
        type=str,
        required= True,
        help="Local Path to the directory where the model is stored"
    )
    parser.add_argument(
        "--model_tag", 
        type=str,
        required= True,
        help="Short name used for output folder"
    )
    parser.add_argument(
        "--input_csv_path", 
        type=str,
        default= "data/xstest/XSTest.csv",
        help="Local Filepath to the dataset CSV file."
    )
    parser.add_argument(
        "--prompt_col", 
        type=str,
        default="prompt",
        help="Name of the column that contains the prompt"
    )
    parser.add_argument(
        "--label_col", 
        type=str,
        default=None,
        help="Name of the column that contains the label (whether safe or unsafe)"
    )
    parser.add_argument(
        "--category_col", 
        type=str,
        default=None,
        help="Optional category/column name"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="outputs/xstest",
        help="Local Path to the directory where the model is stored"
    )
    parser.add_argument(
        "--max_examples", 
        type=int,
        default=None,
        help="Limit the number of examples to inder for debugging purposes"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int,
        default=256
    )
    parser.add_argument(
        "--max_prompt_tokens", 
        type=int,
        default=2048
    )
    parser.add_argument(
        "--dtype", 
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--dataset", 
        type=str,
        default="mixed",
        help="Dataset name to be added into id column for activation."
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_csv_path = Path(args.input_csv_path)

    # creates the activations and output directories
    output_dir = Path(args.output_dir) / args.model_tag  # / joins the path of output_dir with the a clean model name (tag)
    activations_dir = output_dir / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_dir.mkdir(parents=True, exist_ok=True)

    responses_path = output_dir / "responses.jsonl"
    metadata_path = output_dir / "run_metadata.json"

    # id header
    dataset = args.dataset


    print("="*80)
    print(f"Extracting Activations during Inference")
    print("-"*80)
    print(f"INFO: Loading Dataset...")
    print(f"INFO: Input CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    print(f"INFO: Dataset shape: {df.shape}")
    print(f"INFO: COlumns: {df.columns.tolist()}")

    if args.max_examples is not None:
        df = df.head(args.max_examples)
        print(f"INFO: Using the first {len(df)} examples for debug run.")

    if "id" not in df.columns:
        print(f"WARNING: No 'id' column found. Creating one now. ")
        df.insert(0, "id", [f"{dataset}_{i:05d}" for i in range(len(df))])
        
    if args.prompt_col not in df.columns:
        raise ValueError(
            f"Prompt column '{args.prompt_col}' not found."
            f"Available columns: {df.columns.tolist()}"
        )
    print("="*80)

    print(f"INFO: Loading tokenizer and model")
    print(f"INFO: Model path: {model_path}")

    torch_dtype = get_torch_dtype(args.dtype)
    
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=True,
    )

    print(f"INFO: Successfully Loaded Tokenizer!")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # loading the model
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code="True"
    )

    model.eval() # cuz we are not training the model

    print(f"INFO: Successfully Loaded the Model!")
    print(f"INFO: dtype: {args.dtype}")
    print(f"INFO: Output dir: {output_dir}")

    run_metadata = {
        "model_path": str(model_path),
        "model_tag": args.model_tag,
        "input_csv_path": str(input_csv_path),
        "num_examples": int(len(df)),
        "prompt_col": args.prompt_col,
        "label_col": args.label_col,
        "category_col": args.category_col,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "dtype": args.dtype,
        "activation_type": "final_prompt_token_all_layers",
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    print("=" * 80)
    print("INFO: Starting Inference and Activation Extraction...")

    with responses_path.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            example_id = str(row["id"])
            raw_prompt = str(row[args.prompt_col])

            label = None
            if args.label_col is not None and args.label_col in df.columns:
                label = row[args.label_col]

            category = None
            if args.category_col is not None and args.category_col in df.columns:
                category = row[args.category_col]

            formatted_prompt = format_chat_prompt(tokenizer, raw_prompt)

            encoded = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_tokens,
                padding=False,
            )

            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            # 1. Extract prompt activations
            prompt_activations = get_prompt_activations(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # 2. Generate final model response
            response_text, generated_ids = generate_model_response(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
            )

            # 3. Save activations
            activation_path = activations_dir / f"{example_id}.pt"

            torch.save(
                {
                    "id": example_id,
                    "model_tag": args.model_tag,
                    "model_path": str(model_path),
                    "activation_type": "final_prompt_token_all_layers",
                    "activations": prompt_activations,
                    "shape": list(prompt_activations.shape),
                    "prompt_num_tokens": int(input_ids.shape[1]),
                },
                activation_path,
            )

            # 4. Save response record
            output_record = {
                "id": example_id,
                "dataset": "xstest",
                "model_tag": args.model_tag,
                "prompt": raw_prompt,
                "formatted_prompt": formatted_prompt,
                "label": None if pd.isna(label) else label,
                "category": None if pd.isna(category) else category,
                "response": response_text,
                "activation_path": str(activation_path),
                "prompt_num_tokens": int(input_ids.shape[1]),
                "generated_num_tokens": int(generated_ids.shape[1] - input_ids.shape[1]),
            }

            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("INFO: Done")
    print(f"INFO: Responses saved to: {responses_path}")
    print(f"INFO: Activations saved to: {activations_dir}")
    print(f"INFO: Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
       






