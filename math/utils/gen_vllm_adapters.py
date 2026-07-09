import argparse
import torch
import sys
import os
import json
from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="Base model identifier or path")
parser.add_argument('--adapter', type=str, help="PEFT adapter model identifier or path")
parser.add_argument("--data_path", type=str, default="fxmeng/pissa-dataset")
parser.add_argument('--sub_task', nargs='+', help='Sub-task names (if any)')
parser.add_argument('--dataset_split', type=str, default="test", help='Dataset split to use')
parser.add_argument('--output_file', type=str, default="model_response.jsonl", help="Output file for responses")
parser.add_argument("--batch_size", type=int, default=50, help="Batch size for inference")
parser.add_argument('--temperature', type=float, default=0.0, help="Sampling temperature")
parser.add_argument('--top_p', type=float, default=1, help="Top-p sampling")
parser.add_argument('--max_tokens', type=int, default=1024, help="Maximum tokens to generate")
args = parser.parse_args()

# Define sampling parameters
stop_tokens = []
sampling_params = SamplingParams(
    temperature=args.temperature, 
    top_p=args.top_p, 
    max_tokens=args.max_tokens, 
    stop=stop_tokens
)

# ----------------------------
# Load the base model and adapter
# ----------------------------
merged_dir = args.adapter + "/merged"
# merge the adapter if the merged model doesn't exist
if not os.path.exists(merged_dir):
    # Load the base model using Transformers
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # Integrate the adapter with PEFT
    print("loading model with adapter")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    print("merging adapter")
    merged_model = peft_model.merge_and_unload()
    print("saving merged model")
    merged_model.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, max_length=512)
    print("saving tokenizer")
    tokenizer.save_pretrained(merged_dir)

    # Clear memory removing unused models
    del base_model, peft_model, merged_model
    torch.cuda.empty_cache()

# Instantiate the LLM object from vllm using the model with adapter

llm = LLM(model=merged_dir, tensor_parallel_size=torch.cuda.device_count())


# ----------------------------
# Data loading and batching
# ----------------------------
def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data_list = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_data_list.append(data_list[start:end])
    # Add the remaining data as the last batch
    last_start = (n-1) * batch_size
    batch_data_list.append(data_list[last_start:])
    return batch_data_list

if args.sub_task is None:
    dataset = load_dataset(args.data_path, split=args.dataset_split)
else:
    all_test_dataset = []
    for task in args.sub_task:
        ds = load_dataset(args.data_path, data_dir=task, split=args.dataset_split)
        print(f"{args.data_path}/{task}/{args.dataset_split}")
        for k, v in ds[0].items():
            print("-"*100)
            print(f"{k}:\t{v}")
        print("+"*100)
        all_test_dataset.append(ds)
    dataset = concatenate_datasets(all_test_dataset)

batch_dataset_query = batch_data(dataset["instruction"], batch_size=args.batch_size)
batch_dataset_answer = batch_data(dataset["output"], batch_size=args.batch_size)
batch_dataset_task = batch_data(dataset["type"], batch_size=args.batch_size)

# ----------------------------
# Generation loop
# ----------------------------
for idx, (batch_query, batch_answer, batch_task) in enumerate(zip(batch_dataset_query, batch_dataset_answer, batch_dataset_task)):
    with torch.no_grad():
        completions = llm.generate(batch_query, sampling_params)
    for query, completion, answer, task in zip(batch_query, completions, batch_answer, batch_task):
        with open(args.output_file, 'a') as f:
            json.dump({
                'type': task, 
                'query': query, 
                'output': completion.outputs[0].text, 
                'answer': answer
            }, f)
            f.write('\n')
