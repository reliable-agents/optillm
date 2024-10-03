import asyncio
import json
import random
from typing import Any

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import HfArgumentParser

from optillm.server_context import ProxyServer, ServerContext, VLLMServer

from dataclasses import dataclass

@dataclass
class ScriptArguments:
    approach: str
    model: str
    dataset_name: str
    dataset_split: str = "train"
    dataset_column: str = "prompt"
    num_samples: int = 5
    output_file: str = "completions.jsonl"
    hub_dataset_id: str = None
    push_to_hub: bool = False

@dataclass
class SamplingArguments:
    n: int = 1
    temperature: float = 0.7
    max_tokens: int = 2048

# OptILM approaches
APPROACHES = [
    "none",
    "mcts",
    "bon",
    "moa",
    "rto",
    "z3",
    "self_consistency",
    "pvg",
    "rstar",
    "cot_reflection",
    "plansearch",
    "leap",
    "re2",
]


async def generate_response(prompt: str, args: ScriptArguments, sampling_args: SamplingArguments) -> dict[str, Any]:
    """Generate a response using the specified approach."""
    if args.approach == "none":
        # Use the base model without any optimization technique
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=sampling_args.temperature,
            max_tokens=sampling_args.max_tokens,
        )
        return {
            "content": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens,
        }
    else:
        # Use OptILM with the specified approach
        client = AsyncOpenAI(api_key="none", base_url="http://localhost:8080/v1")
        response = await client.chat.completions.create(
            model=f"{args.approach}-{args.model}",  # Assuming OptILM uses this naming convention
            messages=[{"role": "user", "content": prompt}],
            temperature=sampling_args.temperature,
            max_tokens=sampling_args.max_tokens,
        )
        return {
            "content": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens,
        }


async def rank_responses(prompt: str, responses: list[dict[str, Any]]) -> list[int]:
    """Rank the responses using the LLM."""
    ranking_prompt = f"Given the following prompt:\n\n{prompt}\n\nRank the following {len(responses)} responses from best to worst, considering accuracy, completeness, and relevance. Provide the ranking as a comma-separated list of indices (0-indexed). Do not add any explanations or any other text other than the comma-separated list.\n\n"
    for i, response in enumerate(responses):
        ranking_prompt += f"Response {i}:\n{response['content']}\n\n"
    client = AsyncOpenAI()
    ranking_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ranking_prompt}],
    )

    ranking_str = ranking_response.choices[0].message.content.strip()
    return [int(idx) for idx in ranking_str.split(",")]


async def process_sample(sample: dict[str, Any], args: ScriptArguments, sampling_args: SamplingArguments) -> dict[str, Any]:
    """Process a single sample from the dataset."""
    prompt = sample[args.dataset_column]
    results = []

    for _ in range(sampling_args.n):
        response = await generate_response(
            prompt, args, sampling_args
        )
        results.append({"approach": args.approach, **response})

    random.shuffle(results)
    # Rank the responses
    rankings = await rank_responses(prompt, results)

    # Add rankings to results
    if len(rankings) != len(results):
        raise ValueError(f"Number of rankings does not match number of results. Got {len(rankings)} rankings and {len(results)} results.")
    for rank, idx in enumerate(rankings):
        results[idx]["rank"] = rank

    return {
        "prompt": prompt,
        "results": results,
    }


async def generate_dataset(args: ScriptArguments, sampling_args: SamplingArguments):
    """Generate the dataset and save it to a JSONL file."""
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    # List to store the coroutine for each sample
    tasks = [process_sample(sample, args, sampling_args) for sample in dataset]

    with open(f"data/{args.output_file}", "w") as f:
        # Use asyncio.gather to process all samples concurrently
        for result in tqdm(await asyncio.gather(*tasks)):
            f.write(json.dumps(result) + "\n")

    # Push to hub
    if args.push_to_hub:
        if args.hub_dataset_id is None:
            # Set default based on model name
            args.hub_dataset_id = f"{args.model.split('/')[-1]}-completions"
        revision = f"{args.approach}"
        results_ds = load_dataset("json", data_files=f"data/{args.output_file}", split="train")
        dataset = dataset.add_column("optillm_completions", results_ds["results"])
        dataset.push_to_hub(args.hub_dataset_id, revision=revision, private=True)


def main():
    parser = HfArgumentParser((ScriptArguments, SamplingArguments))
    args, sampling_args = parser.parse_args_into_dataclasses()

    with ServerContext(VLLMServer, dict(model_path=args.model)) as vllm_server:
        vllm_server.wait()
        with ServerContext(ProxyServer, dict(model_path=args.model, approach=args.approach)) as proxy_server:
            proxy_server.wait()
            asyncio.run(generate_dataset(args, sampling_args))


if __name__ == "__main__":
    main()
