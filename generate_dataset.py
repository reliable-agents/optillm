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

RANKING_MODEL = "gpt-4o-mini"
RANKING_PROMPT = "Given the following prompt:\n\n{prompt}\n\nRank the following {num_responses} responses from best to worst, considering accuracy, completeness, and relevance. Provide the ranking as a comma-separated list of indices (0-indexed). Do not add any explanations or any other text other than the comma-separated list.\n\n"

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
@dataclass
class ScriptArguments:
    """
    Script configuration.

    Attributes:
    ----------
    approach : str
        The approach to be used.
    model : str
        The model to be used.
    dataset_name : str
        The name of the dataset.
    dataset_split : str, optional
        The split of the dataset (default is "train").
    dataset_column : str, optional
        The column of the dataset with the prompts (default is "prompt").
    prompt_suffix : str, optional
        The suffix to add to the prompt for e.g. CoT and MATH (default is "").
    num_samples : int, optional
        The number of samples to generate (default is 5).
    output_filename : str, optional
        The name of the output file (default is None).
    hub_dataset_id : str, optional
        The ID of the dataset on the hub (default is None).
    push_to_hub : bool, optional
        Whether to push the results to the hub (default is False).
    debug : bool, optional
        Whether to run in debug mode (default is False).
    """
    approach: str
    model: str
    dataset_name: str
    dataset_split: str = "train"
    dataset_column: str = "prompt"
    prompt_suffix: str = ""
    num_samples: int = 5
    output_filename: str = None
    hub_dataset_id: str = None
    push_to_hub: bool = False
    debug: bool = False

@dataclass
class SamplingArguments:
    """
    Sampling configuration.

    Attributes:
    ----------
    n : int, optional
        The number of responses to generate for each prompt (default is 1).
    temperature : float, optional
        The sampling temperature to use (default is 0.7).
    max_tokens : int, optional
        The maximum number of tokens to generate (default is 2048).
    """
    n: int = 1
    temperature: float = 0.7
    max_tokens: int = 2048


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
            messages=[{"role": "user", "content": prompt + args.prompt_suffix}],
            temperature=sampling_args.temperature,
            max_tokens=sampling_args.max_tokens,
        )
        return {
            "content": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens,
        }


async def rank_responses(prompt: str, responses: list[dict[str, Any]], ranking_model: str = None, ranking_prompt: str = None,) -> list[int]:
    """Rank the responses using the LLM."""
    if ranking_model is None:
        ranking_model = RANKING_MODEL
    if ranking_prompt is None:
        ranking_prompt = RANKING_PROMPT.format(prompt=prompt, num_responses=len(responses))
    for i, response in enumerate(responses):
        ranking_prompt += f"Response {i}:\n{response['content']}\n\n"

    client = AsyncOpenAI()
    ranking_response = await client.chat.completions.create(
        model=RANKING_MODEL,
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

    if len(results) > 1:
        rankings = await rank_responses(prompt, results)
    
        for rank, idx in enumerate(rankings):
            results[idx]["rank"] = rank
    else:
        results[0]["rank"] = 0

    return {
        "prompt": prompt,
        "results": results,
    }


async def generate_dataset(args: ScriptArguments, sampling_args: SamplingArguments):
    """Generate the dataset and save it to a JSONL file."""
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    # List to store the coroutine for each sample
    tasks = [process_sample(sample, args, sampling_args) for sample in dataset]

    model_name = args.model.split("/")[-1]
    config_name = f"{args.dataset_name.replace('/', '_')}--{args.approach}--T{sampling_args.temperature}--N{sampling_args.n}"
    if args.output_filename is None:
        args.output_filename = f"{config_name}-completions.jsonl"

    with open(f"data/{args.output_filename}", "w") as f:
        # Use asyncio.gather to process all samples concurrently
        for result in tqdm(await asyncio.gather(*tasks)):
            f.write(json.dumps(result) + "\n")

    # Push to hub
    if args.push_to_hub:
        if args.hub_dataset_id is None:
            # Set default based on model name
            args.hub_dataset_id = f"{model_name}-optillm-completions"
        results_ds = load_dataset("json", data_files=f"data/{args.output_filename}", split="train")
        dataset = dataset.add_column("optillm_completions", results_ds["results"])
        url = dataset.push_to_hub(args.hub_dataset_id, config_name=config_name, private=True)
        print(f"Pushed dataset to: {url}")


def main():
    parser = HfArgumentParser((ScriptArguments, SamplingArguments))
    args, sampling_args = parser.parse_args_into_dataclasses()

    if args.debug:
        asyncio.run(generate_dataset(args, sampling_args))
    else:
        with ServerContext(VLLMServer, dict(model_path=args.model)) as vllm_server:
            vllm_server.wait()
            with ServerContext(ProxyServer, dict(model_path=args.model, approach=args.approach)) as proxy_server:
                proxy_server.wait()
                asyncio.run(generate_dataset(args, sampling_args))


if __name__ == "__main__":
    main()
