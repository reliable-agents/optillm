# optillm

> [!NOTE]
> This fork includes tighter integration with vLLM and better support for generating synthetic preferences from models on the Hugging Face Hub.

optillm is an OpenAI API compatible optimizing inference proxy which implements several state-of-the-art techniques that can improve the accuracy and performance of LLMs. The current focus is on implementing techniques that improve reasoning over coding, logical and mathematical queries. It is possible to beat the frontier models using these techniques across diverse tasks by doing additional compute at inference time.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/codelion/optillm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SpuUb8d9xAoTh32M-9wJsB50AOH54EaH?usp=sharing)

## Installation

Just clone the repository with `git` and use `pip install` to setup the dependencies.

```bash
git clone https://github.com/codelion/optillm.git
cd optillm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set up the `OPENAI_API_KEY` environment variable (for OpenAI) 
or the `AZURE_OPENAI_API_KEY`, `AZURE_API_VERSION` and `AZURE_API_BASE` environment variables (for Azure OpenAI)
or the `AZURE_API_VERSION` and `AZURE_API_BASE` environment variables and login using `az login` for Azure OpenAI with managed identity (see [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity)).

You can then run the optillm proxy as follows.

```bash
python optillm.py
2024-09-06 07:57:14,191 - INFO - Starting server with approach: auto
2024-09-06 07:57:14,191 - INFO - Server configuration: {'approach': 'auto', 'mcts_simulations': 2, 'mcts_exploration': 0.2, 'mcts_depth': 1, 'best_of_n': 3, 'model': 'gpt-4o-mini', 'rstar_max_depth': 3, 'rstar_num_rollouts': 5, 'rstar_c': 1.4, 'base_url': ''}
 * Serving Flask app 'optillm'
 * Debug mode: off
2024-09-06 07:57:14,212 - INFO - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://192.168.10.48:8000
2024-09-06 07:57:14,212 - INFO - Press CTRL+C to quit
```

Note that the server defaults can be overriden when the proxy is spun up as follows:

```shell
python optillm.py --approach mcts --return-full-response true
```

### Starting the optillm proxy for a local server (e.g. llama.cpp, vLLM, etc)

- Set the `OPENAI_API_KEY` env variable to a placeholder value
  - e.g. `export OPENAI_API_KEY="no_key"`
- Run `./llama-server -c 4096 -m path_to_model` to start the server with the specified model and a context length of 4096 tokens
- Run `python3 optillm.py --base_url base_url` to start the proxy
  - e.g. for llama.cpp, run `python3 optillm.py --base_url http://localhost:8080/v1`

> [!WARNING]
> Note that llama-server currently does not support sampling multiple responses from a model, which limits the available approaches to the following:
> `cot_reflection`, `leap`, `plansearch`, `rstar`, `rto`, `self_consistency`, `re2`, and `z3`.
> In order to use other approaches, consider using an alternative compatible server such as [ollama](https://github.com/ollama/ollama).

> [!NOTE]
> You'll later need to specify a model name in the OpenAI client configuration. Since llama-server was started with a single model, you can choose any name you want.

To use the proxy with vLLM, first spin up the OpenAI-compatible server with:

```shell
vllm serve {MODEL_NAME}
```

Then run:

```shell
python optillm.py --base_url http://localhost:8000/v1
```

We also provide context managers for the vLLM server and optillm proxy, so you can run them directly in Python as follows:

```python
with ServerContext(VLLMServer, dict(model_path=args.model)) as vllm_server:
    vllm_server.wait()
    with ServerContext(ProxyServer, dict(model_path=args.model, approach=args.approach)) as proxy_server:
        proxy_server.wait()
        run_task(...) # Placeholder for real task
```

## Usage

Once the proxy is running, you can use it as a drop in replacement for an OpenAI client by setting the `base_url` as `http://localhost:8000/v1`.

```python
import os
from openai import OpenAI

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = "http://localhost:8000/v1"
client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": "Write a Python program to build an RL model to recite text from any position that the user provides, using only numpy."
    }
  ],
  temperature=0.2,
)

print(response)
```
The code above applies to both OpenAI and Azure OpenAI, just remember to populate the `OPENAI_API_KEY` env variable with the proper key. 
There are multiple ways to control the optimization techniques, they are applied in the follow order of preference:

- You can control the technique you use for optimization by prepending the slug to the model name `{slug}-model-name`. E.g. in the above code we are using `moa` or mixture of agents as the optimization approach. In the proxy logs you will see the following showing the `moa` is been used with the base model as `gpt-4o-mini`.

```bash
2024-09-06 08:35:32,597 - INFO - Using approach moa, with gpt-4o-mini
2024-09-06 08:35:35,358 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2024-09-06 08:35:39,553 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2024-09-06 08:35:44,795 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2024-09-06 08:35:44,797 - INFO - 127.0.0.1 - - [06/Sep/2024 08:35:44] "POST /v1/chat/completions HTTP/1.1" 200 -
```

- Or, you can pass the slug in the `optillm_approach` field in the `extra_body`.

```bash
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[{ "role": "user","content": "" }],
  temperature=0.2,
  extra_body={"optillm_approach": "bon|moa|mcts"}
)
```
- Or, you can just mention the approach in either your `system` or user `prompt`, within `<optillm_approach> </optillm_approach>` tags.

```bash
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[{ "role": "user","content": "<optillm_approach>re2</optillm_approach> How many r's are there in strawberry?" }],
  temperature=0.2
)
```

> [!TIP]
> You can also combine different techniques either by using symbols `&` and `|`. When you use `&` the techniques are processed in the order from left to right in a pipeline
> with response from previous stage used as request to the next. While, with `|` we run all the requests in parallel and generate multiple responses that are returned as a list.

Please note that the convention described above works only when the optillm server has been started with inference approach set to `auto`. Otherwise, the `model` attribute in the client request must be set with the model name only.  

We now suport all LLM providers (by wrapping around the [LiteLLM sdk](https://docs.litellm.ai/docs/#litellm-python-sdk)). E.g. you can use the Gemini Flash model with `moa` by setting passing the api key in the environment variable `os.environ['GEMINI_API_KEY']` and then calling the model `moa-gemini/gemini-1.5-flash-002`. In the output you will then see that LiteLLM is being used to call the base model.

```bash
9:43:21 - LiteLLM:INFO: utils.py:2952 - 
LiteLLM completion() model= gemini-1.5-flash-002; provider = gemini
2024-09-29 19:43:21,011 - INFO - 
LiteLLM completion() model= gemini-1.5-flash-002; provider = gemini
2024-09-29 19:43:21,481 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-002:generateContent?key=[redacted] "HTTP/1.1 200 OK"
19:43:21 - LiteLLM:INFO: utils.py:988 - Wrapper: Completed Call, calling success_handler
2024-09-29 19:43:21,483 - INFO - Wrapper: Completed Call, calling success_handler
19:43:21 - LiteLLM:INFO: utils.py:2952 - 
LiteLLM completion() model= gemini-1.5-flash-002; provider = gemini
```

> [!TIP]
> optillm is a transparent proxy and will work with any LLM API or provider that has an OpenAI API compatible chat completions endpoint, and in turn, optillm also exposes 
the same OpenAI API compatible chat completions endpoint. This should allow you to integrate it into any existing tools or frameworks easily. If the LLM you want to use
doesn't have an OpenAI API compatible endpoint (like Google or Anthropic) you can use [LiteLLM proxy server](https://docs.litellm.ai/docs/proxy/quick_start) that supports most LLMs.

The following sequence diagram illustrates how the request and responses go through optillm. 

![Sequance diagram showing optillm in use](./optillm-sequence-diagram.png)

In the diagram:
- `A` is an existing tool (like [oobabooga](https://github.com/oobabooga/text-generation-webui/)), framework (like [patchwork](https://github.com/patched-codes/patchwork))
or your own code where you want to use the results from optillm. You can use it directly using any OpenAI client sdk.
- `B` is the optillm service (running directly or in a docker container) that will send requests to the `base_url`.
- `C` is any service providing an OpenAI API compatible chat completions endpoint. 

### Generate a synthetic dataset

To generate a synthetic dataset with optillm and vLLM, run the following:

```shell
python generate_dataset.py \
  --approach mcts \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name argilla/magpie-ultra-v0.1 \
  --dataset_column instruction \
  --temperature 0.8 \
  --max_tokens 4096 \
  --num_samples 5
```

By default, this generates _one completion per prompt_ and the outputs will be saved to `data/{model}/`. If you want to generate multiple completions per prompt (e.g. for preference modelling), use the `--n` argument and run:

```shell
python generate_dataset.py \
  --approach mcts \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name argilla/magpie-ultra-v0.1 \
  --dataset_column instruction \
  --temperature 0.8 \
  --max_tokens 4096 \
  --num_samples 5 \
  --n 2 
```

To push the dataset to the Hub, run:

```shell
python generate_dataset.py \
  --approach mcts \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name argilla/magpie-ultra-v0.1 \
  --dataset_column instruction \
  --temperature 0.8 \
  --max_tokens 4096 \
  --num_samples 5 \
  --push_to_hub
```

The results will be visible in the `prompt` and `optillm_completions` columns of the dataset.

Note that some models like Llama3 require a CoT prompt to produce the correct format on MATH. For these cases, use the `--system_prompt` argument to provide a custom system prompt to the model:

```shell
python generate_dataset.py \
  --approach mcts \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name lighteval/MATH \
  --dataset_split test \
  --dataset_column problem \
  --system_prompt "Solve the following math problem step-by-step. Simplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}." \
  --temperature 0.8 \
  --max_tokens 4096 \
  --num_samples 5
```

> [!WARNING]
> The choice of system prompt can have a large impact on the ability of the model to emit an "boxed" answer that can be parsed correctly! The prompt above works quite well for Llama models.


For all script options, run:

```shell
python generate_dataset.py -h
```

#### Debugging and development

To iterate fast on improvements to `generate_dataset.py`, it is best to decouple the vLLM server startup from the data generation process. To achieve this, first run the vLLM server and optillm proxy with:

```shell
python optillm/server_context.py --model_path meta-llama/Llama-3.2-1B-Instruct
```

Then run the data generation script with `--debug` in a separate terminal:

```shell
python generate_dataset.py   \
  --approach mcts \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name argilla/magpie-ultra-v0.1 \
  --dataset_column instruction \
  --num_samples 5 \
  --debug
```


#### Launch a Slurm job

To generate a synthetic dataset with Slurm on the Hugging Face cluster, run:

```shell
sbatch launch.slurm \
  --approach mcts \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name argilla/magpie-ultra-v0.1 \
  --dataset_column instruction \
  --num_samples 5 \
  --push_to_hub
```

## Implemented techniques

| Approach                | Slug               | Description                                                                                    |
| ----------------------- | ------------------ | ---------------------------------------------------------------------------------------------- |
| CoT with Reflection     | `cot_reflection`   | Implements chain-of-thought reasoning with \<thinking\>, \<reflection> and \<output\> sections |
| PlanSearch              | `plansearch`       | Implements a search algorithm over candidate plans for solving a problem in natural language   |
| ReRead                  | `re2`              | Implements rereading to improve reasoning by processing queries twice                          |
| Self-Consistency        | `self_consistency` | Implements an advanced self-consistency method                                                 |
| Z3 Solver               | `z3`               | Utilizes the Z3 theorem prover for logical reasoning                                           |
| R* Algorithm            | `rstar`            | Implements the R* algorithm for problem-solving                                                |
| LEAP                    | `leap`             | Learns task-specific principles from few shot examples                                         |
| Round Trip Optimization | `rto`              | Optimizes responses through a round-trip process                                               |
| Best of N Sampling      | `bon`              | Generates multiple responses and selects the best one                                          |
| Mixture of Agents       | `moa`              | Combines responses from multiple critiques                                                     |
| Monte Carlo Tree Search | `mcts`             | Uses MCTS for decision-making in chat responses                                                |
| PV Game                 | `pvg`              | Applies a prover-verifier game approach at inference time                                      |
| CoT Decoding            |  N/A for proxy     | Implements chain-of-thought decoding to elicit reasoning without explicit prompting            |

## Available Parameters

optillm supports various command-line arguments and environment variables for configuration.

| Parameter                | Description                                                     | Default Value   |
|--------------------------|-----------------------------------------------------------------|-----------------|
| `--approach`             | Inference approach to use                                       | `"auto"`        |
| `--simulations`          | Number of MCTS simulations                                      | 2               |
| `--exploration`          | Exploration weight for MCTS                                     | 0.2             |
| `--depth`                | Simulation depth for MCTS                                       | 1               |
| `--best-of-n`            | Number of samples for best_of_n approach                        | 3               |
| `--model`                | OpenAI model to use                                             | `"gpt-4o-mini"` |
| `--base-url`             | Base URL for OpenAI compatible endpoint                         | `""`            |
| `--rstar-max-depth`      | Maximum depth for rStar algorithm                               | 3               |
| `--rstar-num-rollouts`   | Number of rollouts for rStar algorithm                          | 5               |
| `--rstar-c`              | Exploration constant for rStar algorithm                        | 1.4             |
| `--n`                    | Number of final responses to be returned                        | 1               |
| `--return-full-response` | Return the full response including the CoT with <thinking> tags | `False`         |
| `--port`                 | Specify the port to run the proxy                               | 8000            |
| `--optillm-api-key`      | Optional API key for client authentication to optillm           | `""`            |

When using Docker, these can be set as environment variables prefixed with `OPTILLM_`.

## Running with Docker

optillm can optionally be built and run using Docker and the provided [Dockerfile](./Dockerfile).

### Using Docker Compose

1. Make sure you have Docker and Docker Compose installed on your system.

2. Either update the environment variables in the docker-compose.yaml file or create a `.env` file in the project root directory and add any environment variables you want to set. For example, to set the OpenAI API key, add the following line to the `.env` file:

   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Run the following command to start optillm:

   ```bash
   docker compose up -d
   ```

   This will build the Docker image if it doesn't exist and start the optillm service.

4. optillm will be available at `http://localhost:8000`.

When using Docker, you can set these parameters as environment variables. For example, to set the approach and model, you would use:

```bash
OPTILLM_APPROACH=mcts
OPTILLM_MODEL=gpt-4
```

To secure the optillm proxy with an API key, set the `OPTILLM_API_KEY` environment variable:

```bash
OPTILLM_API_KEY=your_secret_api_key
```

When the API key is set, clients must include it in their requests using the `Authorization` header:

```plain
Authorization: Bearer your_secret_api_key
```
## SOTA results on benchmarks with optillm

### plansearch-gpt-4o-mini on LiveCodeBench (Sep 2024)

| Model                  | pass@1 | pass@5 | pass@10 |
| ---------------------- | ------ | ------ | ------- |
| plansearch-gpt-4o-mini | 44.03  | 59.31  | 63.5    |
| gpt-4o-mini            | 43.9   | 50.61  | 53.25   |
| claude-3.5-sonnet      | 51.3   |        |         |
| gpt-4o-2024-05-13      | 45.2   |        |         |
| gpt-4-turbo-2024-04-09 | 44.2   |        |         |

### moa-gpt-4o-mini on Arena-Hard-Auto (Aug 2024)

![Results showing Mixture of Agents approach using gpt-4o-mini on Arena Hard Auto Benchmark](./moa-results.png)

### optillm with Patchwork (July 2024)

Since optillm is a drop-in replacement for OpenAI API you can easily integrate it with existing tools and frameworks using the OpenAI client. We used optillm with [patchwork](https://github.com/patched-codes/patchwork) which is an open-source framework that automates development gruntwork like PR reviews, bug fixing, security patching using workflows
called patchflows. We saw huge performance gains across all the supported patchflows as shown below when using the mixutre of agents approach (moa). 

![Results showing optillm mixture of agents approach used with patchflows](./moa-patchwork-results.png)

## References

- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200)
- [Re-Reading Improves Reasoning in Large Language Models](https://arxiv.org/abs/2309.06275)
- [In-Context Principle Learning from Mistakes](https://arxiv.org/abs/2402.05403)
- [Planning In Natural Language Improves LLM Search For Code Generation](https://arxiv.org/abs/2409.03733)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195)
- [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)
- [Prover-Verifier Games improve legibility of LLM outputs](https://arxiv.org/abs/2407.13692)
- [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451)
- [Unsupervised Evaluation of Code LLMs with Round-Trip Correctness](https://arxiv.org/abs/2402.08699)
- [Patched MOA: optimizing inference for diverse software development tasks](https://arxiv.org/abs/2407.18521)
- [Patched RTC: evaluating LLMs for diverse software development tasks](https://arxiv.org/abs/2407.16557)
