# Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning [NeurIPS 2024]

[Paper on Arxiv](https://arxiv.org/abs/2402.06025) -- Accepted at NeurIPS 2024

## Installation

Create a conda environment and install the requirements and openai-hf-interface submodule
```
conda create -n de_rr python=3.8
conda activate de_rr
pip install -r requirements.txt
cd openai-hf-interface
pip install -e .
```

To run our algorithms, you need to provide OpenAI API key because our algorithms use gpt-4.
We use `openai-hf-interface`, a prompting package that handles sending requests to LLM APIs, in this repo, so you can provide your OpenAI API key by creating `secrets.json` file inside `openai-hf-interface` directory. The json file should have:

```
{
    "openai_api_key": "put-your-key-here",
}
```
Please see [openai-hf-interface](https://github.com/topwasu/openai-hf-interface) for more information.

## Running

Note: please use the `acre` branch to run the ActiveACRE experiments and use the `main` branch to run the Zendo experiments.

Call the below command to run an experiment:
```
python run.py agent/proposal=particle_filter # To run online, fuzzy method on Zendo
python run.py dataset=acre agent=acre agent/proposal=particle_filter # To run online, fuzzy method on ActiveACRE
```

