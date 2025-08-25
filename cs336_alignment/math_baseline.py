"""
Write a script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH. 
This scriptshould: 
(1) load the MATH validation examples from /data/a5-alignment/MATH/validation.jsonl,
(2) format them as string prompts to the language model using the r1_zero prompt, and 
(3) gen-erate outputs for each example. This script should also 
(4) calculate evaluation metrics and
(5) serialize the examples, model generations, and corresponding evaluation scores to disk for
analysis in subsequent problems.
"""

from vllm import LLM, SamplingParams
from typing import Callable, Iterable
from .common import MODEL_PATH, PROMPT_PATH, LOG_PATH, DATA_PATH
from .drgrpo_grader import r1_zero_reward_fn
import json
import re
from datasets import load_dataset

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str] | Iterable[str],
    eval_sampling_params: SamplingParams,
    save_path: str | None,
    ground_truths: list[str] | Iterable[str] | None = None,
) -> None:
    """
    Evaluate a model on a list of prompts,
    compute evaluation matrics and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)

    results = []

    for output, ground_truth in zip(outputs, ground_truths):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        rewards: dict[str, float] = reward_fn(generated_text, ground_truth)

        metainfo = {
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
        }

        results.append( rewards | metainfo )

        # serialize results to disk
        with open(save_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print("Results saved to", save_path)

def main():
    # load the MATH validation examples
    dataset = load_dataset(
        "json", 
        data_files={
            "train": str(DATA_PATH / "gsm8k/train.jsonl" ),
            "test": str( DATA_PATH / "gsm8k/test.jsonl" ),
        }
    )
    test_data = dataset["test"]

    # format them as string prompts to the language model using the r1_zero prompt
    with open(PROMPT_PATH / "r1_zero.prompt", "r") as f:
        r1_zero_prompt_template = f.read()
    prompts = []
    ground_truths = []

    for item in test_data:
        prompt = r1_zero_prompt_template.format(item["question"])
        # parse the short ground truth answer (e.g. #### 3.14)
        match = re.search(r"####\s*(\d+(\.\d+)?)", item["answer"])
        if match:
            ground_truth = match.group(1)
            prompts.append(prompt)
            ground_truths.append(ground_truth)

    print("Loaded", len(prompts), "examples")

    # analyze
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, 
        stop=["</answer>"], include_stop_str_in_output=True
    )

    llm = LLM( model= str(MODEL_PATH / "qwen-2.5-math-1.5b-r1-zero.ggmlv3.q4_0.bin") )

    save_path = str(LOG_PATH / "qwen-2.5-math-1.5b-r1-zero-eval.jsonl")

    evaluate_vllm(llm, r1_zero_reward_fn, 
                  prompts, ground_truths,
                  sampling_params, save_path )

    