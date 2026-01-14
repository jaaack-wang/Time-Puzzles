import os
import logging
import argparse
import pandas as pd
from time import sleep
from pathlib import Path
from tqdm import tqdm
from typing import List

from src.llms import get_llm_response_function
from src.common.timer import time_experiment_in_human_readable
from src.common.data_utils import generate_unique_id


logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s] - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


PROMPT = '''
From the time-related constraints below, determine all valid date(s) (if any) that satisfy them. \
Depending on the conditions, the result may be zero, one, or multiple dates. \
Unless otherwise stated, interpret all constraints using the Gregorian calendar.

Note: Seasons are defined as:
- Winter: December, January, February
- Spring: March, April, May
- Summer: June, July, August
- Autumn: September, October, November

The constraints are as follows:

{constraints}

Carefully review the constraints and reason step-by-step to identify all valid date(s). \
After thorough consideration, end your response on a new line with "MY ANSWER: " \
followed by the valid date(s) in the format "YYYY-MM-DD". \
If there are multiple valid dates, list them separated by commas. \
If no valid date exists, respond with "MY ANSWER: None".
'''.strip()


EXPERIMENT_CONDITIONS = [
    "no_tool_use",
    "with_web_search",
    "with_pseudo_gpt_web_search_context",
    "with_code_interpreter",
    "with_web_search_and_code_interpreter",
    "with_pseudo_gpt_web_search_context_and_code_interpreter",
]

# TODO. Correspond to the **kwargs in get_llm_response_function
EXPERIMENT_CONDITION_2_CONFIGS = {
    "no_tool_use": {},
    "with_web_search": {},
    "with_code_interpreter": {},
    "with_web_search_and_code_interpreter": {},
}


def is_supported_experiment_condition(condition: str) -> bool:
    return condition in EXPERIMENT_CONDITIONS


def format_constraints(constraints: List[str], marker="-") -> str:
    assert len(constraints) > 0, "Constraints list is empty."
    return "\n".join([f"{marker} {c}" for c in constraints])


def get_args():
    parser = argparse.ArgumentParser(description="Experiment without tool use")
    parser.add_argument("--input_fp", type=str, required=True,
                        help="File path to the input data. Must be in Json format.")
    
    parser.add_argument("--condition", type=str, default="no_tool_use",
                        help="Experiment condition to use. Default is no_tool_use.")
    
    parser.add_argument("--use_explicit_constraints", action="store_true", 
                        help="Flag to indicate using explicit constraints in the prompt.")
    
    parser.add_argument("--model_name", type=str, 
                        default="gpt-5-nano-2025-08-07",
                        help="Name of the model to use. Default is gpt-5-nano-2025-08-07.")

    parser.add_argument("--web_search_gpt_model_name", type=str,
                        default="gpt-5-2025-08-07",
                        help="Name of the GPT model used for web search.")

    parser.add_argument("--solution_counts_to_include", type=int, nargs="+", 
                        default=[1,2,3,4,5,6], help="List of solution counts to include in the experiment.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save output files. Default is 'outputs'.")
    
    parser.add_argument("--max_tries_per_llm_call", type=int, default=5,
                        help="Maximum number of tries per LLM call in case of failure.")
    
    parser.add_argument("--test_run", action="store_true",
                        help="Flag to indicate a test run with fewer iterations")
    
    parser.add_argument("--num_test_samples", type=int, default=5,
                        help="Number of samples to use in test run")

    parser.add_argument("--save_freq", type=int, default=5,
                        help="Frequency (in number of samples) to save intermediate results.")
    
    parser.add_argument("--cuda_visible_devices", type=str, nargs="+", default=["0"],
                        help="List of CUDA visible devices to use.")
    
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.cuda_visible_devices)

    # ============== pre-experiment checks ==============

    ### check input file format
    assert args.input_fp.endswith(".json"), "Input file must be in Json format."
    assert is_supported_experiment_condition(args.condition), \
        f"Unsupported experiment condition: {args.condition}.\n" \
        f"Supported conditions are: {EXPERIMENT_CONDITIONS}"

    ### check if model is supported

    condition = args.condition
    input_fn = Path(args.input_fp).name.replace(".json", "")
    
    if condition.startswith("with_pseudo_gpt_web_search_context"):

        if condition == "with_pseudo_gpt_web_search_context":
            condition = "no_tool_use"
        elif condition == "with_pseudo_gpt_web_search_context_and_code_interpreter":
            condition = "with_code_interpreter"
            args.web_search_gpt_model_name = args.model_name
        else:
            raise ValueError(
                f"Invalid condition for pseudo GPT web search context: {condition}"
            )
        
        context_fp = Path(args.output_dir) / input_fn / "with_web_search" / f"{args.web_search_gpt_model_name}_cited_web_search_context.json"

        if not context_fp.exists():
            raise ValueError(
                "Pseudo GPT web search context file not found or invalid. "
                "Please run 'generate_pseudo_gpt_web_search_context.py' first."
            )
        context_df = pd.read_json(context_fp)
        context_dict = dict(
            zip(context_df["identifier"], context_df["web_search_context"])
        )
        logging.info("Loaded pseudo GPT web search contexts for experiment.")

    llm_response_function = get_llm_response_function(
        llm_name=args.model_name,
        experiment_condition=condition,
        tensor_parallel_size=len(args.cuda_visible_devices),
        **EXPERIMENT_CONDITION_2_CONFIGS.get(condition, {})
    )

    # ============= load and handle input data ==============
    input_data = pd.read_json(args.input_fp)
    assert "constraints" in input_data.columns, \
        "Input data must contain 'constraints' column."

    if args.use_explicit_constraints:
        assert "explicit_constraints" in input_data.columns, \
            "Input data must contain 'explicit_constraints' column."

    if "identifier" not in input_data.columns:
        input_data["identifier"] = [
            generate_unique_id() for _ in range(len(input_data))
        ]
        input_data.to_json(args.input_fp, orient="records", lines=False)
        logging.info("'identifier' column not found in input data. "
                     "Generated unique identifiers for each sample.")
    
    input_data = input_data[input_data["solution_count"].isin(args.solution_counts_to_include)]
    logging.info(f"Loaded {len(input_data)} samples from {args.input_fp}.")

    if args.test_run:
        input_data = input_data.sample(min(args.num_test_samples, len(input_data)))
        logging.info(f"Test run enabled. Using {len(input_data)} random samples.")

    # ============== setup output directories ==============

    if "/" in args.model_name:
        args.model_name = args.model_name.split("/")[-1]
    
    condition = args.condition

    if condition == "with_pseudo_gpt_web_search_context":
        condition += f"{args.web_search_gpt_model_name}"
    
    if args.use_explicit_constraints:
        condition += "_with_explicit_constraints"

    output_fp = Path(args.output_dir) / input_fn / condition / f"{args.model_name}.json"

    if not output_fp.exists():
        os.makedirs(output_fp.parent, exist_ok=True)
        logging.info(f"Created output directory at {output_fp.parent}.")
        output_data = pd.DataFrame(columns=[
            "identifier", "prompt", "constraints", "solutions", 
            "solution_count", "response", "response_detailed"
        ])
    else:
        output_data = pd.read_json(output_fp)
        if len(output_data) == 0:
            output_data = pd.DataFrame(columns=[
                "identifier", "prompt", "constraints", "solutions", 
                "solution_count", "response", "response_detailed"
            ])
        logging.info(f"Output file {output_fp} exists. Loaded {len(output_data)} existing samples.")
    
    all_sample_ids = set(input_data["identifier"].tolist())
    if "identifier" in output_data.columns:
        output_data = output_data.dropna(subset=["response"])
        completed_sample_ids = set(output_data["identifier"].tolist())
    else:
        completed_sample_ids = set()
    
    pending_sample_ids = all_sample_ids - completed_sample_ids

    if len(pending_sample_ids) == 0:
        logging.info("All samples have been processed. Exiting experiment.")
        return
    elif len(completed_sample_ids) > 0:
        logging.info(f"{len(completed_sample_ids)} samples already completed. "
                     f"{len(pending_sample_ids)} samples pending.")

    # ============== run experiment ==============
    constraints_col = "explicit_constraints" if args.use_explicit_constraints else "constraints"

    try:
        pending_data = input_data[input_data["identifier"].isin(pending_sample_ids)]

        for idx, (_, row) in enumerate(tqdm(pending_data.iterrows(), total=len(pending_data))):
            sample_id = row["identifier"]
            constraints = row[constraints_col]
            solutions = row["solutions"]
            solution_count = row["solution_count"]

            if isinstance(constraints, list):
                constraints = format_constraints(constraints)

            prompt = PROMPT.format(constraints=constraints)

            if args.condition in ["with_pseudo_gpt_web_search_context", 
                                  "with_pseudo_gpt_web_search_context_and_code_interpreter"]:
                web_search_context = context_dict.get(sample_id, "")

                if web_search_context.strip() == "":
                    logging.info(
                        f"No pseudo GPT web search context found for sample {sample_id}. Skipping."
                    )
                    continue

                prompt = f"Here is some web search context that may help you:\n\n{web_search_context}\n\n{prompt}"

            success = False
            for attempt in range(args.max_tries_per_llm_call):
                try:
                    response, output_text = llm_response_function(prompt)
                    success = True
                    break
                except Exception as e:
                    logging.warning(f"LLM call failed for sample {sample_id} on attempt {attempt + 1}: {e}")
                    sleep(5)  # wait before retrying

            if not success:
                logging.error(f"Failed to get LLM response for sample {sample_id} after "
                            f"{args.max_tries_per_llm_call} attempts. Skipping sample.")
                continue

            output_row = {
                "identifier": sample_id,
                "prompt": prompt,
                "constraints": constraints,
                "solutions": solutions,
                "solution_count": solution_count,
                "response": output_text,
                "response_detailed": response,
            }

            output_data.loc[len(output_data)] = output_row

            # Periodically save output data
            if (idx + 1) % args.save_freq == 0:
                output_data.to_json(output_fp, orient="records", lines=False, indent=4)

    except Exception as e:
        logging.error(f"Experiment interrupted due to error: {e}")
    
    finally:
        # Final save
        output_data.to_json(output_fp, orient="records", lines=False, indent=4)
        logging.info(f"Experiment completed. Final output saved to {output_fp}.")


if __name__ == "__main__":
    time_experiment_in_human_readable(main)