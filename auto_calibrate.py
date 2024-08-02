import argparse
import pandas as pd
import random
import json
import re
from tqdm import tqdm
from typing import List, Tuple
from prompt_templates import *
from correlation_metric import kendall_tau, spearman_correlation
import tagging_guidelines
from utils import *
from generate import predict
import os

DASHED_LINE = "- " * 50


def extract_score(input_string: str, key="Relevance") -> float:
    """
    Extracts the score from a given string using regex.
    Returns -1 if the input is invalid or score is not found.
    """
    if not isinstance(input_string, str):
        print(f"Invalid input: {input_string}")
        return -1

    match = re.search(r"\d+", input_string)
    if match:
        return float(match.group())
    else:
        return -1


def extract_content_from_response(response):
    """
    Extracts content from the response object.
    """
    try:
        return response.choices[0].message.content
    except:
        print(response)


def postprocess_llm_response(response):
    """
    Processes the LLM response and returns the first key if response is a dict.
    Otherwise, attempts to load as JSON and return the first key.
    """
    if isinstance(response, dict):
        return list(response.keys())[0]
    else:
        try:
            x = json.loads(response)
            return list(x.keys())[0]
        except:
            return str(response)


def evaluate_candidate_cot(
    aspect, candidate_cot, df_dev, model_name, test_size=3
) -> float:
    """
    Evaluates the candidate COT and returns average correlation scores.
    """
    actual_scores = []
    predicted_scores = []
    exemplars = []

    # Select appropriate scoring prompt based on aspect
    scoring_prompt = None
    if aspect == "Relevance":
        scoring_prompt = relevance_score_prompt_cot(evaluation_steps=candidate_cot)
    elif aspect == "Aggressiveness":
        scoring_prompt = aggressiveness_score_prompt_cot(evaluation_steps=candidate_cot)
    elif aspect == "Coherence":
        scoring_prompt = coherence_score_prompt_cot(evaluation_steps=candidate_cot)
    elif aspect == "Suitableness":
        scoring_prompt = suitableness_score_prompt_cot(evaluation_steps=candidate_cot)

    print(f"Evaluating on {test_size} datapoints")
    for _, row in tqdm(df_dev[:test_size].iterrows(), desc="Getting Predictions"):
        input_prompt = common_input_prompt(
            row["hatespeech"], row["predicted_counterspeech"]
        )

        response = predict(
            prompt=input_prompt,
            system_description=scoring_prompt,
            model_name=model_name,
        )
        predicted_score = extract_score(response)
        predicted_scores.append(predicted_score)

        # Append actual scores based on aspect
        if aspect == "Relevance":
            actual_score = row["relevance_score"]
        elif aspect == "Aggressiveness":
            actual_score = row["aggressiveness_score"]
        elif aspect == "Coherence":
            actual_score = row["coherence_score"]
        elif aspect == "Suitableness":
            actual_score = row["suitableness_score"]

        actual_scores.append(actual_score)
        exemplars.append(
            (
                row["hatespeech"],
                row["predicted_counterspeech"],
                actual_score,
                predicted_score,
            )
        )

    # Calculate correlation scores
    spearman_score = spearman_correlation(actual_scores, predicted_scores)
    kendalltau_score = kendall_tau(actual_scores, predicted_scores)
    mean_score = (spearman_score + kendalltau_score) / 2

    return mean_score, exemplars


def iterative_refinement_of_cot_candidate(
    candidate_cot, aspect, num_trials, df_dev, few_shot_size, test_size, model_name
) -> List[str]:
    """
    Performs iterative refinement of top candidate via self-assessment prompting.
    Returns topk candidates with highest correlation.
    """
    exemplars = candidate_cot[2]
    refinement_prompts = []
    candidate_cot_set = []

    for trial in range(num_trials):
        print(f"Trial {trial + 1}:")

        # Randomly sample few-shot examples from the golden set
        sampled_examples = random.sample(exemplars, few_shot_size)

        # Infer criteria based on sampled examples
        hs1, cs1, r1, p1 = sampled_examples[0]
        hs2, cs2, r2, p2 = sampled_examples[1]
        hs3, cs3, r3, p3 = sampled_examples[2]

        few_shot_examples = cot_error_few_shot_tempate(
            aspect, hs1, cs1, r1, p1, hs2, cs2, r2, p2, hs3, cs3, r3, p3
        )
        cot_refinement_prompt = candidate_cot_refinement_template(
            aspect, candidate_cot[0], few_shot_examples
        )
        refinement_prompts.append(cot_refinement_prompt)

        # Generate and postprocess refined candidate COT
        candidate_cot_refined = predict(
            prompt=cot_refinement_prompt, system_description="", model_name=model_name
        )
        print(DASHED_LINE)
        print(f"Refined version of CoT {trial + 1}:", candidate_cot_refined)
        print(DASHED_LINE)
        candidate_cot_refined = postprocess_llm_response(candidate_cot_refined)
        candidate_cot_set.append(candidate_cot_refined)

    # Evaluate refined candidates and select the top one
    evaluated_candidates = [
        (
            candidate_cot,
            *evaluate_candidate_cot(
                aspect, candidate_cot, df_dev, model_name, test_size
            ),
        )
        for candidate_cot in candidate_cot_set
    ]
    evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidate = evaluated_candidates[0]

    return top_candidate


def auto_calibrate_pipeline(
    data_path, aspect, model_name, num_trials, few_shot_size, topk, test_size
):
    DASHED_LINE = "- " * 50

    # Initialize tqdm for progress bars
    tqdm.pandas()
    os.environ["http_proxy"] = "http://proxy61.iitd.ac.in:3128"
    os.environ["https_proxy"] = "http://proxy61.iitd.ac.in:3128"

    # Load the dataset
    df_dev = pd.read_csv(data_path)

    # Prepare lists to store scores
    dev_set_gold_relevance_score = []
    dev_set_gold_coherence_score = []
    dev_set_gold_aggressiveness_score = []
    dev_set_gold_suitableness_score = []

    # Populate lists with data from the dataframe
    for _, row in df_dev.iterrows():
        dev_set_gold_relevance_score.append(
            (row["hatespeech"], row["predicted_counterspeech"], row["relevance_score"])
        )
        dev_set_gold_coherence_score.append(
            (row["hatespeech"], row["predicted_counterspeech"], row["coherence_score"])
        )
        dev_set_gold_aggressiveness_score.append(
            (
                row["hatespeech"],
                row["predicted_counterspeech"],
                row["aggressiveness_score"],
            )
        )
        dev_set_gold_suitableness_score.append(
            (
                row["hatespeech"],
                row["predicted_counterspeech"],
                row["suitableness_score"],
            )
        )

    # Set the gold data and evaluation criteria
    dev_set_gold = dev_set_gold_relevance_score
    evaluation_criteria = evaluation_criteria_relevance

    # Perform Monte Carlo trials
    criteria_set = []
    candidate_cot_set = []

    print("Starting Monte Carlo Trials")
    for trial in range(num_trials):
        print(f"Trial {trial + 1}:")

        # Randomly sample few-shot examples from the golden set
        sampled_examples = random.sample(dev_set_gold, few_shot_size)

        # Infer criteria based on sampled examples
        hs1, cs1, r1 = sampled_examples[0]
        hs2, cs2, r2 = sampled_examples[1]
        hs3, cs3, r3 = sampled_examples[2]

        few_shot_examples = cot_gen_few_shot_tempate(
            aspect, hs1, cs1, r1, hs2, cs2, r2, hs3, cs3, r3
        )
        criteria_example = get_criteria_example(aspect)
        criteria = candidate_cot_drafting_template(
            aspect, evaluation_criteria, criteria_example, few_shot_examples
        )

        # Store the inferred criteria
        criteria_set.append(criteria)

        # Generate and postprocess candidate COT
        candidate_cot = predict(
            prompt=criteria,
            system_description="",
            use_temperature_samping=True,
            model_name=model_name,
        )
        candidate_cot = postprocess_llm_response(candidate_cot)
        candidate_cot_set.append(candidate_cot)

        print(f"Sampled Examples: {sampled_examples}")
        print(DASHED_LINE)
        print(f"Inferred Criteria: {criteria}")
        print(DASHED_LINE)
        print(DASHED_LINE)
        print(f"Candidate COT: {candidate_cot}")
        print(DASHED_LINE)
        print(DASHED_LINE)

    # Evaluate candidates
    print(f"Total candidates to evaluate: {len(candidate_cot_set)}")
    evaluated_candidates = [
        (
            candidate_cot,
            *evaluate_candidate_cot(
                aspect, candidate_cot, df_dev, model_name, test_size
            ),
        )
        for candidate_cot in candidate_cot_set
    ]
    evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = evaluated_candidates[:topk]

    print("Top-performing CoT:")
    for candidate_cot, score, exemplars in top_candidates:
        print(DASHED_LINE)
        print(f"Candidate COT: {candidate_cot}, Score: {score}, Exempars: {exemplars}")
        print(DASHED_LINE)

    # Perform iterative refinement on top candidates
    top_candidates_refined = []
    for candidate_cot in top_candidates:
        refined_cot_candidate = iterative_refinement_of_cot_candidate(
            candidate_cot,
            aspect,
            num_trials,
            df_dev,
            few_shot_size,
            test_size,
            model_name,
        )
        top_candidates_refined.append(refined_cot_candidate)

    top_candidates_refined.sort(key=lambda x: x[1], reverse=True)
    top_candidates = top_candidates_refined[:topk]
    top_candidate_global = top_candidates_refined[0]

    return top_candidate_global


if __name__ == "__main__":
    valid_aspects = ["Relevance", "Coherence", "Aggressiveness", "Suitableness"]

    parser = argparse.ArgumentParser(description="Auto Calibrate Pipeline")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/annotations/dev_set_500.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--aspect",
        choices=valid_aspects,
        help="Aspect must be one of: Relevance, Coherence, Aggressiveness, Suitableness",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Model name for prediction",
    )
    parser.add_argument(
        "--num_trials", type=int, default=2, help="Number of Monte Carlo trials"
    )
    parser.add_argument("--few_shot_size", type=int, default=3, help="Few shot size")
    parser.add_argument(
        "--topk", type=int, default=3, help="Number of top candidates to select"
    )
    parser.add_argument(
        "--test_size", type=int, default=3, help="Test size for evaluation"
    )

    args = parser.parse_args()

    top_candidate_global = auto_calibrate_pipeline(
        data_path=args.data_path,
        aspect=args.aspect,
        model_name=args.model_name,
        num_trials=args.num_trials,
        few_shot_size=args.few_shot_size,
        topk=args.topk,
        test_size=args.test_size,
    )

    print("Final Top Candidate Global:")
    print(top_candidate_global)


# Sample Usage for different evaluation metrics
"""
python auto_calibrate.py --data_path /home/amey/depository/cs-eval/data/annotations/dev_set_500.csv \
    --aspect Relevance \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --num_trials 10 \
    --few_shot_size 3 \
    --topk 3 \
    --test_size 300 \
    ;

python auto_calibrate.py --data_path /home/amey/depository/cs-eval/data/annotations/dev_set_500.csv \
    --aspect Coherence \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --num_trials 10 \
    --few_shot_size 3 \
    --topk 3 \
    --test_size 300 \
    ;

python auto_calibrate.py --data_path /home/amey/depository/cs-eval/data/annotations/dev_set_500.csv \
    --aspect Suitableness \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --num_trials 10 \
    --few_shot_size 3 \
    --topk 3 \
    --test_size 300 \
    ;

python auto_calibrate.py --data_path /home/amey/depository/cs-eval/data/annotations/dev_set_500.csv \
    --aspect Aggressiveness \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --num_trials 10 \
    --few_shot_size 3 \
    --topk 3 \
    --test_size 300 \
    ;
"""
