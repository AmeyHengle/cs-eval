import pandas as pd

common_input_prompt = (
    lambda hs, cs: f"""
Hate Speech: "{hs}"\nCounterspeech: "{cs}"
""".strip()
)

instruction_prompt = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). Your
task is to rate the counterspeech on one metric. Please make sure you read and understand these instructions carefully.
""".strip()

evaluation_criteria_relevance = f"""
Contextual Relevance (1-5) - This dimension evaluates whether the counterspeech addresses the central topic, theme subject of the given hate
speech.
""".strip()

evaluation_criteria_aggressiveness = f"""
Aggressiveness (1-5) - This metric assesses the degree of confrontational or inflammatory content in the counterspeech. It considers factors such
as abusive language, intensity of disagreement, tone, engagement style, and the presence of personal attacks, with higher scores indicating
greater aggression.
""".strip()

evaluation_criteria_coherence = f"""
Argument Coherence (1-5) - This metric assesses how logically and smoothly the ideas or arguments within the counterspeech connect and
flow. A coherent counterspeech will present its arguments in an organized manner, making it easy for the reader to follow and understand the
counter-narrative being presented.
""".strip()

evaluation_criteria_suitableness = f"""
Suitableness (1-3) - This metric measures the likelihood of an annotator choosing a given counterspeech for direct use (without editing) in a
real scenario against online hate speech. This assessment considers the counterspeech's suitability, appropriateness, and potential impact on a
reader in a real-world context.
""".strip()

# ----------------------------------------------------------------------- #

relevance_score_prompt_zs = f"""
{instruction_prompt}

Evaluation Criteria:
{evaluation_criteria_relevance}

Ensure that the response is STRICTLY in JSON format as {{"Relevance": ""}}
""".strip()

coherence_score_prompt_zs = f"""
{instruction_prompt}

Evaluation Criteria:
{evaluation_criteria_coherence}

Ensure that the response is STRICTLY in JSON format as {{"Argument Coherence": ""}}.
""".strip()

suitableness_score_prompt_zs = f"""
{instruction_prompt}

Evaluation Criteria:
{evaluation_criteria_suitableness}

Ensure that the response is STRICTLY in JSON format as {{"Suitableness": ""}}.
""".strip()

aggressiveness_score_prompt_zs = f"""
{instruction_prompt}

Evaluation Criteria:
{evaluation_criteria_aggressiveness}

Ensure that the response is STRICTLY in JSON format as {{"Aggressiveness": ""}}.
""".strip()

# ----------------------------------------------------------------------- #

relevance_score_prompt_cot = (
    lambda evaluation_steps: f"""
{instruction_prompt}

Evaluation Criteria:
{evaluation_criteria_relevance}

Evaluation Steps:
{evaluation_steps}

Ensure that the response is STRICTLY in JSON format as {{"Relevance": ""}}
""".strip()
)

coherence_score_prompt_cot = (
    lambda evaluation_steps: f"""
{instruction_prompt}

Evaluation Criteria:
{evaluation_criteria_coherence}

Evaluation Steps:
{evaluation_steps}

Ensure that the response is STRICTLY in JSON format as {{"Argument Coherence": ""}}.
""".strip()
)

suitableness_score_prompt_cot = (
    lambda evaluation_steps: f"""
{instruction_prompt}

Evaluation Criteria:
{evaluation_criteria_suitableness}

Evaluation Steps:
{evaluation_steps}

Ensure that the response is STRICTLY in JSON format as {{"Suitableness": ""}}.
""".strip()
)

aggressiveness_score_prompt_cot = (
    lambda evaluation_steps: f"""
{instruction_prompt}

Evaluation Criteria:
{evaluation_criteria_aggressiveness}

Evaluation Steps:
{evaluation_steps}

Ensure that the response is STRICTLY in JSON format as {{"Aggressiveness": ""}}.
""".strip()
)

# ----------------------------------------------------------------------- #

cot_gen_few_shot_tempate = (
    lambda aspect, hs1, cs1, r1, hs2, cs2, r2, hs3, cs3, r3: f"""
Hatespeech: {hs1}
Counterspeech: {cs1}
Expert Rating ({aspect}): {r1}

Hatespeech: {hs2}
Counterspeech: {cs2}
Expert Rating ({aspect}): {r2}

Hatespeech: {hs3}
Counterspeech: {cs3}
Expert Rating ({aspect}): {r3}
""".strip()
)

candidate_cot_drafting_template = (
    lambda aspect, evaluation_criteria, criteria_example, few_shot_examples: f"""
You will be given sets of in-context examples, each containing a hate speech, a corresponding
counterspeech, and an {aspect} rated by a human expert. Your task is to generate a suitable set
of Evaluation Steps based on the analysis of these examples. Please make sure you read and
understand these instructions carefully.

## Examples:
{few_shot_examples}

## Instruction:
Analyze the examples below to identify patterns and factors that influence the {aspect} rating.
Then, create a detailed set of steps outlining how to evaluate the {aspect} of counterspeeches.

## Evaluation Criteria for {aspect}:
{evaluation_criteria}

## Example Evaluation Criteria:
{criteria_example}

DO NOT INCLUDE Exemplars in your response. Output only the Evaluation Steps. 
""".strip()
)

cot_error_few_shot_tempate = (
    lambda aspect, hs1, cs1, r1, p1, hs2, cs2, r2, p2, hs3, cs3, r3, p3: f"""
Hatespeech: {hs1}
Counterspeech: {cs1}
Actual Rating ({aspect}): {r1}
Predicted Rating ({aspect}): {p1}

Hatespeech: {hs2}
Counterspeech: {cs2}
Actual Rating ({aspect}): {r2}
Predicted Rating ({aspect}): {p2}

Hatespeech: {hs3}
Counterspeech: {cs3}
Actual Rating ({aspect}): {r3}
Predicted Rating ({aspect}): {p3}

""".strip()
)

candidate_cot_refinement_template = (
    lambda aspect, candidate_cot, few_shot_examples: f"""
Please refine and improve the chain-of-thought (CoT) evaluation steps used by a large language model in evaluating {aspect} of counterspeech
generation.

Large language models (LLMs) are powerful neural models that can evaluate the quality of counterspeech generation. However, LLMs may not
always agree with human judgments. Please refine the CoT used by LLMs to improve its correlation with human expert scores. To refine the
scoring criteria used by the LLM in evaluating the {aspect}, please follow the following instructions step-by-step:
1. Carefully read each example, understand each hate speech and its corresponding counterspeech, and get your initial assessment of its
quality on {aspect}.
2. Compare the test score obtained by the LLM according to the CoT and the ground-truth score from human experts. Please think why the
correlation is limited by using the current CoT, and how can you improve the CoT to increase the correlation between LLM’s score and
human expert score. If there is a small gap or no gap, this means the CoT work well in this case.
3. Read all of the test cases and rethink how you could refine the current CoT based on your observations and analysis. Then, refine the
CoT to make it concise, accurate, and consistent with human judgments. When refining the CoT, you can do the following: 1)
modification: adjust some parts of the CoT to increase its correlation with the scoring CoT that you think might used by human experts;
2) paraphrase: if the CoT is good enough, you can consider paraphrasing it to make more concise and easy to understand; 3) adding
aspects or details: if you find some new underlying scoring rules not covered by the current CoT, consider adding them as a new line of
injecting to current CoT, but make sure not to make the CoT too long and redundant; 4) calibrate: you can take other methods you think
being helpful to improve the correlation with human experts.
Please return only your refined criteria without any additional sentences.

Old Criteria (CoT):
{candidate_cot}

Error Examples:
{few_shot_examples}

DO NOT INCLUDE Exemplars or any explanation in your response. Just output the refined version of Old Criteria (CoT).
""".strip()
)
