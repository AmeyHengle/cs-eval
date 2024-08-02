common_input_prompt = (
    lambda hs, cs: f"""
Hate Speech: "{hs}"\nCounterspeech: "{cs}"
""".strip()
)

relevance_score_prompt_zs = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on 'Relevance' metric. 

Evaluation Criteria:
Relevance (1-5) - This dimension evaluates whether the counterspeech addresses the central theme or subject of the hate speech. 

Ensure that the response is STRICTLY in JSON format as {{"Relevance": ""}}
""".strip()

coherence_score_prompt_zs = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on 'Argument Coherence' metric. 

Evaluation Criteria:
Argument Coherence (1-5) - Argument Coherence measures how logically the arguments within the counterspeech connect and support each other towards refuting the hate speech A coherent counterspeech will present its arguments in a clear and logically structured manner, making it easy for the reader to follow and understand the counter-narrative being presented.

Ensure that the response is STRICTLY in JSON format as {{"Argument Coherence": ""}}.
""".strip()

suitableness_score_prompt_zs = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on the 'Choose or Not' metric. 

Evaluation Criteria:
Choose or Not  (1-3) -  This metric measures the likelihood of an annotator choosing a given counterspeech for direct use (without editing) in a real scenario against online hate speech. This assessment considers the counterspeech's suitability, appropriateness, and potential impact on a reader in a real-world context.

Ensure that the response is STRICTLY in JSON format as {{"Choose or Not": ""}}.
""".strip()

aggressiveness_score_prompt_zs = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on 'Aggressiveness' metric. 

Evaluation Criteria:
Aggressiveness (1-5) - This metric assesses the degree of confrontational or inflammatory content in the counterspeech. It considers factors such as abusive language, intensity of disagreement, tone, engagement style, and the presence of personal attacks, with higher scores indicating greater aggression.

Ensure that the response is STRICTLY in JSON format as {{"Aggressiveness": ""}}.
""".strip()

relevance_score_prompt_gpt4 = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 
Please make sure you read and understand these instructions carefully. 

Evaluation Criteria:
Relevance (1-5) - This dimension evaluates whether the counterspeech addresses the central theme or subject of the hate speech. 

Evaluation Steps:
Read and Understand the Hate Speech: Thoroughly read the hate speech to grasp its central theme, subject, or specific point of hate, including the target group, type of hateful message, and context.

Score 5 (Highly Relevant): The counterspeech directly addresses the central theme or subject of the hate speech.
Score 4 (Mostly Relevant): The counterspeech mainly addresses the central theme but includes minor unrelated elements.
Score 3 (Moderately Relevant): The counterspeech addresses the hate speech but contains significant unrelated content or deviates from the main point.
Score 2 (Slightly Relevant): The counterspeech only partially addresses the hate speech, with much content being unrelated or off-topic.
Score 1 (Not Relevant): The counterspeech fails to address the central theme or subject of the hate speech and is mostly or entirely off-topic.
Consider Contextual Factors: Evaluate any contextual factors like cultural references, historical events, or social nuances that might affect the relevance, especially if specific references or subtleties are involved.

Finalize Your Score: After careful consideration, assign a relevance score from 1 to 5.

Ensure that the response is STRICTLY in JSON format as {{"Relevance": ""}}
""".strip()

relevance_score_prompt_llama3 = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 
Please make sure you read and understand these instructions carefully. 

Evaluation Criteria:
Relevance (1-5) - This dimension evaluates whether the counterspeech addresses the central theme or subject of the hate speech. 

Evaluation Steps:
Initial Analysis: Start by thoroughly reading and understanding the hate speech. Identify the central theme, subject, or point being addressed, including the target group, type of hateful message (e.g., stereotypes, derogatory remarks), and context in which it is presented.
Relevance Assessment (1-5):
Highly Relevant (5): The counterspeech directly addresses the central theme or subject of the hate speech.
Mostly Relevant (4): The counterspeech primarily addresses the central theme of the hate speech, with some minor deviations or unrelated elements.
Moderately Relevant (3): The counterspeech addresses the hate speech, but also includes significant unrelated content or deviates from the main point at times.
Partially Relevant (2): The counterspeech only partially addresses the hate speech, with a substantial portion of the content being unrelated or off-topic.
Not Relevant (1): The counterspeech fails to address the central theme or subject of the hate speech, being largely or entirely off-topic.
Contextual Considerations: When evaluating relevance, consider any contextual factors that may impact the assessment, such as cultural references, historical events, or social nuances, especially if the hate speech or counterspeech involves specific references or subtleties.

Ensure that the response is STRICTLY in JSON format as {{"Relevance": ""}}
""".strip()

relevance_score_prompt_mistral = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 
Please make sure you read and understand these instructions carefully. 

Evaluation Criteria:
Relevance (1-5) - This dimension evaluates whether the counterspeech addresses the central theme or subject of the hate speech. 

Evaluation Steps:
Score 1 (Not Relevant): The counterspeech fails to address the central focus or subject matter of the hate speech and is mostly or entirely off-topic.
Score 2 (Slightly Relevant): The counterspeech only partially addresses the hate speech, with much content being unrelated or off-topic.
Score 3 (Moderately Relevant): The counterspeech addresses the hate speech but contains significant unrelated content or deviates from the main point.
Score 4 (Mostly Relevant): The counterspeech primarily addresses the central focus but includes minor unrelated elements.
Score 5 (Highly Relevant): The counterspeech directly addresses the central focus or subject matter of the hate speech.
Ensure that the response is STRICTLY in JSON format as {{"Relevance": ""}}
""".strip()


aggressiveness_score_prompt_gpt4 = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Aggressiveness (1-5) - This metric assesses the degree of confrontational or inflammatory content in the counterspeech. It considers factors such as abusive language, intensity of disagreement, tone, engagement style, and the presence of personal attacks, with higher scores indicating greater aggression.

Evaluation Steps:
Spot Key Aggression Elements: Identify components that indicate an aggressive tone. Look for Offensive Language, Degree of Disagreement, Tone and Style of Engagement, Character Attacks

Assign a Rating from 1-5: Based on the identified elements, rate the counterspeech on a scale from 1 to 5 according to these guidelines:
1 (Not Aggressive): The counterspeech is calm, respectful, devoid of abusive language or personal attacks, and disagrees constructively and politely.
2 (Slightly Aggressive): May contain mild language or an assertive tone, but remains largely respectful and avoids personal attacks.
3 (Moderately Aggressive): Exhibits a noticeable aggressive tone, may use stronger language, clearly shows strong disagreement, and might include mild personal criticisms.
4 (Highly Aggressive): Contains harsh language, strong personal attacks, or very intense disagreement. The tone is hostile and confrontational.
5 (Extremely Aggressive): Filled with abusive language, severe personal attacks, and displays an extremely confrontational and inflammatory tone.
Determine the Final Score: After thorough consideration, assign a final score from 1 to 5 based on the aggressiveness criteria.

Ensure that the response is STRICTLY in JSON format as {{"Aggressiveness": ""}}.
""".strip()

aggressiveness_score_prompt_llama3 = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Aggressiveness (1-5) - This metric assesses the degree of confrontational or inflammatory content in the counterspeech. It considers factors such as abusive language, intensity of disagreement, tone, engagement style, and the presence of personal attacks, with higher scores indicating greater aggression.

Evaluation Steps:
Assess Aggressiveness: Evaluate the counterspeech on a scale of 1 to 5, considering the following guidelines:
1 (Low Aggressiveness): The counterspeech is calm, respectful, and free of abusive language or personal attacks, with a constructive and polite tone.
2 (Mild Aggressiveness): The counterspeech may exhibit mild language or a somewhat assertive tone, but remains generally respectful and avoids personal attacks.
3 (Noticeable Aggressiveness): The counterspeech has a noticeable aggressive tone, may include stronger language, and shows a clear and strong disagreement, possibly with mild personal criticisms.
4 (High Aggressiveness): The counterspeech includes harsh language, strong personal attacks, or very intense disagreement, with a hostile and confrontational tone.
5 (Extreme Aggressiveness): The counterspeech is filled with abusive language, severe personal attacks, and exhibits an extremely confrontational and inflammatory tone.
Based on your careful evaluation, assign a score from 1 to 5, reflecting the level of aggressiveness in the counterspeech.

Ensure that the response is STRICTLY in JSON format as {{"Aggressiveness": ""}}.
""".strip()

aggressiveness_score_prompt_mistral = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Aggressiveness (1-5) - This metric assesses the degree of confrontational or inflammatory content in the counterspeech. It considers factors such as abusive language, intensity of disagreement, tone, engagement style, and the presence of personal attacks, with higher scores indicating greater aggression.

Evaluation Steps:
Assess the Aggressiveness (5-1): Evaluate the counterspeech on a scale of 1 to 5 based on the following:
5 (Extremely Confrontational): The counterspeech is filled with abusive language, severe personal attacks, and an extremely argumentative and inflammatory tone.
4 (Highly Confrontational): The counterspeech includes harsh language, direct personal criticisms, or very intense disagreement. The tone is aggressive and argumentative.
3 (Moderately Confrontational): The counterspeech displays a discernible confrontational tone, may use stronger language, and shows clear, strong disagreement, possibly with mild criticisms of the other party.
2 (Slightly Confrontational): The counterspeech may contain mild language or a somewhat assertive tone but remains courteous and avoids personal insults.
1 (Non-Confrontational): The counterspeech is composed, courteous, and devoid of abusive language or personal insults. It expresses dissent in a constructive and polite manner.

Ensure that the response is STRICTLY in JSON format as {{"Aggressiveness": ""}}.
""".strip()


coherence_score_prompt_gpt4 = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Argument Coherence (1-5) - Argument Coherence measures how logically the arguments within the counterspeech connect and support each other towards refuting the hate speech A coherent counterspeech will present its arguments in a clear and logically structured manner, making it easy for the reader to follow and understand the counter-narrative being presented.

Evaluation Steps:
Rate on a Scale of 1-5:
1 (Not Coherent): Disorganized and illogical, with arguments that don't follow a clear sequence.
2 (Slightly Coherent): Some logical connections, but lacks clarity and smooth transitions, making it hard to follow.
3 (Moderately Coherent): Fair coherence with logical flow and some supporting evidence, but may have minor gaps.
4 (Mostly Coherent): Clear and logical sequence with good transitions and consistency, though minor improvements are possible.
5 (Fully Coherent): Excellent coherence with a compelling, logical progression of arguments and thorough support, making it highly persuasive and easy to follow.
Finalize Your Score: Decide on a score reflecting the coherence of the counterspeech based on the aspects above.

Ensure that the response is STRICTLY in JSON format as {{"Argument Coherence": ""}}.
""".strip()


coherence_score_prompt_mistral = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Argument Coherence (1-5) - Argument Coherence measures how logically the arguments within the counterspeech connect and support each other towards refuting the hate speech A coherent counterspeech will present its arguments in a clear and logically structured manner, making it easy for the reader to follow and understand the counter-narrative being presented.

Evaluation Steps:
1 (Inconsistent): Disorganized and illogical, with arguments that don't follow a clear sequence.
2 (Slightly Consistent): Some logical connections, but lacks clarity and smooth transitions, making it hard to follow.
3 (Moderately Consistent): Fair consistency with logical flow and some supporting evidence, but may have minor gaps.
4 (Mostly Consistent): Clear and logical sequence with good transitions and consistency, though minor improvements are possible.
5 (Consistent): Excellent consistency with a compelling, logical progression of arguments and thorough support, making it highly persuasive and easy to follow.

Ensure that the response is STRICTLY in JSON format as {{"Argument Coherence": ""}}.
""".strip()

coherence_score_prompt_llama3 = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Argument Coherence (1-5) - Argument Coherence measures how logically the arguments within the counterspeech connect and support each other towards refuting the hate speech A coherent counterspeech will present its arguments in a clear and logically structured manner, making it easy for the reader to follow and understand the counter-narrative being presented.

Evaluation Steps:
1 (Inconsistent): Disorganized and illogical, with arguments that don't follow a clear sequence.
2 (Slightly Consistent): Some logical connections, but lacks clarity and smooth transitions, making it hard to follow.
3 (Moderately Consistent): Fair consistency with logical flow and some supporting evidence, but may have minor gaps.
4 (Mostly Consistent): Clear and logical sequence with good transitions and consistency, though minor improvements are possible.
5 (Consistent): Excellent consistency with a compelling, logical progression of arguments and thorough support, making it highly persuasive and easy to follow.

Ensure that the response is STRICTLY in JSON format as {{"Argument Coherence": ""}}.
""".strip()


suitableness_score_prompt_gpt4 = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Choose or Not  (1-3) -  This metric measures the likelihood of an annotator choosing a given counterspeech for direct use (without editing) in a real scenario against online hate speech. This assessment considers the counterspeech's suitability, appropriateness, and potential impact on a reader in a real-world context.

Evaluation Steps:
Assess for Direct Use in Real Scenarios:
Score 1 (Unlikely to Choose): The counterspeech may have good intentions but lacks effectiveness, appropriateness, or clarity for real-world use. It might escalate the situation, be off-topic, or fail to address the hate speech directly.
Score 2 (Possibly to Choose): The counterspeech has merit and could be used as is in real scenarios. It addresses the hate speech directly and is appropriate but might need minor revisions for clarity, tone, or effectiveness.
Score 3 (Likely to Choose): The counterspeech is well-crafted, directly addresses the hate speech, and is appropriate for real-world contexts. It is impactful, promotes positive dialogue, and counters the hate speech effectively without edits.
Evaluate Suitability and Appropriateness: Consider if the counterspeech is suitable for the intended audience and platform. Ensure it promotes constructive dialogue, and is respectful and empathetic.
Finalize Your Score: Decide on a score from 1 to 3 based on the likelihood of choosing the counterspeech for real-world use against hate speech, considering its suitability, appropriateness, and potential for positive impact.

Ensure that the response is STRICTLY in JSON format as {{"Choose or Not": ""}}.
""".strip()

suitableness_score_prompt_mistral = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Choose or Not  (1-3) -  This metric measures the likelihood of an annotator choosing a given counterspeech for direct use (without editing) in a real scenario against online hate speech. This assessment considers the counterspeech's suitability, appropriateness, and potential impact on a reader in a real-world context.

Evaluation Steps:
Evaluate Suitability and Appropriateness: Assess if the counterspeech is suitable for the target audience and platform. Ensure it fosters constructive dialogue, and is respectful and empathetic.
Scores 1 (Unlikely to Recommend): The counterspeech may have good intentions, but it may not be effective, appropriate, or clear enough for practical use. It could potentially worsen the situation, stray from the topic, or fail to directly challenge the hate speech.
Scores 2 (Potentially Recommendable): The counterspeech has potential and could be used in real-life scenarios with minor adjustments. It directly addresses the hate speech, is appropriate, but may require revisions for clarity, tone, or effectiveness.
Scores 3 (Recommendable): The counterspeech is well-crafted, directly challenges the hate speech, and is suitable for real-world contexts. It is impactful, encourages positive dialogue, and effectively counters the hate speech without needing any edits.

Finalize Your Score: Decide on a score from 1 to 3 based on the likelihood of choosing the counterspeech for real-world use against hate speech, considering its suitability, appropriateness, and potential for positive impact.

Ensure that the response is STRICTLY in JSON format as {{"Choose or Not": ""}}.
""".strip()

suitableness_score_prompt_llama3 = f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Choose or Not  (1-3) -  This metric measures the likelihood of an annotator choosing a given counterspeech for direct use (without editing) in a real scenario against online hate speech. This assessment considers the counterspeech's suitability, appropriateness, and potential impact on a reader in a real-world context.

Evaluation Steps:
Scores 1 (Not Likely): The counterspeech, while well-intentioned, may not be effective, suitable, or clear enough for practical applications. It could potentially worsen the situation, deviate from the topic, or fail to directly challenge the hate speech.
Scores 2 (Potentially): The counterspeech holds potential and could be used in real-life situations. It directly confronts the hate speech, is appropriate, but might require minor adjustments for improved clarity, tone, or impact.
Scores 3 (Likely): The counterspeech is well-structured, directly challenges the hate speech, and is suitable for real-world contexts. It is impactful, fosters positive discourse, and effectively counters the hate speech without needing modifications.

Ensure that the response is STRICTLY in JSON format as {{"Choose or Not": ""}}.
""".strip()

stance_prompt = """
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Stance (1-3) - This dimension evaluates how strongly the counterspeech opposes, contests, or contradicts the hate speech. It focuses on the expression of an opposing sentiment, regardless of the argument’s quality or relevance.

Evaluation Steps:
1. Read the Counterspeech: Carefully read through the counterspeech to capture the sentiment it expresses in response to the hate speech.
2. Understand the Scale of 1-3:
   - Score 1 (Weak Opposition): The counterspeech only weakly opposes or slightly contests the bias or stereotype of the hate speech. The sentiment of opposition is either vague or minimally expressed.
   - Score 2 (Moderate Opposition): The counterspeech clearly opposes or contests the bias or stereotype, but the sentiment could be stronger or more emphatic. It provides a noticeable counter-narrative but stops short of a full contradiction.
   - Score 3 (Strong Opposition): The counterspeech strongly and emphatically opposes, contests, or contradicts the bias or stereotype in the hate speech. It expresses a clear and powerful opposing sentiment that is unequivocal.

Ensure that the response is STRICTLY in JSON format as {{"Stance": ""}}
"""
