relevance_score_prompt = lambda hs, cs: f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 
Please make sure you read and understand these instructions carefully. 

Evaluation Criteria:
Relevance (1-5) - This dimension evaluates whether the counterspeech addresses the central theme or subject of the hate speech. 

Evaluation Steps:
1. **Read and Understand the Hate Speech**: Begin by reading the hate speech thoroughly. Understand the central theme, subject, or the specific point of hate it is addressing. This might involve recognizing the target group, the type of hateful message (e.g., stereotypes, derogatory remarks), and the context in which it is presented.

2. **Evaluate Relevance**:
   - **Score 5 (Highly Relevant)**: The counterspeech directly addresses the central theme or subject of the hate speech.
   - **Score 4 (Mostly Relevant)**: The counterspeech mainly addresses the central theme of the hate speech but may include minor elements or arguments that are not directly related.
   - **Score 3 (Moderately Relevant)**: The counterspeech addresses the hate speech but also includes significant unrelated content or deviates from the main point at times.
   - **Score 2 (Slightly Relevant)**: The counterspeech only partially addresses the hate speech, with a considerable portion of the content being unrelated or off-topic.
   - **Score 1 (Not Relevant)**: The counterspeech fails to address the central theme or subject of the hate speech. It is largely or entirely off-topic.

3. **Consider Contextual Factors**: Sometimes, the relevance of a counterspeech can be influenced by contextual factors such as cultural references, historical events, or social nuances. Ensure these factors are considered in your evaluation, especially if the hate speech or counterspeech involves specific references or subtleties.

4. **Finalize Your Score**: After careful consideration, assign a score from 1 to 5 (Likert Scale) based on the relevance criteria. 

Example:
Hate Speech: "{hs}"
Counterspeech: "{cs}"

Ensure that the response is STRICTLY in JSON format. 

Evaluation Form:
{{ 
'Relevance': ''
}}
"""

relevance_score_prompt_reasoning = lambda hs, cs: f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Relevance (1-5) - This dimension evaluates whether the counterspeech addresses the central theme or subject of the hate speech. 

Evaluation Steps:
1. **Read and Understand the Hate Speech**: Begin by reading the hate speech thoroughly. Understand the central theme, subject, or the specific point of hate it is addressing. This might involve recognizing the target group, the type of hateful message (e.g., stereotypes, derogatory remarks), and the context in which it is presented.

2. **Evaluate Relevance**:
   - **Score 5 (Highly Relevant)**: The counterspeech directly addresses the central theme or subject of the hate speech.
   - **Score 4 (Mostly Relevant)**: The counterspeech mainly addresses the central theme of the hate speech but may include minor elements or arguments that are not directly related.
   - **Score 3 (Moderately Relevant)**: The counterspeech addresses the hate speech but also includes significant unrelated content or deviates from the main point at times.
   - **Score 2 (Slightly Relevant)**: The counterspeech only partially addresses the hate speech, with a considerable portion of the content being unrelated or off-topic.
   - **Score 1 (Not Relevant)**: The counterspeech fails to address the central theme or subject of the hate speech. It is largely or entirely off-topic.

3. **Consider Contextual Factors**: Sometimes, the relevance of a counterspeech can be influenced by contextual factors such as cultural references, historical events, or social nuances. Ensure these factors are considered in your evaluation, especially if the hate speech or counterspeech involves specific references or subtleties.

4. **Finalize Your Score**: After careful consideration, assign a score from 1 to 5 (Likert Scale) based on the relevance criteria. 

Example:
Hate Speech: "{hs}"
Counterspeech: "{cs}"

Ensure that the response is STRICTLY in JSON format. 

Evaluation Form:
{{ 
Relevance: '',
Reasoning: ''
}}
"""

aggressiveness_score_prompt = lambda hs, cs: f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Aggressiveness (1-5) - This metric assesses the degree of confrontational or inflammatory content in the counterspeech. It considers factors such as abusive language, intensity of disagreement, tone, engagement style, and the presence of personal attacks, with higher scores indicating greater aggression.

Evaluation Steps:
1. **Read the Counterspeech Carefully**: Begin by thoroughly reading the counterspeech. Understand the context and the message it intends to convey in response to the hate speech.

2. **Identify Key Elements of Aggression**: Look for elements that contribute to an aggressive tone. This includes:
   - **Abusive Language**: Check for the presence of swear words, slurs, or derogatory terms.
   - **Intensity of Disagreement**: Assess the level of opposition in the counterspeech. A polite or calm disagreement generally indicates a lower level of aggression, while a vehement or harsh opposition suggests higher aggression.
   - **Tone and Engagement Style**: Consider the overall tone. Is it calm and reasoned, or does it come across as angry and hostile? Also, note if the counterspeech engages in a constructive manner or if it is dismissive and belittling.
   - **Personal Attacks**: Look for any direct attacks on the character or qualities of the individual who made the initial hate speech. Personal attacks significantly increase the aggression score.

3. **Rate on a Scale of 1-5**: After identifying these elements, rate the counterspeech on a scale from 1 to 5 based on the following guidelines:
   - **1 (Not Aggressive)**: The counterspeech is calm, respectful, and free of any abusive language or personal attacks. It disagrees in a constructive and polite manner.
   - **2 (Slightly Aggressive)**: There might be mild language or a somewhat assertive tone, but it mostly remains respectful and avoids personal attacks.
   - **3 (Moderately Aggressive)**: The counterspeech has a noticeable aggressive tone, may include stronger language, and shows a clear and strong disagreement, possibly with mild personal criticisms.
   - **4 (Highly Aggressive)**: The response includes harsh language, strong personal attacks, or very intense disagreement. The tone is hostile and confrontational.
   - **5 (Extremely Aggressive)**: The counterspeech is filled with abusive language, severe personal attacks, and exhibits an extremely confrontational and inflammatory tone.

4. **Finalize Your Score**: After careful consideration, assign a score from 1 to 5 based on the aggressiveness criteria.

Example:
Hate Speech: "{hs}"
Counterspeech: "{cs}"

Ensure that the response is STRICTLY in JSON format. 

Evaluation Form:
{{ 
'Aggressiveness': ''
}}
"""

coherence_score_prompt = lambda hs, cs: f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Coherence (1-5) - This metric assesses how logically and smoothly the ideas or arguments within the counterspeech connect and flow. A coherent counterspeech will present its arguments in an organized manner, making it easy for the reader to follow and understand the counter-narrative being presented.

Evaluation Steps:
1. **Read the Counterspeech**: Carefully go through the counterspeech. Pay attention to how the ideas or arguments are structured and whether the transitions between them are smooth and logical.
2. **Identify Key Aspects of Coherence**:
    - **Logical Flow**: Assess whether the counterspeech presents its arguments or points in a logical sequence that builds upon each previous point.
    - **Clarity of Ideas**: Determine if the ideas are clearly expressed and if the counterspeech avoids ambiguity and confusion.
    - **Transitions**: Look for transitional phrases or sentences that help connect different parts of the counterspeech, enhancing its overall coherence.
    - **Consistency**: Check for consistency in the arguments, tone, and perspective throughout the counterspeech.
3. **Rate on a Scale of 1-5**:
    - **1 (Not Coherent)**: The counterspeech is disjointed, with ideas or arguments presented in a confusing or illogical order. It lacks clear transitions and consistency, making it difficult to follow.
    - **2 (Slightly Coherent)**: While some parts of the counterspeech may be logically connected, overall, it struggles with clarity, transitions, and logical flow, making it somewhat difficult to follow.
    - **3 (Moderately Coherent)**: The counterspeech has a reasonable structure and flow, with some effective transitions and mostly clear ideas, though there may be occasional lapses in logic or clarity.
    - **4 (Mostly Coherent)**: The counterspeech presents its arguments in a clear and logical sequence with good transitions and consistency, making it easy to follow, though minor improvements could enhance its coherence further.
    - **5 (Fully Coherent)**: The counterspeech is exceptionally well-structured, with arguments presented in a clear, logical, and consistent manner. Transitions are smooth, enhancing the overall flow and making it very easy to follow.
4. **Finalize Your Score**: After analyzing the counterspeech against these coherence criteria, decide on a score between 1 to 5 that best represents its level of coherence.

Example:
Hate Speech: "{hs}"
Counterspeech: "{cs}"

Ensure that the response is STRICTLY in JSON format. 

Evaluation Form:
{{ 
'Coherence': ''
}}
"""

suitableness_score_prompt = lambda hs, cs: f"""
You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a hateful statement (hate speech). 
Your task is to rate the counterspeech on one metric. 

Evaluation Criteria:
Choose or not  (1-3) -  This metric measures the likelihood of an annotator choosing a given counterspeech for direct use (without editing) in a real scenario against online hate speech. This assessment considers the counterspeech's suitability, appropriateness, and potential impact on a reader in a real-world context.

Evaluation Steps:
1. **Read the Counterspeech**: Carefully review the counterspeech provided. Remember, this evaluation is about determining the practical applicability of the counterspeech in real-life situations. It's crucial to consider the counterspeech's overall message, tone, and how it addresses the hate speech to decide on its direct usability without modifications.
2. **Assess for Direct Use in Real Scenarios**:
   - **Score 1 (Unlikely to Choose)**: The counterspeech, while potentially having good intentions, lacks the effectiveness, appropriateness, or clarity to be used as is in a real-world scenario. It may escalate the situation, be off-topic, or fail to directly address the hate speech.
   - **Score 2 (Possibly to Choose)**: The counterspeech has merit and could potentially be used as is in real scenarios. It addresses the hate speech directly and is appropriate for the context. However, it might benefit from minor revisions for clarity, tone, or effectiveness.
   - **Score 3 (Likely to Choose)**: The counterspeech is well-crafted, directly addresses the hate speech, and is appropriate for a real-world context. It is impactful, promotes positive dialogue, and counters the hate speech effectively without needing any edits.

3. **Evaluate Suitability and Appropriateness**: Consider if the counterspeech is suitable for the audience and platform where it might be used. Think about whether it promotes constructive dialogue and if it's respectful and empathetic.

4. **Finalize Your Score**: Decide on a score from 1 to 3, based on how likely you would be to choose the counterspeech for use in a real scenario against hate speech, considering its suitability, appropriateness, and potential for positive impact.

Example:
Hate Speech: "{hs}"
Counterspeech: "{cs}"

Ensure that the response is STRICTLY in JSON format. 

Evaluation Form:
{{ 
'Choose or not': ''
}}
"""