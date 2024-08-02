## Aim: Illustrations for Readme

# Score a given Counter Speech based on the following dimensions - **Contextual Relevance, Independent Counterspeech, Aggressiveness,** and **Coherence.**

# The Annotators are provided the appropriate context, knowledge-sources, and information regarding commonly encountered topics and Target Groups for reference.

## Example Hate Speech and Counter Speech used for illustration:

# **Hate Speech:** Maybe the UN could talk to those asian and african nations responsible for 90%+ of the pollution in the oceans' instead of insisting on this bullshit about climate change.

# **Intent:** Informative

# **Counter Speech:**

# The US is the second most polluting country in the world - as the world’s biggest industrial and commercial power. We are all to blame here and need to work hand in hand to create sustainable change, as opposed to pointing the finger to others.

# General Definition of Evaluation Dimensions
relevance = """Relevance assesses whether the counter
speech is in line with the hate speech’s central
theme, subject, or topic. Contextual relevance is
an important counterspeech quality, especially considering 
the implied nature of hate speech that can confuse language models."""

coherence = """Coherence measures whether a counter
speech provides specific and coherent arguments
to effectively refute or counter any bias, stereotype
or prejudice expressed in the hate speech. A high
score indicates that the counterspeech provides arguments that are consistent, 
evidence-based, and follow a clear logical flow and use."""

aggressiveness = """Aggressiveness evaluates the level of confrontational 
or inflammatory content in the counterspeech, including the use of any 
abusive language, the intensity of disagreement, the tone, and
whether it contains personal attacks. A lower score indicates less aggressive 
and hence more effective counterspeech."""

suitableness = """
Suitableness measures whether a counter speech can be directly used without editing in a 
real setting. It considers a counterspeech’s overall stance and potential impact on the listener.
"""

### Algorthim overview
"""
- **Candidate Generation**: Given an evaluation criteria and a small validation set of around 500 human labels, where inputs have been rated by humans on the given criteria, we leverage a Large Language Model (LLM) to generate candidate chain-of-thought (CoT) evaluation steps. These steps are generated using few-shot exemplars sampled from the validation set. We employ a prompt template to guide the generation of these candidate CoT evaluation steps.
- **Initial Scoring**: Utilizing the initial pool of candidate CoT prompts, we score the inputs from the validation set for the given criteria. This generates a set of scores, Sij, for each input i based on each candidate CoT prompt j.
- **Correlation Calculation**: We calculate the Spearman correlation, ρj, between the generated scores and human ratings for each of the candidate CoT prompts j. This provides a measure of alignment between each candidate CoT prompt and human judgment.
- **Top-Performing CoT Prompts Selection**: We filter out the top-performing CoT prompts based on the correlation score. This step narrows down the pool of candidate CoT prompts to those that best align with human judgment.
- **Iterative Refinement**: We repeat this process iteratively for n iterations, where n=10 in our case. Each iteration refines the pool of candidate CoT prompts and improves their alignment with human judgment.
- **Final CoT Prompt Selection**: After n iterations, we select the best performing CoT prompt (with the highest correlation score) as our final CoT prompt. This final CoT prompt is the one that best aligns with human judgment and will be used for evaluating new inputs.
"""

### Evaluation Guideline with examples for annotators

def_relevance = """
This dimension evaluates whether the counterspeech addresses the central theme or subject of the hate speech.

Annotation Steps:

1. **Read and Understand the Hate Speech**: Begin by reading the hate speech thoroughly. Understand the central theme, subject, or the specific point of hate it is addressing. This might involve recognizing the target group, the type of hateful message (e.g., stereotypes, derogatory remarks), and the context in which it is presented.
2. **Analyze the Counterspeech**: Read the counterspeech carefully. Your focus should be on how it addresses the hate speech. Consider whether the counterspeech directly responds to the main points of the hate speech or if it deviates to unrelated topics.
3. **Evaluate Relevance**:
    - **Score 5 (Highly Relevant)**: The counterspeech directly addresses the central theme or subject of the hate speech. It effectively counters the hateful message with relevant arguments, facts, or perspectives.
    - **Score 4 (Mostly Relevant)**: The counterspeech mainly addresses the central theme of the hate speech but may include minor elements or arguments that are not directly related.
    - **Score 3 (Moderately Relevant)**: The counterspeech addresses the hate speech but also includes significant unrelated content or deviates from the main point at times.
    - **Score 2 (Slightly Relevant)**: The counterspeech only partially addresses the hate speech, with a considerable portion of the content being unrelated or off-topic.
    - **Score 1 (Not Relevant)**: The counterspeech fails to address the central theme or subject of the hate speech. It is largely or entirely off-topic.
4. **Consider Contextual Factors**: Sometimes, the relevance of a counterspeech can be influenced by contextual factors such as cultural references, historical events, or social nuances. Ensure these factors are considered in your evaluation, especially if the hate speech or counterspeech involves specific references or subtleties.
5. **Finalize Your Score**: After careful consideration, assign a score from 1 to 5 based on the relevance criteria. Remember that relevance is about how directly and effectively the counterspeech confronts the hate speech.
"""

examples_relevance = """
Hate speech "Jews are extremely cheap, and they want to control the world" 
Here are example counterspeeches rated from 1 to 5 for Hate Speech:

### Score 1 (Not Relevant)

**Counterspeech**: "Everyone should strive to be environmentally friendly and reduce their carbon footprint."

- **Explanation**: This counterspeech fails to address the hate speech's theme or subject, focusing instead on an entirely unrelated topic (environmentalism), making it contextually irrelevant.

### Score 2 (Slightly Relevant)

**Counterspeech**: "It's not right to label any group based on financial stereotypes. We should all aim to be more financially responsible."

- **Explanation**: While this counterspeech touches on the theme of financial stereotypes, it broadly addresses financial responsibility without directly engaging with the specific stereotypes about Jews mentioned in the hate speech, making it slightly relevant.

### Score 3 (Moderately Relevant)

**Counterspeech**: "Stereotyping any group as 'cheap' is unfair. Many communities, including Jews, have members across the economic spectrum."

- **Explanation**: This counterspeech addresses the stereotype of being "cheap" and includes Jews as an example. However, it does not fully engage with the broader accusation of wanting to control the world, making it moderately relevant.

### Score 4 (Mostly Relevant)

**Counterspeech**: "The stereotype that Jews are 'cheap' or want to 'control the world' is not only untrue but also harmful. These myths have been debunked many times."

- **Explanation**: This counterspeech directly addresses the specific stereotypes mentioned in the hate speech, providing a clear stance against them but could enhance its relevance by offering more specific examples or evidence to debunk the myths.

### Score 5 (Fully Relevant)

**Counterspeech**: "The notion that Jews are 'cheap' and aspire to 'control tcouhe world' is a dangerous stereotype that fuels antisemitism. Historical evidence and economic studies show that Jews, like any other group, contribute positively to societies worldwide without any hidden agendas."

- **Explanation**: This counterspeech is fully relevant as it directly tackles the stereotypes mentioned, counters them with historical evidence and economic studies, and highlights the harm caused by such stereotypes, making it completely pertinent to the hate speech's context.

These examples illustrate varying levels of contextual relevance in counterspeech, from entirely off-topic responses to those that directly and effectively counter the specific themes and accusations in the hate speech.
"""

def_aggressiveness = """
This metric assesses the degree of confrontational or inflammatory content in the counterspeech. It considers factors such as abusive language, intensity of disagreement, tone, engagement style, and the presence of personal attacks, with higher scores indicating greater aggression.

Annotation Steps:

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
"""

examples_aggressiveness = """
Hate speech "Jews are extremely cheap, and they want to control the world" 
Here are example counterspeeches rated from 1 to 5 for Aggresiveness score:

### Score 1 (Not Aggressive)

**Counterspeech**: "Stereotypes like these are not only inaccurate but also harmful. Let's focus on understanding and celebrating the diverse contributions of Jewish communities around the world."

- **Explanation**: This counterspeech calmly addresses the stereotype with a constructive message promoting understanding and appreciation, without any aggressive language or personal attacks.

### Score 2 (Slightly Aggressive)

**Counterspeech**: "It's disappointing to hear such outdated stereotypes. Jews, like any other group, have individuals with varied characteristics and have made positive contributions throughout history."

- **Explanation**: The counterspeech expresses disagreement and addresses the stereotype directly but remains respectful and focuses on counteracting the negative message with positive information.

### Score 3 (Moderately Aggressive)

**Counterspeech**: "Blaming an entire group for the actions of a few is unfair and ignorant. Jews have been unfairly stereotyped for centuries, and it's high time we moved beyond these baseless accusations."

- **Explanation**: This response shows a stronger level of disagreement with the hate speech and calls out the ignorance behind stereotypes, using firmer language but without crossing into personal attacks or inflammatory rhetoric.

### Score 4 (Highly Aggressive)

**Counterspeech**: "Only someone completely ignorant of history would spout such nonsense about Jews. These stereotypes are as ridiculous as they are hateful."

- **Explanation**: The counterspeech includes harsher criticism of the person spreading the stereotype, bordering on personal attacks by questioning their knowledge and understanding, making it more confrontational.

### Score 5 (Extremely Aggressive)

**Counterspeech**: "This kind of ignorant hate speech is exactly what's wrong with the world. People who spread lies about Jews wanting to control the world are not only bigoted but also dangerously stupid."

- **Explanation**: This counterspeech is filled with strong language that attacks the character of the person making the original hate speech ("dangerously stupid"), making it very aggressive and confrontational.

These examples illustrate a range of aggression levels in counterspeech, from calm and constructive responses to those that confront the hate speech with increasing levels of intensity, language strength, and directness of disagreement.
"""

def_coherence = """
Argument Coherence measures how logically the arguments within the counterspeech connect and support each other towards refuting the hate speech A coherent counterspeech will present its arguments in a clear and logically structured manner, making it easy for the reader to follow and understand the counter-narrative being presented.

Annotation Steps:

1. **Read the Counterspeech**: Carefully go through the counterspeech. Pay attention to how the ideas or arguments are structured and whether the transitions between them are smooth and logical.
2. **Identify Key Aspects of Coherence**:
    - **Logical Flow**: Assess whether the counterspeech presents its arguments or points in a logical sequence that builds upon each previous point.
    - **Clarity of Ideas**: Determine if the ideas are clearly expressed and if the counterspeech avoids ambiguity and confusion.
    - **Transitions**: Look for transitional phrases or sentences that help connect different parts of the counterspeech, enhancing its overall coherence.
    - **Consistency**: Check for consistency in the arguments, tone, and perspective throughout the counterspeech.
    - **Counter-Argument Integration**: Consider how well the counterspeech integrates counter-arguments or addresses potential objections within its narrative.
3. **Rate on a Scale of 1-5**:
    - **1 (Not Coherent)**: The counterspeech is disorganized or illogical, with arguments that do not follow a clear sequence or support the overall goal of countering the hate speech.
    - **2 (Slightly Coherent)**: While some parts of the counterspeech may be logically connected, overall, it struggles with clarity, transitions, and logical flow, making it somewhat difficult to follow.
    - **3 (Moderately Coherent)**: The counterspeech has a fair level of coherence, with a logical flow of arguments and some supporting evidence, though it may have minor gaps in logic or clarity.
    - **4 (Mostly Coherent)**: The counterspeech presents its arguments in a clear and logical sequence with good transitions and consistency, making it easy to follow, though minor improvements could enhance its coherence further.
    - **5 (Fully Coherent)**: The counterspeech demonstrates excellent coherence, with a compelling, logical progression of arguments, thorough support, and effective integration of counter-arguments, making it highly persuasive and easy to follow.
4. **Finalize Your Score**: After considering the aspects mentioned above, decide on a score that reflects the coherence of the argument presented in the counterspeech.

Remember, the focus here is on the logical structure, clarity, and persuasiveness of the counterspeech in countering the hate speech effectively.
"""

examples_coherence = """
Hate speech "Jews are extremely cheap, and they want to control the world" 
Below are examples of counterspeech addressing the hate speech with Argument Coherence scores from 1 to 5:

### Score 1 (Not Coherent)

**Counterspeech**: "Cheap? World? Control? Look at the sky, it’s blue! And remember, not all birds can fly."

- **Explanation**: This counterspeech is disjointed and illogical, failing to address the hate speech directly. It jumps from the topic abruptly to unrelated observations, lacking a coherent argument against the hate speech.

### Score 2 (Slightly Coherent)

**Counterspeech**: "Being cheap is a stereotype. But what about the sky? Some people like blue, but others prefer green. Not all want to control anything."

- **Explanation**: While attempting to refute the stereotype, this counterspeech quickly diverges into irrelevant topics, such as color preferences, without forming a clear or logical argument against the specific accusations made in the hate speech.

### Score 3 (Moderately Coherent)

**Counterspeech**: "Stereotyping Jews as cheap and power-hungry is unfair. Many communities have wealthy and influential individuals. It's not just about one group."

- **Explanation**: This counterspeech starts to form a logical argument against stereotyping but fails to fully develop a coherent or persuasive refutation of the specific claims about Jews. It attempts to generalize the issue but lacks direct evidence or examples to effectively counter the hate speech.

### Score 4 (Mostly Coherent)

**Counterspeech**: "The idea that Jews are cheap and want to control the world is a harmful stereotype. Throughout history, Jews have contributed significantly to society in various fields without any hidden agenda of world domination."

- **Explanation**: This counterspeech presents a logical argument by identifying the stereotype and providing a counter-argument focused on historical contributions of Jews. It begins to offer a coherent response but could further strengthen its argument with more specific examples or evidence.

### Score 5 (Fully Coherent)

**Counterspeech**: "Labeling Jews as 'cheap' or 'world-controlling' is rooted in historical prejudices that have no place in modern society. Economic behavior is not tied to any one ethnicity, and the conspiracy theory of world control has been debunked numerous times. Jews, like any other group, include diverse individuals with varying beliefs and aspirations. Highlighting their positive contributions across science, arts, and social justice offers a more accurate and respectful understanding."

- **Explanation**: This counterspeech effectively counters the hate speech with a clear, logical progression of arguments. It directly addresses and refutes each claim made in the hate speech with reasoned arguments and invites a broader, more positive perspective on the Jewish community.

These examples illustrate how the coherence of a counterspeech can range from disjointed and irrelevant responses to logically structured and persuasive arguments directly countering the hate speech.
"""

def_suitableness = """
Suitableness (also known as Choose or Not) determines whether the annotators would select the given counter-narrative (without editing) and use it in a real .

1. **Understand the Criterion**: "Choose-or-not" (1-3) measures the likelihood of an annotator choosing a given counterspeech for direct use (without editing) in a real scenario against online hate speech. This assessment considers the counterspeech's suitability, appropriateness, and potential impact in a real-world context.
2. **Read the Counterspeech**: Carefully review the counterspeech provided. Pay attention to its message, tone, and how it addresses the hate speech. Consider the counterspeech's potential to effectively counteract the hate speech in a constructive and impactful manner.
3. **Evaluate for Real-world Use**:
    - **Score 1 (Unlikely to Choose)**: The counterspeech is seen as ineffective, inappropriate, or potentially escalatory. It lacks the qualities that would make it a suitable candidate for post-editing and use in addressing hate speech.
    - **Score 2 (Possibly to Choose)**: The counterspeech has merit and could be effective with some revisions. It contains elements that are constructive and appropriate but needs refinement to fully meet the requirements for real-world application.
    - **Score 3 (Likely to Choose)**: The counterspeech is effective, appropriate, and impactful as is. It stands as a strong candidate for post-editing and use in real scenarios, requiring minimal to no adjustments.
4. **Consider Impact and Appropriateness**: Reflect on the counterspeech's potential impact on the audience and its appropriateness for the platform or context where it might be used. Consider whether it promotes constructive dialogue and whether it could be perceived as respectful and empathetic.
5. **Finalize Your Score**: After considering the above factors, assign a score from 1 to 3 based on how likely you would be to choose the counterspeech for post-editing and use in a real case scenario against hate speech.
"""

examples_suitableness = """
Hate speech "Jews are extremely cheap, and they want to control the world" 
Here are example counterspeeches ranked from 1 to 3 for their likelihood of being chosen for direct use in a real scenario:

### Score 1 (Unlikely to Choose)

**Counterspeech**: "People who say things like this are just jealous and can't see the truth."

- **Reason**: This counterspeech indirectly addresses the hate speech but does so in a way that could perpetuate further hostility without providing a clear, constructive counter-narrative. It lacks specificity and may not effectively challenge the stereotypes or change anyone's opinion.

### Score 2 (Possibly to Choose)

**Counterspeech**: "It's unfair to label any group based on stereotypes. While it's true that some individuals may fit certain characteristics, it's important to judge people as individuals, not as monoliths."

- **Reason**: This counterspeech addresses the problem of stereotyping and promotes individual judgment, which is a step in the right direction. However, it's somewhat generic and may not fully engage with the specific stereotypes about Jews mentioned in the hate speech, requiring some edits to increase its impact.

### Score 3 (Likely to Choose)

**Counterspeech**: "The idea that Jews are cheap or want to control the world is a harmful stereotype that has no basis in reality. Throughout history, Jews have contributed to society in various ways, just like any other group. Spreading such myths only serves to divide us and ignores the rich diversity and accomplishments of Jewish communities around the world."

- **Reason**: This counterspeech directly challenges the stereotypes mentioned in the hate speech with clear arguments and promotes unity and recognition of Jewish contributions. It's informative, respectful, and engages constructively with the topic, making it suitable for direct use in countering hate speech online.

These examples illustrate different levels of suitability for counterspeech, considering its relevance, appropriateness, and potential impact in countering the specific stereotypes mentioned in the hate speech.
"""
