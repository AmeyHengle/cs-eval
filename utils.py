import json
import os
import random
import re
import string

import numpy as np
import pandas as pd
import torch
from cleantext import clean
from loguru import logger
import tagging_guidelines
from sklearn.model_selection import train_test_split

dashed_line = "- " * 50


def clean_text(text: str, min_len=4) -> str:
    """
    Returns cleaned text (post/title) by removing extra whitespaces, unprintable characters, recurring punctions, & urls

    Before:

    My ðŸ¤ [ndad] (https://np.reddit.com/r/raisedbynarcissists/comments/4puzqb/if_ndad_shows_up_to_my_graduation_im_going_to/) and I have been estranged for about 4-5 months now. Since the estrangement, I have noticed feeling this desire to tighten my abdominal muscles when I remember something my ndad did or didn't do that led me to feel so much pain as a result of feeling angry, hurt or depressed by all the hurt he has caused me.

    The ¤ only other time I think of tightening my abdominal muscles is during exercise because I remember when I used exercise videos, the instructor would instruct such...!

    I'm taking medications ([at the very least, I am trying](https://np.reddit.com/r/Anger/comments/4rjatz/new_doctor_wants_me_to_go_off_antidepressants/)) and am going to therapy so I hope I am handling this pretty well, just wanted to get second opinions.

    Should I be concerned about this.......? Do you have any idea why one would desire to tighten abdominal muscles in this situation or in situations such as these......!?
    Ÿ


    *********************************************************************************************************************************

    After:
    my [ndad] (<url>) and i have been estranged for about 4-5 months now. since the estrangement, i have noticed feeling this desire to tighten my abdominal muscles when i remember something my ndad did or didn't do that led me to feel so much pain as a result of feeling angry, hurt or depressed by all the hurt he has caused me. the  only other time i think of tightening my abdominal muscles is during exercise because i remember when i used exercise videos, the instructor would instruct such! i'm taking medications ([at the very least, i am trying](<url>)) and am going to therapy so i hope i am handling this pretty well, just wanted to get second opinions. should i be concerned about this? do you have any idea why one would desire to tighten abdominal muscles in this situation or in situations such as these?

    """

    punc = """()-[]{};:'"\<>/@#$%^&*_~"""

    if not isinstance(text, str):
        logger.debug(ValueError(f"encountered invalid format string: {text}"))
        return None

    text = re.sub(
        f"[^{re.escape(string.printable)}]",
        "",
        re.sub(
            r"[\?\.\!]+(?=[\?\.\!])",
            "",
            clean(
                text,
                fix_unicode=False,
                to_ascii=False,
                lower=False,
                no_line_breaks=True,
                no_urls=True,
                no_emails=True,
                no_punct=False,
                no_emoji=True,
                no_currency_symbols=True,
                lang="en",
                replace_with_url="",
                replace_with_email="",
            ),
        ),
    ).strip()

    for _ in punc:
        text = text.replace(_, "")
    text = text.replace(f". .", ". ").replace("  ", " ").replace('"', "")
    text = text.strip()

    if len(text) < min_len:
        return "<EMPTY_TEXT>"
    else:
        return text.strip()


def load_json(fpath: str):
    if not fpath.endswith(".json"):
        raise ValueError(f"{fpath} not a json file")

    with open(fpath, "r") as fp:
        return json.load(fp)


def save_json(data, fpath: str):
    if not fpath.endswith(".json"):
        raise ValueError(f"{fpath} not a json file")

    with open(fpath, "w") as fp:
        return json.dump(data, fp)


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    id_col: str,
    stratify_col: str,
    random_state: int,
):
    x_train, x_test, y_train, y_test = train_test_split(
        df[id_col].values.tolist(),
        df[stratify_col].values.tolist(),
        stratify=df[stratify_col].values.tolist(),
        test_size=test_size,
        random_state=random_state,
    )

    df_train = df[df[id_col].isin(x_train)]
    df_test = df[df[id_col].isin(x_test)]
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    logger.debug(
        f"\nTrain set: {df_train.shape}\n{df_train[stratify_col].value_counts()}"
    )
    logger.debug(f"\nTest Set: {df_test.shape}\n{df_test[stratify_col].value_counts()}")

    return df_train, df_test


import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def search_best_match(search_text: str, df: pd.DataFrame, column: str):
    # Load the pre-trained sentence transformer model
    model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

    # Compute the embeddings for the search text and the DataFrame column
    search_text_embedding = model.encode([search_text])
    column_embeddings = model.encode(df[column].tolist())

    # Compute the cosine similarity between the embeddings
    similarities = cosine_similarity(search_text_embedding, column_embeddings)

    # Find the index of the best match
    best_match_index = similarities.argmax()

    # Return the best match
    return best_match_index, df.iloc[best_match_index]


import glob
import pandas as pd
import json
import ast
import re
from utils import dashed_line
import string


def clean_str(input_string) -> str:
    input_string = input_string.strip()  # Remove extra line spaces
    input_string = input_string.replace("\n", " </s> ")  # Replace line spaces with /s<>
    input_string = input_string.replace("  ", " ")
    input_string = input_string.replace("\\", " ")
    input_string = input_string.replace("/", " ")
    input_string = input_string.lower()  # Lower all characters.
    return input_string


def postprocess_llm_pred(llm_prediction: str, key="") -> str:
    """
    Function to parse the right answer from noisy LLM prediction
    """
    answer = "<None>"
    if (
        not isinstance(llm_prediction, str)
        or len(llm_prediction) < 1
        or llm_prediction == "{}"
    ):  # Return default value in case prediction == None or empty.
        return answer
    else:
        if (
            "Question:" in llm_prediction
        ):  # For models like llama, output is both the prompt and prediction.
            llm_prediction = llm_prediction.split("Answer:")[-1]
        llm_prediction = clean_str(llm_prediction)

        if (
            "{" in llm_prediction and "}" in llm_prediction
        ):  # Check if any json / dict object is present within the prediction.
            dict_str = extract_json(llm_prediction)  # Extract dict object.
            if dict_str is not None:  # Succefully extracted dict from string.
                dict_obj = str_to_dict(dict_str)
                if (
                    isinstance(dict_obj, dict) and len(dict_obj) > 0
                ):  # Succesfully converted string to dict
                    try:
                        key = list(dict_obj.keys())[0]
                        key_ = remove_punctuation(key)
                        key_ = key_.strip()
                        key_ = key_.lower()
                        dict_obj[key_] = dict_obj[key]
                    except:
                        print(dict_obj)
                    if not "answer" in dict_obj:
                        print(
                            f"Encountered a prediction with dict object withou 'answer' key:\n{llm_prediction}"
                        )
                        return remove_punctuation(llm_prediction)

                    answer = dict_obj["answer"]
                    answer = remove_punctuation(answer)
                    return answer

        else:  # No json / dict object in prediction.
            answer = remove_punctuation(llm_prediction)
            return answer

    return answer


def extract_json(input_string):
    start_index = input_string.find("{")
    end_index = input_string.find("}")

    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return None
    else:
        dict_str = "{" + input_string[start_index + 1 : end_index] + "}"
        return dict_str


def str_to_dict(input_string):
    input_string = input_string.strip()
    if len(input_string.split(":")) == 2:  # dict object in string and not a set object
        input_string = input_string.split(":")
        key = input_string[0]
        key = key.lower().strip().replace('"', "").replace("'", "").replace("{", "")
        key = f'"{key}"'

        value = input_string[1]
        value = value.lower().strip().replace('"', "").replace("'", "").replace("}", "")
        value = f'"{value}"'

        input_string = "{" + key + ":" + value + "}"

    try:
        dict_obj = json.loads(input_string)
        if isinstance(dict_obj, set):
            dict_obj = {"answer": list(dict_obj)[0]}
        return dict_obj
    except:
        try:
            dict_obj = ast.literal_eval(input_string)
            if isinstance(dict_obj, set):
                dict_obj = {"answer": list(dict_obj)[0]}
            return dict_obj
        except:
            print(f"Could not convert the following string to dict:\n{input_string}")
            print(dashed_line)
            return None


def remove_punctuation(input_string, replace_by=""):
    # return re.sub(r'[^\w\s]', ' ', input_string)
    punctuations = string.punctuation
    punctuations.replace("<", "")
    punctuations.replace(">", "")
    punctuations.replace("'", "")
    return f"{replace_by}".join(
        char for char in input_string if char not in punctuations
    )


def get_val(d):
    """
    Returns the value from a dictionary containing a single key-value pair.

    Args:
    d (dict): A dictionary with a single key-value pair.

    Returns:
    The value of the single key-value pair in the dictionary.
    """
    assert isinstance(d, str)
    d = ast.literal_eval(d)
    return float(next(iter(d.values())))


def extract_score(input_string: str, key="Relevance") -> float:
    if not isinstance(input_string, str):
        print(f"Invalid input: {input_string}")
        return -1
    """
    try:
        input = ast.literal_eval(input)
        score = input[key]
        score = score.strip().replace(' ','')
        score = float(score)
        return score
    except:
        return -1
    """
    # Use regex to find the number in the string
    match = re.search(r"\d+", input_string)

    # If a number is found, convert it to float and return it
    if match:
        return float(match.group())
    else:
        return -1


def extract_content_from_response(response):
    try:
        return response.choices[0].message.content
    except:
        print(response)


def postprocess_llm_response(response):
    if isinstance(response, dict):
        return list(response.keys())[0]
    else:
        try:
            x = json.loads(response)
            return list(x.keys())[0]
        except:
            return str(response)


def get_criteria_example(aspect_name):
    if aspect_name == "Relevance":
        return tagging_guidelines.def_relevance
    elif aspect_name == "Coherence":
        return tagging_guidelines.def_coherence
    elif aspect_name == "Aggressiveness":
        return tagging_guidelines.def_aggressiveness
    elif aspect_name == "Suitableness":
        return tagging_guidelines.def_suitableness
