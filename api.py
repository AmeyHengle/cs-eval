import asyncio
import logging

import openai
import pandas as pd
from openai import error
from tqdm.asyncio import tqdm_asyncio
from typing import Any

import aiolimiter
import openai
from litellm import completion, acompletion
from aiohttp import ClientSession
from openai import error
from tqdm.asyncio import tqdm_asyncio

# from zeno_build.models import lm_config
# from zeno_build.prompts import chat_prompt

import os

ERROR_ERRORS_TO_MESSAGES = {
    error.InvalidRequestError: "OpenAI API Invalid Request: Prompt was filtered",
    error.RateLimitError: "OpenAI API rate limit exceeded. Sleeping for 10 seconds.",
    error.APIConnectionError: "OpenAI API Connection Error: Error Communicating with OpenAI",  # noqa E501
    error.Timeout: "OpenAI APITimeout Error: OpenAI Timeout",
    error.ServiceUnavailableError: "OpenAI service unavailable error: {e}",
    error.APIError: "OpenAI API error: {e}",
}


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        for _ in range(3):
            try:
                return await acompxletion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:
                if isinstance(e, (error.ServiceUnavailableError, error.APIError)):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                elif isinstance(e, error.InvalidRequestError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Invalid Request: Prompt was filtered"
                                }
                            }
                        ]
                    }
                else:
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    messages_list: list,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int = 150,
):
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of responses to generate for each API call.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model,
            messages=full_context,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for full_context in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    return [x["choices"][0]["message"]["content"] for x in responses]


if __name__ == "__main__":
    
    # input_filename = "/home/ameyh/counterspeech-EVAL/data/results_data_pass2_aggressiveness.csv"
    # output_filename = "/home/ameyh/counterspeech-EVAL/data/results_data_pass2_aggressiveness_nan-rerun.csv"
    # prompt_col = 'prompt_aggressiveness_score'
    input_filename = "//home/ameyh/counterspeech-EVAL/preds/results_combined.csv"
    output_filename = "/home/ameyh/counterspeech-EVAL/data/results_combined_suitableness_nonHuman.csv"
    prompt_col = 'prompt_suitableness_score'
    model_name = 'gpt-3.5-turbo'

    prompt_data = pd.read_csv(input_filename)
    prompt_data = prompt_data[prompt_data.source != 'Human']
    print(prompt_data.shape)
    # prompt_data = prompt_data[prompt_data['prediction_(prompt_aggressiveness_score)_(gpt-4)'].isna()]
    prompt_data['system_description'] = ''

    # prompt_data = prompt_data[prompt_data.Intent == 'Denouncing']
    
    input = []
    for system_description, prompt in zip(
        prompt_data["system_description"].values.tolist(),
        prompt_data[prompt_col].values.tolist(),
    ):
        input.append(
            [
                {"role": "system", "content": system_description},
                {"role": "user", "content": prompt},
            ]
        )

    print(f"\nSample Input: {input[0]}")

    print("\n\nGenerating responses for Prompts:\n")
    predictions = asyncio.run(
        generate_from_openai_chat_completion(
            messages_list=input,
            model=model_name,
            temperature=1,
            max_tokens=64,
            top_p=1,
            requests_per_minute=150
        )
    )

    prompt_data[f"prediction_({prompt_col})_({model_name})"] = predictions
    prompt_data.to_csv(output_filename, index=False)