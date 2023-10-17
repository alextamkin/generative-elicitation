from tenacity import (
    retry,
    wait_random_exponential,
)
import openai
import json
import os
import asyncio

import numpy as np
from scipy.interpolate import interp1d


"""
OPENAI FUNCTIONS
"""

def save_openai_cache(new_entry, openai_cache=None, openai_cache_file=None):
    '''Saves the new entry to the openai cache file and updates the openai_cache dict.
    
    Args:
        new_entry (dict): The new entry to save to the cache.
        openai_cache (dict): The openai cache dict to update.
        openai_cache_file (str): The path to the openai cache file.
    
    Returns:
        None
    '''
    if openai_cache_file:
        with open(openai_cache_file, "a") as wf:
            wf.write(json.dumps(new_entry)+"\n")
        openai_cache.update(new_entry)


def async_query_api(
    message_history_list,
    engine: str,
    openai_cache=None,
    openai_cache_file=None,
    **kwargs,
):
    return asyncio.run(dispatch_openai_requests(message_history_list, engine, openai_cache, openai_cache_file, **kwargs))

async def dispatch_openai_requests(
    message_history_list,
    engine: str,
    openai_cache=None,
    openai_cache_file=None,
    **kwargs,
):
    """Dispatches requests to OpenAI API asynchronously.

    Adapted from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a
    Uses a cache to retrieve cached responses
    
    Args:
        messages_history_list: List of message histories to be sent to async OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        messages_to_responses: dict of {query: response} pairs from OpenAI API.
    """
    noncached_message_list = []
    messages_to_responses = {}
    
    # Add cache hits to messages_to_responses dict
    for message_history in message_history_list:
        messages_cache_key = json.dumps(message_history)
        if openai_cache and messages_cache_key in openai_cache:
            response = openai_cache[messages_cache_key]
            messages_to_responses[messages_cache_key] = response
        else:
            noncached_message_list.append(message_history)
    
    # Do async API call for remaining uncached queries
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0
    if engine == "gpt-4" or engine == "gpt-3.5-turbo":
        async_responses = [
            openai.ChatCompletion.acreate(
                model=engine,
                messages=message_history,
                **kwargs,
            )
            for message_history in noncached_message_list
        ]
    else:
        async_responses = [
            openai.Completion.acreate(
                engine=engine,
                prompt=message_history[0],
                **kwargs,
            )
            for message_history in noncached_message_list
        ]
    responses = await asyncio.gather(*async_responses)
    
    # Add new {query: response} pairs to cache
    for i, response in enumerate(responses):
        messages_cache_key = json.dumps(noncached_message_list[i])
        messages_to_responses[messages_cache_key] = response
        save_openai_cache({messages_cache_key: response}, openai_cache, openai_cache_file)

    message_history_to_response_text = {}
    # Parse responses
    for message_history in message_history_list:
        # for message_cache_key in messages_to_responses:
        message_history_key = json.dumps(message_history)
        response = messages_to_responses[message_history_key]
        if engine == "gpt-4" or engine == "gpt-3.5-turbo":
            response_text = response['choices'][0]['message']['content']
            message_history_to_response_text[message_history_key] = response_text
            # message_history.append({'role': 'assistant', 'content': response_text})
        else:
            message_history_to_response_text[message_history_key] = response['choices'][0]['text']
    return message_history_to_response_text, messages_to_responses


@retry(wait=wait_random_exponential(min=1, max=60))
def query_api(messages, engine, openai_cache=None, openai_cache_file=None, **kwargs):
    '''Queries the OpenAI API with the given messages.
    
    NOTE: This function mutates the messages list to add the new_message and the response from the API.
    
    Args:
        messages (list): A list of past messages to send to the API.
        openai_cache (dict, optional): The openai cache dict. Stores the API responses to avoid duplicate queries. Defaults to None.
        openai_cache_file (str, optional): The path to write the cache entries to. Defaults to None.
    
    Returns:
        str: The response from the API.
    '''
    messages_cache_key = json.dumps(messages)
    if openai_cache and messages_cache_key in openai_cache:
        response = openai_cache[messages_cache_key]
    else:
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.0
        if engine == "gpt-4" or engine == "gpt-3.5-turbo":
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                **kwargs
            )
        else:
            response = openai.Completion.create(
                engine=engine,
                prompt=messages[0],
                **kwargs
            )
        save_openai_cache({messages_cache_key: response}, openai_cache, openai_cache_file)
    if engine == "gpt-4" or engine == "gpt-3.5-turbo":
        response_text = response['choices'][0]['message']['content']
        messages.append({'role': 'assistant', 'content': response_text})
    else:
        response_text = response['choices'][0]['text']
    return response_text, response



def load_openai_cache(openai_cache_file):
    '''Loads the openai cache file into a dict.
    
    Args:
        openai_cache_file (str): The path to the openai cache file.
        
    Returns:
        dict: The openai cache dict.
    '''
    if not openai_cache_file:
        return None
    openai_cache = {}
    if os.path.exists(openai_cache_file):
        with open(openai_cache_file) as f:
            for line in f:
                openai_cache.update(json.loads(line))
    return openai_cache


"""
METRICS
"""

def update_metrics(metrics, new_metrics):
    if len(metrics) == 0:
        return {
            metric: [new_metrics[metric]] for metric in new_metrics
        }
    for metric in metrics:
        metrics[metric].append(new_metrics[metric])
    return metrics


def update_test_responses(all_test_responses, new_test_responses):
    if len(all_test_responses) == 0:
        all_test_responses = []
        for t in range(len(new_test_responses)):
            all_test_responses.append(new_test_responses[t])
            all_test_responses[t]["pred"] = [all_test_responses[t]["pred"]]
            all_test_responses[t]["pred_prob"] = [all_test_responses[t]["pred_prob"]]
            all_test_responses[t]["correct_prob"] = [all_test_responses[t]["correct_prob"]]
            all_test_responses[t]["correct?"] = [all_test_responses[t]["correct?"]]
        return all_test_responses

    for t in range(len(all_test_responses)):
        all_test_responses[t]["pred"].append(new_test_responses[t]["pred"])
        all_test_responses[t]["pred_prob"].append(new_test_responses[t]["pred_prob"])
        all_test_responses[t]["correct_prob"].append(new_test_responses[t]["correct_prob"])
        all_test_responses[t]["correct?"].append(new_test_responses[t]["correct?"])
    return all_test_responses


def average_lines(lines, num_points=100):
    """
    Average a list of lines.

    Parameters:
        lines: A list of lines, where each line is a 2D numpy array with shape (n, 2),
               where n is the number of points on the line, and the 2 columns are x and y coordinates.
        num_points: The number of points to use in the averaged line. Default is 100.

    Returns:
        average_line: A 2D numpy array with shape (num_points, 2), where the columns are x and y coordinates
                      of the averaged line.
    """
    
    # Step 1: Create interpolation functions
    interp_funcs = [interp1d(line[:,0], line[:,1], kind='linear', fill_value='extrapolate') for line in lines]

    # Step 2: Define a set of x values
    min_x = min(line[:,0].min() for line in lines)
    max_x = max(line[:,0].max() for line in lines)
    x_values = np.linspace(min_x, max_x, num_points)

    # Step 3: Calculate y values at these x's for each line and average them
    y_values = np.mean([interp(x_values) for interp in interp_funcs], axis=0)

    # Step 4: Calculate errors on the mean
    y_errors = np.std([interp(x_values) for interp in interp_funcs], axis=0)

    # Now you have your average line
    average_line = np.column_stack((x_values, y_values))
    assert y_errors.shape[0] == y_values.shape[0]
    
    return average_line, y_errors