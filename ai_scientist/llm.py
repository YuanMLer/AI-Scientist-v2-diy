import json
import os
import re
from typing import Any
from ai_scientist.utils.token_tracker import track_token_usage
from ai_scientist.config_loader import get_config, get_llm_config

import anthropic
import backoff
import openai
import requests

MAX_NUM_TOKENS = 4096

# Load available models from configuration
try:
    _conf = get_config()
    AVAILABLE_LLMS = list(_conf.llm_config.models.keys())
except Exception:
    AVAILABLE_LLMS = []

# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
@track_token_usage
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []
    
    # Get provider info
    try:
        conf = get_llm_config(model)
        provider = conf.get("provider")
        # Handle model override if specified in config (e.g. for Llama via OpenRouter)
        # But for batch responses, we usually pass the model name directly unless client is pre-configured?
        # Actually create_client returns (client, model_name). The model_name passed here is likely that one.
    except Exception:
        raise ValueError(f"Provider {provider} not supported for model {model} in get_batch_responses_from_llm.")

    if provider == "ollama":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif provider == "openai":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    # Add deepseek special handling if needed, or treat as openai
    else:
        raise ValueError(f"Provider {provider} not supported for model {model} in get_batch_responses_from_llm.")

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    try:
        conf = get_llm_config(model)
        provider = conf.get("provider")
    except Exception:
        provider = "openai"

    if provider == "ollama":
        return client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
    elif provider == "openai":
        # Check for O1/O3 models which have different API params
        if "o1" in model or "o3" in model:
             return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": system_message},
                    *prompt,
                ],
                temperature=1, # O1/O3 often fixed temp
                n=1,
                seed=0,
            )
        else:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *prompt,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
                seed=0,
            )
    else:
        raise ValueError(f"Model {model} with provider {provider} not supported in make_llm_call.")


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    try:
        conf = get_llm_config(model)
        provider = conf.get("provider")
    except Exception:
        logger.warning(f"模型{model}未在配置中找到（{e}），开始自动识别provider")
        raise RuntimeError(f"模型{model}未在配置中找到（{e}），开始自动识别provider")

    if provider == "anthropic":
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif provider == "ollama":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif provider == "openai":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} with provider {provider} not supported in get_response_from_llm.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> dict | None: 
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(model) -> tuple[Any, str]:
    conf = get_llm_config(model)
    provider = conf.get("provider")
    api_key = conf.get("api_key")
    base_url = conf.get("base_url")
    
    if provider == "anthropic":
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(api_key=api_key), model
        
    elif provider == "ollama":
        print(f"Using Ollama with model {model}.")
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        ), model
        
    elif provider == "openai":
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        ), model
        
    else:
        raise ValueError(f"Provider {provider} not supported for model {model} in create_client.")
