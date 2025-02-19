# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import asyncio
from openai import AsyncAzureOpenAI, AsyncOpenAI, RateLimitError
import tiktoken

import time 

from functools import cache

from transformers import AutoTokenizer
from typing import Union, List

from vertexai.preview.tokenization import get_tokenizer_for_model
import vertexai.preview.generative_models as generative_models

from langchain_aws import ChatBedrockConverse

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from tenacity import retry, wait_random_exponential, stop_after_delay, retry_if_exception_type, retry_if_exception_message, wait_random

class APIConnector:
    def __init__(
        self,
        api_key: str,
        api_url: str,
        api_provider: str,
        model: str,
        **kwargs
    ) -> None:
        """
        APIConnector class to unify the API calls for different API providers
        
        Parameters:
            api_key (`str`): API Key for the API
            api_url (`str`): API URL for the API
            api_provider (`str`): API Provider (openai, gemini, aws, vllm, azure-openai, etc.)
            model (`str`): Model name to use for the API
        """
        self.api_provider = api_provider

        if api_provider == "openai":
            self.api = AsyncOpenAI(
                api_key=api_key,
                base_url=api_url,
            )
            if "o1" in model or "o3" in model:
                self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
            else:
                self.tokenizer = tiktoken.encoding_for_model(model)
            self.SYSTEM_PROMPT = "You are a helpful assistant"
        elif api_provider == "azure-openai":
            self.api = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_url,
                api_version=kwargs["azure_api_version"]
            )
            if "o1" in model or "o3" in model:
                self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
            else:
                self.tokenizer = tiktoken.encoding_for_model(model)
            self.SYSTEM_PROMPT = "You are a helpful assistant"
        elif api_provider == "vllm":
            self.api = AsyncOpenAI(
                api_key=api_key,
                base_url=api_url,
                max_retries=kwargs["max_retries"],
                timeout=kwargs["timeout"]
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
            self.SYSTEM_PROMPT = "You are a helpful assistant"
        elif api_provider == "gemini":
            self.SYSTEM_PROMPT = "You are a helpful assistant"
            self.tokenizer = get_tokenizer_for_model(model)
            self.api = genai.Client(
                vertexai=True, project=kwargs["project_ID"], location=kwargs["location"]
            )

            self.gemini_retry_config ={
                "initial": kwargs["retry_delay"],
                "maximum": kwargs["retry_max_delay"],
                "multiplier": kwargs["retry_multiplier"],
                "timeout": kwargs["retry_timeout"]
            }

        elif api_provider == "aws":
            self.api = ChatBedrockConverse(
                model=model,
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"],
                region_name=kwargs["region"],
                top_p=kwargs["top_p"],
            )
            self.SYSTEM_PROMPT = """"""

            ### NOTE: Since claude hasn't released an official tokenizer, we use a llama tokenizer instead to get an estimate of the token count
            if "claude" in model:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(kwargs["tokenizer_name"], use_fast=True)
                self.SYSTEM_PROMPT = "You are a helpful assistant"

        self.model = model

    def encode(self, text: str) -> list:
        """
        Encodes the given text using the tokenizer
        
        Parameters:
            text (`str`): Text to encode
        
        Returns:
            `dict`: Encoded text
        """
        if self.api_provider == "openai" or self.api_provider == "azure-openai":
            return self.tokenizer.encode(text)
        elif self.api_provider == "vllm":
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        elif self.api_provider == "gemini":
            return self.tokenizer._sentencepiece_adapter._tokenizer.encode(text)
        elif self.api_provider == "aws":
            # Check the NOTE in the __init__ method
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        else:
            raise ValueError(f"Invalid API provider: {self.api_provider}")

    def decode(self, tokens: list) -> str:
        """
        Decodes the given tokens using the tokenizer
        
        Parameters:
            tokens (`list`): Tokens to decode
        
        Returns:
            `str`: Decoded text
        """
        if self.api_provider == "openai" or self.api_provider == "azure-openai":
            return self.tokenizer.decode(tokens)
        elif self.api_provider == "vllm":
            return self.tokenizer.decode(tokens)
        elif self.api_provider == "gemini":
            return self.tokenizer._sentencepiece_adapter._tokenizer.decode(tokens)
        elif self.api_provider == "aws":
            # Check the NOTE in the __init__ method
            return self.tokenizer.decode(tokens)
        else:
            raise ValueError(f"Invalid API provider: {self.api_provider}")

    @cache
    def token_count(self, text: str) -> int:
        """
        Returns the token count of the given text
        
        Parameters:
            text (`str`): Text to count tokens
            use_cache (`bool`, optional): Use cache for token count. Defaults to True.
        
        Returns:
            `int`: Token count of the text
        """
        if self.api_provider == "openai" or self.api_provider == "azure-openai":
            return len(self.tokenizer.encode(text))
        elif self.api_provider == "vllm":
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        elif self.api_provider == "gemini":
            return self.tokenizer.count_tokens(text).total_tokens
        elif self.api_provider == "aws":
            # Check the NOTE in the __init__ method
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        else:
            raise ValueError(f"Invalid API provider: {self.api_provider}")
            
    async def generate_response(
        self, 
        system_prompt: str,
        user_prompt: Union[str, List[str]],
        max_tokens: int = 100, 
        temperature: float = 0.0, 
        top_p: float = 1.0,
        add_default_system_prompt: bool = True
    ) -> dict:
        """
        Generates a response using the API with the given prompts
        
        Parameters:
            system_prompt (`str`): System prompt
            user_prompt (`str`): User prompt
            max_tokens (`int`, optional): Maximum tokens to generate. Defaults to 100.
            temperature (`float`, optional): Temperature. Defaults to 0.0 (Greedy Sampling).
            top_p (`float`, optional): Top-p. Defaults to 1.0.
        
        Returns:
            `dict`: Response from the API that includes the response, prompt tokens count, completion tokens count, total tokens count, and stopping reason
        """
        if add_default_system_prompt and self.SYSTEM_PROMPT != "":
            messages = [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                }
            ]
        elif system_prompt != "":
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
        else:
            messages = []
        if isinstance(user_prompt, list):
            for prompt in user_prompt:
                messages.append({
                    "role": "user",
                    "content": prompt
                })
        else:
            messages.append({
                "role": "user",
                "content": user_prompt
            })
        if self.api_provider == "openai" or self.api_provider == "vllm" or self.api_provider == "azure-openai":
            @retry(reraise=True, wait=wait_random(1, 20), retry=retry_if_exception_type(RateLimitError), stop=stop_after_delay(300))
            async def generate_content():
                completion = await self.api.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    top_p=top_p,
                    seed=43,
                    **({"max_completion_tokens": max_tokens} if "o1" in self.model or "o3" in self.model else {"max_tokens": max_tokens, "temperature": temperature})
                )
                return completion
            
            completion = await generate_content()
            

            output = {
                "response": completion.choices[0].message.content,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
                "finish_reason": completion.choices[0].finish_reason
            }
            if "o1" in self.model or "o3" in self.model:
                output["reasoning_tokens"] = completion.usage.completion_tokens_details.reasoning_tokens
            return output

        elif self.api_provider == "gemini":
            @retry(reraise=True, wait=wait_random_exponential(multiplier=self.gemini_retry_config["multiplier"], max=self.gemini_retry_config["maximum"]), stop=stop_after_delay(self.gemini_retry_config["timeout"]), retry=retry_if_exception_type(ClientError))
            async def generate_content():
                completion = await self.api.aio.models.generate_content(
                    model=self.model,
                    contents=messages[-1]["content"],
                    config=types.GenerateContentConfig(
                        system_instruction=self.SYSTEM_PROMPT,
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                        seed=43,

                    )
                )

                return completion

            completion = await generate_content()

            return {
                "response": completion.text,
                "prompt_tokens": completion.usage_metadata.prompt_token_count,
                "completion_tokens": completion.usage_metadata.candidates_token_count,
                "total_tokens": completion.usage_metadata.total_token_count,
                "finish_reason": completion.candidates[0].finish_reason
            }
        elif self.api_provider == "aws":
            @retry(reraise=True, wait=wait_random(5, 20), retry=retry_if_exception_message(match=r".*ThrottlingException.*"))
            async def generate_content():
                completion = await self.api.ainvoke(messages)
                return completion
            
            completion = await generate_content()

            return {
                "response": completion.content,
                "prompt_tokens": completion.usage_metadata["input_tokens"],
                "completion_tokens": completion.usage_metadata["output_tokens"],
                "total_tokens": completion.usage_metadata["total_tokens"],
                "finish_reason": completion.response_metadata["stopReason"]
            }
        else:
            raise ValueError(f"Invalid API provider: {self.api_provider}")

