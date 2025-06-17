from typing import Optional, Dict, Any, Union
import json
import httpx
import logging
from pydantic import BaseModel, Field
from deepeval.models import DeepEvalBaseLLM

import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class LLMRequestError(Exception):
    """
    Custom exception for LLM request errors, providing detailed context for debugging.
    """
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        request_url: Optional[str] = None,
        request_payload: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.status_code = status_code
        self.response_text = response_text
        self.request_url = request_url
        self.request_payload = request_payload
        self.original_exception = original_exception

        error_msg = f"LLMRequestError: {message}"
        if status_code is not None:
            error_msg += f" | Status: {status_code}"
        if request_url:
            error_msg += f" | URL: {request_url}"
        if request_payload:
            try:
                payload_str = json.dumps(request_payload)
            except Exception:
                payload_str = str(request_payload)
            error_msg += f" | Payload: {payload_str}"
        if response_text:
            error_msg += f" | Response: {response_text}"
        if original_exception:
            error_msg += f" | Original Exception: {repr(original_exception)}"
        super().__init__(error_msg)

class CustomOpenAILLM(DeepEvalBaseLLM):
    """
    A custom LLM class that implements OpenAI-compatible endpoints for DeepEval
    """
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        verify_ssl: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the custom LLM.

        Args:
            api_key (str): API key for authentication
            model_name (str): Name of the model to use
            base_url (Optional[str]): Base URL for the API endpoint
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
            verify_ssl (bool): Whether to verify SSL certificates
            debug (bool): Enable debug logging
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip('/')
        self.temperature = min(max(temperature, 0.0), 1.0)  # Clamp between 0 and 1
        self.max_tokens = max_tokens
        self.verify_ssl = verify_ssl

        if debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)

    def load_model(self) -> 'CustomOpenAILLM':
        """Required by DeepEval"""
        return self

    def get_model_name(self) -> str:
        """Required by DeepEval"""
        return self.model_name

    def _prepare_request_payload(self, prompt: str) -> Dict[str, Any]:
        """Prepare the request payload"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        return payload

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _handle_response(self, response: httpx.Response) -> str:
        """Handle the API response and extract the content"""
        try:
            response_data = response.json()
            self.logger.debug(f"Response data: {response_data}")

            if not response.is_success:
                raise LLMRequestError(
                    "Request failed",
                    status_code=response.status_code,
                    response_text=response.text
                )

            if "choices" not in response_data or not response_data["choices"]:
                raise LLMRequestError("No choices in response",
                                    response.status_code,
                                    response.text)

            if "message" not in response_data["choices"][0]:
                # Try alternative response formats
                if "text" in response_data["choices"][0]:
                    return response_data["choices"][0]["text"]
                raise LLMRequestError("Unexpected response format",
                                    response.status_code,
                                    response.text)

            return response_data["choices"][0]["message"]["content"]

        except json.JSONDecodeError:
            raise LLMRequestError("Invalid JSON response",
                                response.status_code,
                                response.text)

    def generate(self, prompt: str) -> Union[str, BaseModel]:
        """Generate a response from the LLM"""
        self.logger.debug(f"Generating response for prompt: {prompt}")

        headers = self._prepare_headers()
        payload = self._prepare_request_payload(prompt)

        self.logger.debug(f"Request URL: {self.base_url}/chat/completions")
        self.logger.debug(f"Request Headers: {headers}")
        self.logger.debug(f"Request Payload: {payload}")

        try:
            with httpx.Client(verify=self.verify_ssl) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )

                response_text = self._handle_response(response)

                return response_text

        except httpx.RequestError as e:
            raise LLMRequestError(f"Request failed: {str(e)}")

    async def a_generate(self, prompt: str) -> Union[str, BaseModel]:
        """Generate a response from the LLM asynchronously"""
        self.logger.debug(f"Generating async response for prompt: {prompt}")

        headers = self._prepare_headers()
        payload = self._prepare_request_payload(prompt)

        try:
            async with httpx.AsyncClient(verify=self.verify_ssl) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )

                response_text = self._handle_response(response)
                
                return response_text

        except httpx.RequestError as e:
            raise LLMRequestError(f"Async request failed: {str(e)}")
     