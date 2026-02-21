"""
LLM abstraction layer for BJJ Search.

Dispatches LLM calls to either Ollama (laptop profile) or OpenAI API (homeserver profile).
"""


class LLMClient:
    """Unified LLM interface that dispatches to Ollama or OpenAI based on profile."""

    def __init__(self, profile: str = "laptop", ollama_model: str = "llama3.1:8b"):
        self.profile = profile
        self.ollama_model = ollama_model

        if profile == "homeserver":
            from openai import OpenAI
            self._openai_client = OpenAI()
            self._openai_model = "gpt-4o-mini"
            print(f"OpenAI connected, using model: {self._openai_model}")
        else:
            import ollama
            self._ollama = ollama
            try:
                ollama.list()
                print(f"Ollama connected, using model: {ollama_model}")
            except Exception as e:
                raise RuntimeError(
                    f"Ollama not running. Start with: ollama serve\n"
                    f"Then pull model: ollama pull {ollama_model}"
                ) from e

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_output: bool = False,
    ) -> str:
        """Generate text from a prompt."""
        if self.profile == "laptop":
            options = {"temperature": temperature, "num_predict": max_tokens}
            kwargs = {"model": self.ollama_model, "prompt": prompt, "options": options}
            if json_output:
                kwargs["format"] = "json"
            response = self._ollama.generate(**kwargs)
            return response["response"]
        else:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": self._openai_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_output:
                kwargs["response_format"] = {"type": "json_object"}
            response = self._openai_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Chat-style completion from messages."""
        if self.profile == "laptop":
            response = self._ollama.chat(
                model=self.ollama_model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            return response["message"]["content"]
        else:
            response = self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
