from typing import List, Optional, Any
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.prompts import ChatPromptTemplate
import uuid
import json
import requests

class GPT(LLM):

    def __init__(self):
        super().__init__()
        print("construct gpt")

    def _call(self,
              prompt: ChatPromptTemplate
              ) -> str:
        messages = [
            {"role": "user", "content": prompt.messages[0].content}
        ]
        response = self.venus_completion(messages)
        return response['content']

    def call(self,
             messages: List[dict],
             stop: Optional[List[str]] = None,
             callbacks: Optional[CallbackManagerForLLMRun] = None,
             **kwargs: Any,
             ) -> str:
        response = self.completion(messages)
        return response

    def _llm_type(self) -> str:
        return "gpt"

    def completion(self, messages: List[dict]) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        model = "gpt-4o-2024-08-06"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Your API Key",
        }

        data = {
            "model": model,
            "messages": messages,
        }
        
        while True:
            try:
                response = requests.post(url, headers=headers, json=data)
                break
            except Exception as e:
                print(f"Retrying...")
   
        return json.loads(response.text)['choices'][0]['message']

if __name__ == '__main__':
    llm = GPT()
    messages = [{"role": "user", "content": "Hello"}]
    response = llm.call(messages)
    print(response)
    