from ollama import chat
from pydantic import BaseModel

class Summary(BaseModel):
  theses: list[str]

def summarize(text: str):
    response = chat(
        model='qwen3:8b',
        messages=[{'role': 'user', 'content':
            f"Summarize the key points from this chapter in at most 3 succinct self contained claims, each a single sentence long.\n\n{text[:-5000]}"
        }],
        format=Summary.model_json_schema(),
        options={'temperature': 0})
    return Summary.model_validate_json(response.message.content).theses
