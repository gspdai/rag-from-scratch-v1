from anthropic import Anthropic
from anthropic.types import Message

from .vectordb import TopMatch


class Answerer():
    def __init__(self, key:str, model_name:str, system_prompt:str):
        self.client = Anthropic(api_key = key)
        self.model = model_name
        self.prompt = system_prompt
        self.max_token = 1024

    def ask(self, query:str, top_matches:list[TopMatch]) -> Message:
        text = []
        citation = []
        for top_match in top_matches:
            citations = top_match.chunk.citation
            text.append(top_match.chunk.text)
            citation.append(",".join(["[Page "+str(cite)+"]" for cite in citations]))

        matching_text = "\n".join([c+"\n"+t+"\n" for c,t in zip(citation,text)])

        message = self.client.messages.create(
            max_tokens = self.max_token,
            model = self.model,
            system= self.prompt,
            messages = [{"role": "user", "content": f'''Question:\n Please respond to this {query} using the context that follows.\nContext:\n {matching_text}'''}]
        )
        return message