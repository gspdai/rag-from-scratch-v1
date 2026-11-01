import mlflow
from anthropic import Anthropic
from anthropic.types import Message

from rag_from_scratch_v1.guardrails import mask_pii
from rag_from_scratch_v1.pricing import Price

from .vectordb import TopMatch


class Answerer():
    def __init__(self, key:str, model_name:str, system_prompt:str, price:Price):
        self.client = Anthropic(api_key = key)
        self.model = model_name
        self.prompt = system_prompt
        self.max_token = 1024
        self.token_price = price


    def ask(self, query:str, top_matches:list[TopMatch]) -> Message:
        with mlflow.start_span(name="llm_response") as span:
            text = []
            citation = []
            chunk_ids = []
            for top_match in top_matches:
                citations = top_match.chunk.citation
                text.append(top_match.chunk.text)
                citation.append(",".join(["[Page "+str(cite)+"]" for cite in citations]))
                chunk_ids.append(top_match.chunk.id)

            matching_text = "\n".join([c+"\n"+t+"\n" for c,t in zip(citation,text)])

            mask_query = mask_pii(query)

            span.set_inputs({"user_query":mask_query, "chunk_ids":chunk_ids})

            message = self.client.messages.create(
                max_tokens = self.max_token,
                model = self.model,
                system= self.prompt,
                messages = [{"role": "user", "content": f'''Question:\n Please respond to this {query} using the context that follows.\nContext:\n {matching_text}'''}]
            )
            response_text = ""
            for block in message.content:
                if block.type == "text":
                    response_text = mask_pii(block.text)
                    break            
            input_cost = self.token_price.input_token_cost_per_million*message.usage.input_tokens/1000000
            output_cost = self.token_price.output_token_cost_per_million*message.usage.output_tokens/1000000
            span.set_outputs({"llm_response":response_text, 
                              "input_tokens":message.usage.input_tokens, 
                              "output_tokens":message.usage.output_tokens,
                              "input_cost":input_cost,
                              "output_cost":output_cost,
                              "total_cost":input_cost+output_cost})
            return message