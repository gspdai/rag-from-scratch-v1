from dataclasses import dataclass

@dataclass (frozen=True)
class Price:
    input_token_cost_per_million:float
    output_token_cost_per_million:float = 0.0