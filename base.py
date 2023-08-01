from pydantic import BaseModel, Field
import os

BASE_MODEL_DIR = os.getenv("BASE_MODEL_DIR", "/mnt/")
MODEL_NAME = os.getenv("MODEL_NAME", "model")

model_path = os.path.join(BASE_MODEL_DIR, MODEL_NAME)
lora = os.getenv("LORA", "no")
print(f"Model path: {model_path}")

class Req(BaseModel):
    prompt: str
    max_new_tokens: int | None = Field(default= 30, ge=3, le=2048, description="Max number of tokens to generate. Default 30. Range [3, 2048]")
    temperature: float | None = Field(default= 0.2, ge=0, le=1, description="Temperature of model is a float range [0,1]. Temperature 0 means minimum creativity, temperature 1 means max creativity.")
    repetition_penalty: float | None = Field(default= 1.2, ge=1.0, description="The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.")
    num_beams: int | None = Field(default= 1, ge=1, le=10, description="The parameter for num_beams refers to beam search, which is used for text generation. It returns the n most probable next words, rather than greedy search which returns the most probable next word.")

class Rep(BaseModel):
    prediction: str
    generated_tokens: int
