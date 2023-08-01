from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from base import Req, Rep, model_path, lora
import time

#
# Inference class using Transformers and PEFT
#
class Predictor():
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        config = PeftConfig.from_pretrained(model_path)
        print(config)
        inference_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_4bit=(lora == "int4"),
            load_in_8bit=(lora == "int8"),
            device_map={"": "cuda:0"}
            )

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side="left", legacy=False)
        self.model = PeftModel.from_pretrained(inference_model, model_path)

        # Initialize instance variables to keep track of metrics
        self.total_execution_time = 0.0
        self.total_tokens_generated = 0
        self.num_calls = 0

        return config

    # The arguments and types the model takes as input
    def predict(self, req: Req) -> Rep:
        start_time = time.time()  # Measure the start time
        """Run a single prediction on the model"""
        inputs = self.tokenizer(req.prompt, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to("cuda:0")

        # generate configuration can be modified to your needs
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
            num_beams=req.num_beams
        )[0]
        tokens = tokens[inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True)

        end_time = time.time()  # Measure the end time
        execution_time = end_time - start_time

        # Update cumulative metrics
        self.total_execution_time += execution_time
        self.total_tokens_generated += len(tokens)
        self.num_calls += 1

        print(f"Prediction executed in {execution_time:.4f} seconds")
        print(f"Number of tokens generated: {len(tokens)}")
        print(f"Tokens per second: {len(tokens) / execution_time:.2f}")
        print(f"Average Tokens per second: {self.total_tokens_generated / self.total_execution_time:.2f}")
        print(f"Number of calls: {self.num_calls}")

        rep = Rep(prediction=answer, generated_tokens=(len(tokens)))

        return rep
