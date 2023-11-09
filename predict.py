from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
from base import Req, Rep, model_path, lora
import time
from monitor import Monitor

#
# Inference class using Transformers and PEFT
#
class Predictor():

    def __init__(self, monitor: Monitor):
        self.monitor = monitor

    def setup(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="cuda")


    # The arguments and types the model takes as input
    def predict(self, req: Req) -> Rep:
        start_time = time.time()  # Measure the start time

        generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=req.temperature,
            max_new_tokens=req.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )

        """Run a single prediction on the model"""
        inputs = self.tokenizer(req.prompt, return_tensors='pt')
        inputs.to(device=self.device)

        # generate configuration can be modified to your needs
        tokens = self.model.generate(**inputs, generation_config=generation_config)[0]
        
        tokens = tokens[inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True)

        self.monitor.update_metrics(start_time, len(tokens))
        self.monitor.display_metrics()

        rep = Rep(prediction=answer, generated_tokens=(len(tokens)))

        return rep
