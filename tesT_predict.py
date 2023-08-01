from predict import Predictor
from base import Req

predictor = Predictor(
)
    
predictor.setup()

json_req = {
    "prompt": "Give me a description of machine learning:",
    "max_new_tokens": 30,
    "temperature": 0.2,
    "repetition_penalty": 1.2
}

req = Req(**json_req)

print(predictor.predict(req))
