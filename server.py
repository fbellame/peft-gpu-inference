from fastapi import FastAPI
from predict import Predictor
from base import Req, model_path

# Create a FastAPI instance
app = FastAPI()
predictor = Predictor()
config = predictor.setup()

@app.get("/")
async def root():
    return {
        "model path": model_path, 
        "peft_type": config.peft_type, 
        "base_model_name_or_path": config.base_model_name_or_path,
        "task_type=":  config.task_type,
        "revision": config.revision
        }

# Add the FastAPI route for the /predict endpoint
@app.post("/predict")
async def predict(req: Req):
    return predictor.predict(req)

# This block is only required for development purposes
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
