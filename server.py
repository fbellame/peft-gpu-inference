from fastapi import FastAPI
from predict import Predictor
from monitor import Monitor
from base import Req, model_path

# Create a FastAPI instance
monitor = Monitor()
app = FastAPI()
predictor = Predictor(monitor)
predictor.setup()

@app.get("/")
async def root():
    return {
        "model path": model_path
        }

@app.get("/metrics")
async def metrics():
    return monitor.metrics()

# Add the FastAPI route for the /predict endpoint
@app.post("/predict")
async def predict(req: Req):
    return predictor.predict(req)

# This block is only required for development purposes
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
