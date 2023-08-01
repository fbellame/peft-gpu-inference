
```sh
sudo docker build . -t peft-gpu-inference:11.7.1
```

```sh
sudo docker run --gpus all -p 8000:8000 -v /mnt/ssd2/h2o-llmstudio/output/user/pdf_to_quizz_llama_13B.1.1.1.1.1.1:/mnt/model -v /home/farid/.cache/huggingface/hub:/mnt/.cache/huggingface/hub  peft-gpu-inference:latest
```

```sh
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Machine learning is"}' http://0.0.0.0:8000/predict
```
