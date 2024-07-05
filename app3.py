from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
import ray
from ray import serve

app = FastAPI()

# Initialize Ray and Ray Serve only once
ray.init(ignore_reinit_error=True)
serve.start()

class TextInput(BaseModel):
    text: str

@serve.deployment
@serve.ingress(app)
class MyModelDeployment:
    def __init__(self):
        model_path = "./model"
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer)

    @app.post("/classify/")
    def classify_text(self, input: TextInput):
        result = self.pipeline(input.text)
        return {"label": result[0]['label'], "score": result[0]['score']}

# Deploy the model with route_prefix
serve.run(MyModelDeployment.bind(), name="my_model", route_prefix="/classify")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
