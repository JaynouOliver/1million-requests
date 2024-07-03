#using huggingface model

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline

app = FastAPI()

# Load the model and tokenizer
model_path = "./model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

class TextInput(BaseModel):
    text: str

@app.post("/classify/")
def classify_text(input: TextInput):
    result = pipeline(input.text)
    return {"label": result[0]['label'], "score": result[0]['score']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
