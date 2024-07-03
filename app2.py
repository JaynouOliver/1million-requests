#using onnyx model

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline

# Initialize the FastAPI app
app = FastAPI()

# Define the model path
model_path = "./model"

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Create the pipeline
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

# Define the input schema
class TextInput(BaseModel):
    text: str

# Define the predict endpoint
@app.post("/predict")
def predict(input: TextInput):
    # Perform the prediction
    result = pipeline(input.text)[0]
    return result
