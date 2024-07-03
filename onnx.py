from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import BertTokenizer
import numpy as np

app = FastAPI()

# Load the tokenizer
model_path = "./model"
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load the ONNX model
onnx_model_path = "model.onnx"  # Use the optimized and quantized model
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# Check the expected sequence length
expected_seq_len = ort_session.get_inputs()[0].shape[1]

class TextInput(BaseModel):
    text: str

def classify_text(text):
    inputs = tokenizer(text, return_tensors="np", padding='max_length', truncation=True, max_length=expected_seq_len)
    ort_inputs = {k: np.array(v) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    ort_outs = ort_session.run(None, ort_inputs)
    label_id = ort_outs[0].argmax(axis=1)[0]
    labels = ["NEGATIVE", "POSITIVE"]  # Assuming binary classification
    return {"label": labels[label_id], "score": float(ort_outs[0].max())}

@app.post("/classify/")
async def classify_text_endpoint(input: TextInput):
    result = classify_text(input.text)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
