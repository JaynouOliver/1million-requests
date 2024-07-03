# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model directory into the container
COPY ./model ./model

# Copy the application code into the container
COPY app3.py .

# Expose the port that the app runs on
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "app3:app", "--host", "0.0.0.0", "--port", "8000"]
