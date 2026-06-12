# 1. Use a lightweight Python base image to minimize footprint
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first to leverage Docker layer caching
COPY requirements.txt .

# 4. Install dependencies while clearing the pip cache to save space
# Note: We pull the CPU-only version of PyTorch to drastically reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code into the container
COPY src/ .

# 6. Set default model name
ARG HF_MODEL_NAME=pranshur10/codeberta-vuln-detector

# 7. Set model name as the environment variable
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# 8. Set the default command to execute the inference script
CMD ["python", "inference.py"]
