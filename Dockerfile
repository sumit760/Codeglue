# 1. Use a lightweight Python base image to minimize footprint
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first to leverage Docker layer caching
COPY requirements.txt .

# 4. Install dependencies while clearing the pip cache to save space
# Note: We pull the CPU-only version of PyTorch to drastically reduce image size
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code into the container
COPY Dev/SRC/ ./Dev/SRC/

# 6. Set the default command to execute the inference script
CMD ["python", "Dev/SRC/inference.py"]