# Use an official Python runtime as a parent image
FROM python:3.9-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt and system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the application code and model files into the container at /app
COPY classify_api.py .
COPY best_cnn_lstm_model_deeper.keras .
COPY tokenizer_config.json .

# Set environment variable for NLTK data path inside the container
ENV NLTK_DATA=/app/nltk_data

# Download NLTK data directly during the build process
# This is the reliable way to get data into the container image
RUN python -c "import nltk; \
nltk.download('punkt', download_dir='/app/nltk_data'); \
nltk.download('stopwords', download_dir='/app/nltk_data'); \
nltk.download('wordnet', download_dir='/app/nltk_data'); \
nltk.download('omw-1.4', download_dir='/app/nltk_data'); \
nltk.download('vader_lexicon', download_dir='/app/nltk_data'); \
nltk.download('punkt_tab', download_dir='/app/nltk_data');" # <-- Added punkt_tab here


# Make port 8000 available to the outside world (the container network)
EXPOSE 8000

# Define the command to run your application using Gunicorn and Uvicorn workers
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "classify_api:app"]