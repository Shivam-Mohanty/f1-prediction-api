FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create the cache directory for FastF1
RUN mkdir -p fastf1_cache

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}