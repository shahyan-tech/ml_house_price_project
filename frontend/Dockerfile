# frontend/Dockerfile
# Dockerfile for the Streamlit frontend.
# Builds a containerized app that can run alongside the API in K8s.

FROM python:3.11-slim

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code
COPY app.py .

# Expose port 8501 (default for Streamlit)
EXPOSE 8501

# Run the app with server settings for containerized deployment
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]