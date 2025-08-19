# Dockerfile for FragNet Streamlit app
FROM python:3.11-slim

# Install system dependencies for building torch-scatter
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html


# Copy the source code
COPY . .

# Install FragNet as a package
RUN pip install .

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables for Streamlit
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501

# Run the Streamlit app
CMD ["streamlit", "run", "fragnet/vizualize/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
