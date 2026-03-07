FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config.yaml .
COPY environment.py .
COPY agent.py .
COPY simulate.py .
COPY visualize.py .
COPY dashboard.py .
COPY main.py .
COPY TEAM.txt .
COPY templates/ templates/

# Create output directories
RUN mkdir -p output charts

EXPOSE 5000

# Default: run the live dashboard
CMD ["python", "dashboard.py"]
