FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port 80
EXPOSE 80

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "4", "app:app"]
