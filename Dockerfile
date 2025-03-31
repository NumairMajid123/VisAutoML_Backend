FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    libc6 \
    nodejs\
    npm \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Collect static files
RUN python manage.py collectstatic --noinput

# Apply database migrations
RUN python manage.py migrate

# Expose the port the app runs on
EXPOSE 8000

# Start the application
CMD ["gunicorn", "VisAutoML.wsgi:application", "--bind", "0.0.0.0:8000"]
