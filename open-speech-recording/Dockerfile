FROM google/cloud-sdk:alpine

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=main.py

ENV PYTHONUNBUFFERED 1

EXPOSE 8080
CMD exec gunicorn --bind :8080 --timeout 3600 app:app
