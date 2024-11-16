FROM python:3.10-slim

# Set environment variables
ENV FLASK_ENV=production
ENV SECRET_KEY="***REMOVED***"

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Copy the requirements file and install dependencies
COPY local_requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your application runs on
EXPOSE 5005

# Command to run the application in production mode using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5005", "app:app"]
