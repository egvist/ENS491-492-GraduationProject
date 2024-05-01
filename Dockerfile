FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire app directory into the container
COPY ./app /app/app

# Set the entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
