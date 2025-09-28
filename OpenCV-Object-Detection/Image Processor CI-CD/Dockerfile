# Use a modern, actively maintained Python runtime as a parent image
FROM python:3.9-bullseye

# Install the necessary dependencies for OpenCV.
# This base image should have up-to-date repositories,
# so the previous apt-get install command should work now.
RUN apt-get update -y && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    ffmpeg \
    --no-install-recommends

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run your application when the container launches
CMD ["python", "simple_image_processor.py"]
