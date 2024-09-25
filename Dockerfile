FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN apt update && apt install ffmpeg libpq-dev libsm6 libxext6 build-essential -y
RUN pip install -r requirements.txt
EXPOSE 8011
CMD python ./server.py