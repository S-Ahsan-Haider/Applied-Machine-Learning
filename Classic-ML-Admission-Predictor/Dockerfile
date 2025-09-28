FROM python:3.13.2-slim-bullseye 

WORKDIR /docker

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["python3","-m","flask","--app","admission", "run", "--host=0.0.0.0"] 

