FROM ubuntu
WORKDIR /app
COPY . .

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip install -r requirements.txt
RUN python3 nltk_stopwords.py
CMD python3 -m uvicorn --host 0.0.0.0 --port 8000 main:app
