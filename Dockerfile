FROM python:3.10
WORKDIR /opt
COPY . .
RUN apt-get update \
    && apt-get -y install ffmpeg git\
    && pip3 install --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -r requirements.txt
RUN curl -o /root/nltk_data/tokenizers/punkt.zip \
    --create-dirs -L https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip \
    && unzip /root/nltk_data/tokenizers/punkt.zip -d /root/nltk_data/tokenizers/
EXPOSE 8000
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]