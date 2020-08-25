FROM ubuntu:18.04
FROM python:3
COPY . /usr/src/app
EXPOSE 8080
WORKDIR /usr/src/app
RUN set -xe \
    && apt-get update \
    && apt-get install python-pip -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "python", "./main.py" ]
