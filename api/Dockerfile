FROM python:3.9-slim-buster


RUN apt-get update --allow-releaseinfo-change
RUN apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]