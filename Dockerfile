FROM ubuntu:18.04
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
COPY . /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade cython 
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get install -y p7zip-full && apt-get install -y wget
RUN mkdir output_yt
RUN mkdir output_yt/m
RUN wget https://cloud.monetka.name/s/B9iYgYAYgan6fWD/download && 7z x download && mv checkpoint-3740000/* output_yt/m/ && rm download
ENV PYTHONIOENCODING=UTF-8
ENTRYPOINT ["python3"]
CMD ["app.py"]