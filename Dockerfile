FROM nvcr.io/nvidia/pytorch:19.12-py3
RUN apt-get update -y
#RUN apt-get install -y python3-pip python3-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install --upgrade cython 
RUN pip install --no-cache torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install transformers==2.2
#RUN pip install --ignore-installed certifi -r requirements.txt
RUN pip install loguru
#RUN pip install sentencepiece
RUN pip install youtokentome
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install nest_asyncio 
RUN apt-get install -y p7zip-full && apt-get install -y wget
RUN mkdir output_yt
RUN mkdir output_yt/m
RUN wget https://cloud.monetka.name/s/B9iYgYAYgan6fWD/download && 7z x download && mv checkpoint-3740000/* output_yt/m/ && rm download
ENV PYTHONIOENCODING=UTF-8
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["api.py"]