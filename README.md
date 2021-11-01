# SBWBackend
FastAPI server to make requests to finetuned model based on [ru_transformers](https://github.com/mgrankin/ru_transformers).
Model is finetuned on 750 mb of My Little Pony fanfics using v100 GPU which you can access [here](https://cloud.monetka.name/s/B9iYgYAYgan6fWD/download).\
Text corpus is from [ponyfiction](https://ponyfiction.org/news/torrent/)/\
DockerHub with gpu support is [here](https://hub.docker.com/repository/docker/monetka/sbwgpu) and only cpu-version is [here](https://hub.docker.com/repository/docker/monetka/sbwbackend).\
Use `/generate` POST method with payload like that: 
```json
{"text": "Твайлайт Спаркл удивлённо ", "lenght": 40, "temperature": 1}
```
To generate.
Params of request are:
```
Input:
    text (str): input text
    length (int): lenght of generation in tokens
    temperature (float): "creativness" of generation
Returns:
    str: generated text
```