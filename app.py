#!/usr/local/bin/python3
# coding: utf8
import os
from rq import Queue
from rq.job import Job
from worker import conn
#from sqlalchemy import SQLAlchemy
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, abort, jsonify, request
from flask_cors import CORS, cross_origin

from prometheus_flask_exporter import PrometheusMetrics

from run_generation2 import generate_text, load_model


app = Flask(__name__)

#app.config.from_object(os.environ['APP_SETTINGS'])
#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
#db = SQLAlchemy(app)

q = Queue(connection=conn)


metrics = PrometheusMetrics(app)

# static information as metric
metrics.info('app_info', 'Application info', version='2.3')

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model, tokenizer = load_model(no_cuda=False)

def send_to_generation(text):
    return  generate_text(
            model,
            tokenizer,
            model_type='gpt2',
            length=20,
            prompt=text,
            temperature=0.9,
            no_cuda=False
            )

@app.route("/generate", methods=['POST'])
@cross_origin()
def get_gen():
    data = request.get_json()

    if 'text' not in data or len(data['text']) == 0:
        abort(400)
    else:
        text = data['text']
        #model = data['model']
        '''
        result = generate_text(
            model,
            tokenizer,
            model_type='gpt2',
            length=20,
            prompt=text,
            temperature=0.9,
            no_cuda=False
            )
        '''
        #_args = (model, tokenizer, model_type = 'gpt2', length=20, prompt=text,temperature=0.9,no_cuda=False)
        job = q.enqueue_call(
            func=send_to_generation, args = text, result_ttl=5000
        )
        return jsonify({'result': result})


app.run(debug=False,host='0.0.0.0', port = 5000)