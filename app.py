#!/usr/local/bin/python3
# coding: utf8
from flask import Flask, abort, jsonify, request
from flask_cors import CORS, cross_origin

from prometheus_flask_exporter import PrometheusMetrics

from run_generation2 import generate_text, load_model

app = Flask(__name__)

metrics = PrometheusMetrics(app)

# static information as metric
metrics.info('app_info', 'Application info', version='2.3')

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model, tokenizer = load_model()

@app.route("/generate", methods=['POST'])
@cross_origin()
def get_gen():
    data = request.get_json()

    if 'text' not in data or len(data['text']) == 0 or 'model' not in data:
        abort(400)
    else:
        text = data['text']
        #model = data['model']

        result = generate_text(
            model,
            tokenizer,
            model_type='gpt2',
            length=20,
            prompt=text,
            temperature=0.9
            )

        return jsonify({'result': result})


app.run(debug=False,host='0.0.0.0', port = 5000)