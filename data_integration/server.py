from flask import Flask, render_template, Response
import json
from data_integration.data_generator import get_data_stream

app = Flask(__name__,
          template_folder='templates',
          static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    return Response(data_stream(), mimetype="text/event-stream")

def data_stream():
    last_sent = None
    for data in get_data_stream():
        if data != last_sent:
            last_sent = data
            yield f"data: {json.dumps(data)}\n\n"


if __name__ == '__main__':
    app.run(threaded=True, port=5000)