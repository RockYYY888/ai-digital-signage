from flask import Flask, render_template, Response
import time
import json
from data_integration.data_generator import get_data_stream
# from data_integration.data_interface import eye_tracking_queue

app = Flask(__name__,
          template_folder='templates',
          static_folder='static')

# Status constants
STATUS_NO_FACE = "no_face"
STATUS_DETECTED = "detected"
STATUS_ANALYZING = "analyzing"
STATUS_FINISHED = "finished"
STATUS_FEEDBACK = "feedback"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    return Response(data_stream(), mimetype="text/event-stream")

# @app.route('/eye_tracking_stream')
# def eye_tracking_stream():
#     def generate():
#         while True:
#             if not eye_tracking_queue.empty():
#                 data = eye_tracking_queue.get()
#                 yield f"data: {json.dumps(data)}\n\n"
#             time.sleep(0.1)
#     return Response(generate(), mimetype="text/event-stream")

def data_stream():
    """Generating stateful data streams"""
    for data in get_data_stream():
        # print("Generated data:", data)  # Print data
        yield f"data: {json.dumps(data)}\n\n"

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
