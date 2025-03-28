# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
from flask import render_template, Response
import json
from Server.data_generator import get_data_stream
from Server.data_interface import frame_queue
import cv2
import os

from flask import Blueprint

from util import get_resource_path

secondary_screen_app = Blueprint(
    "secondary_screen",  # Blueprint name
    __name__,  # import_name
    template_folder=get_resource_path("Server/templates"),
    static_folder=get_resource_path("static")
)

@secondary_screen_app.route('/')
def index():
    return render_template('/index.html')

@secondary_screen_app.route('/stream')
def stream():
    return Response(data_stream(), mimetype="text/event-stream")

@secondary_screen_app.route('/face_image')
def face_image():
    """Returns a single image of the detected face"""
    if not frame_queue.empty():
        # Get the first frame
        frame = frame_queue.get()
        # Clear the remaining items in the queue
        while not frame_queue.empty():
            frame_queue.get()
        # print("[server.py]:Found a frame of face and cleared queue.")

        _, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    # Load default image using absolute path
    default_image_path = os.path.join(secondary_screen_app.static_folder, 'no_face.jpg')
    if os.path.exists(default_image_path):
        with open(default_image_path, 'rb') as f:
            return Response(f.read(), mimetype='image/jpeg')
    
    # Return empty response if file does not exist
    print(f"Warning: {default_image_path} not found")
    return Response(b'', mimetype='image/jpeg')

def data_stream():
    last_sent = None
    for data in get_data_stream():
        if data != last_sent:
            last_sent = data
            yield f"data: {json.dumps(data)}\n\n"

if __name__ == '__main__':
    secondary_screen_app.run(threaded=True, port=5000)