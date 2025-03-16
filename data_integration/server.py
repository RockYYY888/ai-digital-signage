from flask import Flask, render_template, Response, Blueprint
import json
from data_integration.data_generator import get_data_stream
from data_integration.data_interface import frame_queue
import cv2
import os

from flask import Blueprint

secondary_screen_app = Blueprint(
    "secondary_screen",  # Blueprint 名称
    __name__,  # import_name
    template_folder="templates",
    static_folder="static"
)

@secondary_screen_app.route('/')
def index():
    return render_template('/index.html')

@secondary_screen_app.route('/stream')
def stream():
    return Response(data_stream(), mimetype="text/event-stream")

@secondary_screen_app.route('/face_image')
def face_image():
    """返回检测到人脸的单张图片"""
    if not frame_queue.empty():
        frame = frame_queue.get()
        print("[server.py]:Found a frame of face.")
        _, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    # 使用绝对路径加载默认图片
    default_image_path = os.path.join(secondary_screen_app.static_folder, 'no_face.jpg')
    if os.path.exists(default_image_path):
        with open(default_image_path, 'rb') as f:
            return Response(f.read(), mimetype='image/jpeg')
    # 如果文件不存在，返回空响应
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