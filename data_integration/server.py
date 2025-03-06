from flask import Flask, render_template, Response
import json
from data_integration.data_generator import get_data_stream
from data_integration.data_interface import frame_queue
import cv2
import os

app_1 = Flask(__name__,
            template_folder='templates',
            static_folder='static')

@app_1.route('/')
def index():
    return render_template('index.html')

@app_1.route('/stream')
def stream():
    return Response(data_stream(), mimetype="text/event-stream")

@app_1.route('/face_image')
def face_image():
    """返回检测到人脸的单张图片"""
    if not frame_queue.empty():
        frame = frame_queue.get()
        ret, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    # 使用绝对路径加载默认图片
    default_image_path = os.path.join(app_1.static_folder, 'no_face.jpg')
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
    app_1.run(threaded=True, port=5000)