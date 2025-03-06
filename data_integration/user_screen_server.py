
from flask import Flask, jsonify, render_template, Response
from data_integration.data_interface import video_queue, ad_queue  
from queue import Empty
from pathlib import Path
import json
import time

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')


@app.route('/')
def index():
    # 渲染主页面，初始视频名为空
    return render_template('main_screen.html', video_name="")


@app.route('/stream')
def stream():
    def event_stream():
        last_video = None
        while True:
            try:
                # 只有当两个队列都不为空时才获取数据并推送
                if not video_queue.empty() and not ad_queue.empty():
                    video = video_queue.get_nowait()  # 非阻塞获取 video
                    ad_text = ad_queue.get_nowait()  # 非阻塞获取 ad_text
                    if video != last_video:  # 避免重复推送相同视频
                        last_video = video
                        yield f"data: {json.dumps({'video': video, 'ad_text': ad_text})}\n\n"
            except Empty:
                pass  # 队列为空时跳过
            time.sleep(1)  # 每次循环休眠 1 秒
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(threaded=True, port=5001)  # Run on port 5001
