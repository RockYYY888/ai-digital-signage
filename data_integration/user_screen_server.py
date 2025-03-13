from flask import Flask, jsonify, render_template, Response, request, Blueprint
from data_integration.data_interface import video_queue, ad_queue  
from queue import Empty
from pathlib import Path
import json
import time

user_screen = Blueprint(
    "user_screen",  # Blueprint 名称（必须是字符串）
    __name__,  # 让 Flask 知道这个 Blueprint 属于哪个模块
    template_folder="templates",  # 指定 HTML 模板路径
    static_folder="static"  # 指定静态文件路径
)

# 全局变量存储 context
context = None

def set_context(ctx):
    global context
    context = ctx

@user_screen.route('/')
def index():
    return render_template('main_screen.html', video_name="")

@user_screen.route('/stream')
def stream():
    def event_stream():
        last_video = None
        while True:
            try:
                if not video_queue.empty() and not ad_queue.empty():
                    video = video_queue.get_nowait()
                    ad_text = ad_queue.get_nowait()
                    if video != last_video:
                        last_video = video
                        yield f"data: {json.dumps({'video': video, 'ad_text': ad_text})}\n\n"
                elif not video_queue.empty() and ad_queue.empty():
                    queue_status = {
                        'video_size': video_queue.qsize(),
                        'ad_size': ad_queue.qsize()
                    }
                    yield f"data: {json.dumps({'queue_status': queue_status})}\n\n"
            except Empty:
                pass
            time.sleep(1)
    return Response(event_stream(), mimetype="text/event-stream")

@user_screen.route('/video-ended', methods=['POST'])
def video_ended():
    data = request.json
    video = data.get('video')
    print(f"[Server] Received video ended notification for: {video}")
    if context:
        context.video_completed.set()  # 设置事件，通知视频播放已完成
    else:
        print("[Error] Failed passing context in user_screen server")
    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    user_screen.run(threaded=True, port=5001)