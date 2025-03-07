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
    return render_template('main_screen.html', video_name="")

@app.route('/stream')
def stream():
    def event_stream():
        last_video = None
        while True:
            try:
                # 当两个队列都不为空时，推送视频和广告
                if not video_queue.empty() and not ad_queue.empty():
                    video = video_queue.get_nowait()
                    ad_text = ad_queue.get_nowait()
                    if video != last_video:
                        last_video = video
                        yield f"data: {json.dumps({'video': video, 'ad_text': ad_text})}\n\n"
                # 当 video_queue 不为空且 ad_queue 为空时，推送队列状态
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

if __name__ == "__main__":
    app.run(threaded=True, port=5001)