from flask import jsonify, render_template, Response, request, Blueprint
from data_integration.data_interface import video_queue, ad_queue
from queue import Empty
from util import get_resource_path
import json
import time

user_screen = Blueprint(
    "user_screen",  # Blueprint name (must be a string)
    __name__,  # Let Flask know which module this Blueprint belongs to
    template_folder=get_resource_path("data_integration/templates"),  # Specify the HTML template path
    static_folder=get_resource_path("static")  # Specify the static file path
)

# Global variable to store context
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
                queue_empty = video_queue.empty()  # Check if video_queue is empty
                if not queue_empty and not ad_queue.empty():
                    video = video_queue.get_nowait()
                    ad_text = ad_queue.get_nowait()
                    if video != last_video:
                        last_video = video
                        yield f"data: {json.dumps({'video': video, 'ad_text': ad_text, 'queue_empty': queue_empty})}\n\n"
                else:
                    # Send queue_empty status even if there is no new video
                    yield f"data: {json.dumps({'queue_empty': queue_empty})}\n\n"
            except Empty:
                pass
            time.sleep(1)
    return Response(event_stream(), mimetype="text/event-stream")

@user_screen.route('/video-ended', methods=['POST'])
def video_ended():
    data = request.json
    # video = data.get('video')
    ad_type = data.get('ad_type')  # New field: advertisement type
    # print(f"[Server] Received video ended notification for: {video}, type: {ad_type}")
    if context:
        if ad_type == 'default':
            context.default_video_completed.set()  # Set default advertisement end signal
        elif ad_type == 'personalized':
            context.personalized_video_completed.set()  # Set personalized advertisement end signal
    else:
        print("[Error] Failed passing context in user_screen server")
    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    user_screen.run(threaded=True, port=5001)