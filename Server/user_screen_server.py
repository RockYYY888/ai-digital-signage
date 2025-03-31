# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

from flask import jsonify, render_template, Response, request, Blueprint
from Server.data_interface import video_queue, ad_queue
from queue import Empty
from util import get_resource_path
import json
import time

user_screen = Blueprint(
    "user_screen",  # Blueprint name (must be a string)
    __name__,  # Let Flask know which module this Blueprint belongs to
    template_folder=get_resource_path("Server/templates"),  # Specify the HTML template path
    static_folder=get_resource_path("static")  # Specify the static file path
)

# Global variable to store context
context = None

def set_context(ctx):
    """Set the global context for the user screen.

    Args:
        ctx: The context object containing shared state.
    """
    global context
    context = ctx

@user_screen.route('/')
def index():
    """Render the main screen HTML template."""
    return render_template('main_screen.html', video_name="")

@user_screen.route('/focus', methods=['POST'])
def focus():
    """Handle focus events for the main screen.

    This function sets or clears the focus event in the shared context.
    
    Returns:
        A JSON response indicating success.
    """
    data = request.json
    focus = data.get('focus')
    if context:
        if focus:
            context.user_screen_focus.set()
            print("[Main Screen] Main screen gained focus.")
        else:
            context.user_screen_focus.clear()
            print("[Main Screen] Main screen lost focus.")
    else:
        print("[Error] Context not set in user_screen server.")
    return jsonify({"status": "success"}), 200

@user_screen.route('/stream')
def stream():
    """Provide a server-sent event stream of video and advertisement data.

    This function continuously checks the video and advertisement queues
    and streams updates to the client.

    Returns:
        A streaming HTTP response with event-stream mimetype.
    """
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
    """Handle video completion events.

    This function receives a notification when a video ends and sets
    the appropriate event in the shared context.

    Returns:
        A JSON response indicating success.
    """
    data = request.json
    ad_type = data.get('ad_type')  # New field: advertisement type
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
