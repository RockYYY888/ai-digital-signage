from flask import Flask, jsonify, render_template
from data_interface import product_queue

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

@app.route('/')
def index():
    return render_template('main_screen.html')

@app.route('/get_product_queue', methods=['GET'])
def get_product_queue():
    """Return the contents of the product_queue as a list."""
    if not product_queue.empty():  # Check if the queue is empty
        product = product_queue.get()  # Retrieve the product name from the queue
        return jsonify({"product": product})
    else:
        return jsonify({"product": None})  # If the queue is empty, return None

if __name__ == "__main__":
    app.run(threaded=True, port=5001)  # Run on port 5001