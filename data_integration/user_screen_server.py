from flask import Flask, render_template

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

@app.route('/')
def index():
    return render_template('main_screen.html')

if __name__ == "__main__":
    app.run(threaded=True, port=5001)  # Run on port 5001