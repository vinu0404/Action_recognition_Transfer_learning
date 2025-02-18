from flask import Flask, Response
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return "Real-Time Video Analytics"

@app.route('/graph')
def graph():
    plt.plot([1, 2, 3], [4, 5, 6])
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return '<img src="data:image/png;base64,{}">'.format(plot_url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)