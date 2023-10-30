#import tensorflow as tf
#print("TensorFlow version:", tf.__version__)
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('base.html')

@app.route('/process_coordinates', methods=['POST'])
def process_coordinates():
    data = request.get_json()  # Assuming data is sent as JSON
    coordinates = data['coordinates']
    print(f'User clicked at coordinates: {coordinates}')
    # You can process or store the coordinates as needed

    # Optionally, you can return a response to the client
    return jsonify({'message': 'Coordinates received'})


if __name__ == "__main__":
    app.run(debug=True)