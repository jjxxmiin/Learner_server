import os
import flask
import werkzeug
import requests
from pathlib import Path

datasets_path = Path("train_datasets")

app = flask.Flask(__name__)


@app.route('/train', methods = ['GET', 'POST'])
def handle_train():
    imagefile = flask.request.files['image']
    label = flask.request.form['label']

    filename = imagefile.filename

    # filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + filename)

    save_path = datasets_path / label

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imagefile.save(save_path / filename)

    return "Image Uploaded Successfully"


@app.route('/infer', methods = ['GET', 'POST'])
def handle_infer():
    imagefile = flask.request.files['image']

    return "정재민"


app.run(host="0.0.0.0", port=5000, debug=True)