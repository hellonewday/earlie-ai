from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import cv2
import os
import io
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/face", methods=['POST'])
def text_to_speech_utils():
    img = request.files['image_data']
    if img and allowed_file(img.filename):
        filename = secure_filename(img.filename)
        file_path = os.path.join("tmp", filename)
        img.save(file_path)
        print(file_path)
        image = cv2.imread(file_path)
        KEY = "6e3b7b7c6aa547aea0ca39c601e2d4d6"
        ENDPOINT = "https://earlie-face.cognitiveservices.azure.com/"
        face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
        face_attributes = ['age', 'gender']
        ret, buf = cv2.imencode('.jpg', image)
        stream = io.BytesIO(buf)
        detected_faces = face_client.face.detect_with_stream(stream, return_face_id=True,
                                                             return_face_attributes=face_attributes)
        if len(detected_faces) == 0:
            return jsonify({
                "message": "Can't verfiy face!"
            })
        return jsonify({
            'gender': detected_faces[0].face_attributes.gender,
            'age': detected_faces[0].face_attributes.age
        })
    else:
        return jsonify({
            'success': False,
            'message': "Wrong type format"
        })
