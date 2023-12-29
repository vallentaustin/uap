import os
import cv2
import time
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template('select.html')

def load(filename):
    img = image.load_img(filename, target_size=(224, 224, 3))
    np_image = image.img_to_array(img)
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('EffNetmod5.h5')
    file = request.files["file"]
    file.save(os.path.join('static','temp.jpg'))
    img = load('static/temp.jpg')
    start = time.time()
    pred = model.predict(img)
    labels = np.argmax(pred, axis=-1)
    print(labels)
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred.flatten()]
    return predict_result(model, runtimes, respon_model, 'temp.jpg')

def predict_result(model, run_time, probs, img):
    class_list = {'Rock': 0, 'Paper': 1, 'Scissor':2}
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('result_select.html', labels=labels,
                            probs=probs, model=model, pred=idx_pred,
                            run_time=run_time, img=img)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=2000)