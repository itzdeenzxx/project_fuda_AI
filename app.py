import cv2
from flask import Flask, render_template, Response , request , jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

model = load_model('static/model.h5')

label_map = ['โกรธ', 'ปกติ', 'กลัว', 'มีความสุข', 'เศร้า', 'ประหลาดใจ']
predict = ""
acc = 0

pause_cam = False
cam = False
cam_model = False

@app.route('/gen_frames', methods=['POST'])
def gen_frames():
    global predict , acc
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
            faces = cascade.detectMultiScale(gray, 1.1, 3)
            if len(faces) == 0:
                predict = "ไม่พบใบหน้า"
                acc = 0
            else:
                for x, y, w, h in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cropped = frame[y:y + h, x:x + w]
                try:
                    img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                except:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                img = cv2.resize(img, (48, 48))
                img = img / 255
                img = img.reshape(1, 48, 48, 1)
                pred = model.predict(img)
                pred = np.argmax(pred)
                final_pred = label_map[pred]
                predict = final_pred
                
                accuracy = np.max(model.predict(img)) * 100
                print("Accuracy:", accuracy)
                acc = accuracy

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/get_prediction')
def get_prediction():
    
    pred = predict
    print(pred)
    return Response(pred, mimetype='text/plain')

@app.route('/get_accuracy')
def get_accuracy():
    accuracy = round(acc, 2)
    print(accuracy)
    return Response(str(accuracy), mimetype='text/plain')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=8080)
