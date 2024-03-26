import cv2
import json
import time

# Models
FACE_PROTO = "face_detector/opencv_face_detector.pbtxt"
FACE_MODEL = "face_detector/opencv_face_detector_uint8.pb"
AGE_PROTO = "age/age_deploy.prototxt"
AGE_MODEL = "age/age_net.caffemodel"
GENDER_PROTO = "gender/gender_deploy.prototxt"
GENDER_MODEL = "gender/gender_net.caffemodel"

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST = ["Male", "Female"]
CONFIDENCE_THRESHOLD = 0.7
PADDING = 20
GREEN = (0, 255, 0)

def load_models():
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return face_net, age_net, gender_net

def detect_faces(face_net, frame):
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    bboxs = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxs.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 1)
    return bboxs

def detect_age_gender(face_net, age_net, gender_net, frame, bboxs):
    data = []
    for bbox in bboxs:
        face = frame[max(0, bbox[1] - PADDING):min(bbox[3] + PADDING, frame.shape[0] - 1),
                     max(0, bbox[0] - PADDING):min(bbox[2] + PADDING, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_prediction = gender_net.forward()
        gender = GENDER_LIST[gender_prediction[0].argmax()]
        age_net.setInput(blob)
        age_prediction = age_net.forward()
        age = AGE_LIST[age_prediction[0].argmax()]
        timestamp = int(time.time())
        data.append({"gender": gender, "age": age, "timestamp": timestamp})
    return data

def main():
    face_net, age_net, gender_net = load_models()
    video = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error reading from camera")
            break
        
        bboxs = detect_faces(face_net, frame)
        data = detect_age_gender(face_net, age_net, gender_net, frame, bboxs)
        
        for bbox, info in zip(bboxs, data):
            x1, y1, x2, y2 = bbox
            gender, age = info["gender"], info["age"]
            label = "{},{}".format(gender, age)
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), GREEN, -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        json_data = json.dumps(data, indent=4)
        print(json_data)
        
        cv2.imshow("Identiface", frame)
        key = cv2.waitKey(1)
        
        if key == ord('q') or cv2.getWindowProperty("Identiface", cv2.WND_PROP_VISIBLE) < 1:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
