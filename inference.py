import numpy as np
from tensorflow.keras.models import load_model
import cv2
from config import RTSP_STREAM, RESULTS_STREAM, MODEL_PATH, CLASSES

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Adjust according to your model's input size
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def perform_inference(video_stream=RTSP_STREAM, results_stream=RESULTS_STREAM):
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(video_stream)
    out = cv2.VideoWriter(f'appsrc ! videoconvert ! x264enc ! rtspclientsink location={results_stream}', 
                          0, 30, (640, 480))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)
        class_index = np.argmax(prediction[0])
        confidence = prediction[0][class_index]
        
        cv2.putText(frame, f'Action: {CLASSES[class_index]} Confidence: {confidence:.2f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    perform_inference()