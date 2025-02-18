import cv2
from config import VIDEO_SOURCE, RTSP_STREAM

def start_video_stream(video_path=VIDEO_SOURCE, rtsp_stream=RTSP_STREAM):
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(f'appsrc ! videoconvert ! x264enc ! rtspclientsink location={rtsp_stream}', 
                          0, 30, (640, 480))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    start_video_stream()