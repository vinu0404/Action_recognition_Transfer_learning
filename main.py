import sys
from config import VIDEO_SOURCE
from streaming import start_video_stream
from inference import perform_inference
from visualization import app as flask_app
from training import train_model

def main(video_file=None, train=False):
    if video_file:
        config.VIDEO_SOURCE = video_file
    
    if train:
        model, history = train_model()
        print("Model trained and saved.")
    
    # Start streaming
    import threading
    stream_thread = threading.Thread(target=start_video_stream)
    stream_thread.start()

    # Start inference
    inference_thread = threading.Thread(target=perform_inference)
    inference_thread.start()

    # Run Flask app in main thread
    flask_app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    train = '--train' in sys.argv
    video_file = next((arg for arg in sys.argv if arg.endswith('.avi')), None)
    main(video_file, train)