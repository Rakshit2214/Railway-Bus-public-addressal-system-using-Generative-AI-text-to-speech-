

from flask import Flask, Response, send_file

app = Flask(__name__)

@app.route('/')
def video_feed():
    video_path = 'videos\\train.mp4'  # Replace with the path to your MP4 file
    return send_file(video_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
