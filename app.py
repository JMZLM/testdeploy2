from flask import Flask, redirect, url_for, request, session, render_template, jsonify
import requests
import cv2
import os
import base64
import numpy as np
import gdown
from urllib.parse import urlencode
from ultralytics import YOLO
import threading
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Spotify API credentials
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')

# Spotify URLs
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"

# Google Drive file ID of your YOLO model (you can get this from the shareable link)
MODEL_URL = "https://drive.google.com/uc?id=15T5uc8iMm5Fs8XQIHaRy8W6LYQrVErCQ"

# Function to download YOLO model weights from Google Drive
def download_model():
    output_path = "/opt/render/project/Yolo-Weights"  # Absolute path in Render's environment
    os.makedirs(output_path, exist_ok=True)  # Ensure the directory exists
    gdown.download(MODEL_URL, os.path.join(output_path, "best.pt"), quiet=False)

# Modify your YOLO initialization to download the model if not already present
if not os.path.exists("../Yolo-Weights"):
    print("Downloading YOLO model...")
    download_model()

# Initialize YOLO model
model = YOLO("/opt/render/project/Yolo-Weights/best.pt")
classNames = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Global variables
detected_emotion = None
emotion_songs = []
current_song_index = 0
is_paused = False

# Fetch songs based on emotion
def fetch_songs_for_emotion(emotion, access_token):
    headers = {'Authorization': f"Bearer {access_token}"}
    params = {'q': emotion, 'type': 'track', 'limit': 50}
    response = requests.get(f"{SPOTIFY_API_BASE_URL}/search", headers=headers, params=params)

    if response.status_code == 200:
        tracks = response.json().get('tracks', {}).get('items', [])
        return [{'id': track['id'], 'name': track['name'], 'artist': track['artists'][0]['name'],
                 'album': track['album']['name'],
                 'cover_url': track['album']['images'][0]['url'] if track['album']['images'] else ''}
                for track in tracks]
    return []

# Function to play a song on the active device
def play_song(song, access_token):
    headers = {'Authorization': f"Bearer {access_token}"}

    # Get available devices
    devices_response = requests.get(f"{SPOTIFY_API_BASE_URL}/me/player/devices", headers=headers)
    devices = devices_response.json().get('devices', [])
    if not devices:
        print("No active devices found.")
        return

    device_id = devices[0]['id']  # Use the first available device
    track_uri = f"spotify:track:{song['id']}"
    play_url = f"{SPOTIFY_API_BASE_URL}/me/player/play?device_id={device_id}"

    # Start playback
    requests.put(play_url, headers=headers, json={"uris": [track_uri]})
    print(f"Playing: {song['name']} by {song['artist']}")

# Spotify login route
@app.route('/login_spotify')
def login_spotify():
    if 'access_token' in session:
        return redirect(url_for('home'))

    params = {
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': REDIRECT_URI,
        'scope': 'user-read-playback-state user-modify-playback-state user-read-private user-read-email streaming',
    }
    url = f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"
    return redirect(url)

# Spotify OAuth callback
@app.route('/callback')
def callback():
    code = request.args.get('code')
    if code:
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': REDIRECT_URI,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
        }
        response = requests.post(SPOTIFY_TOKEN_URL, data=data)
        token_info = response.json()

        if 'access_token' in token_info:
            session['access_token'] = token_info['access_token']
            session['refresh_token'] = token_info.get('refresh_token')
            return redirect(url_for('home'))
    return "Error: Authorization failed."

# Home route
@app.route('/')
def home():
    if 'access_token' not in session:
        return redirect(url_for('login_spotify'))

    return render_template('index.html', access_token=session['access_token'])

# Detect emotion from webcam frame
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    global detected_emotion
    data = request.json
    frame_data = data.get('frame')

    if not frame_data:
        return jsonify({'error': 'No frame data received'}), 400

    # Decode the image
    _, encoded_image = frame_data.split(',')
    image_data = base64.b64decode(encoded_image)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLO model
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            detected_emotion = classNames[cls]
            break

    return jsonify({'detected_emotion': detected_emotion})

# Start Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Default to port 8080 if PORT is not set
    app.run(debug=True, host='0.0.0.0', port=port)
