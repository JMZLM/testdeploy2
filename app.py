from flask import Flask, redirect, url_for, request, session, render_template, jsonify
import requests
import threading
import cv2
import os
from ultralytics import YOLO
from urllib.parse import urlencode

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# MBTI Results
mbti_results = {
    "INTJ": "The Architect",
    "INFP": "The Mediator",
    "ENTJ": "The Commander",
    "ENFP": "The Campaigner",
    "ISTJ": "The Logistician",
    "ISFJ": "The Defender",
    "ESTJ": "The Executive",
    "ESFJ": "The Consul",
    "INTP": "The Logician",
    "INFJ": "The Advocate",
    "ENTP": "The Debater",
    "ENFJ": "The Protagonist",
    "ISFP": "The Adventurer",
    "ISTP": "The Virtuoso",
    "ESTP": "The Entrepreneur",
    "ESFP": "The Entertainer"
}

# Spotify API credentials
CLIENT_ID = 'fdfcbb82df104a2ea67df35410eed1f6'
CLIENT_SECRET = '3280a9e0b5804025bcff0d5367eec8eb'
REDIRECT_URI = 'http://localhost:8080/callback'

# Spotify URLs
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"

# Initialize YOLO model
model = YOLO("../Yolo-Weights/best.pt")
classNames = ["anger", "disgust", "fear", "happy", "neutral", "sad", "neutral"]

# Global variables
detected_emotion = None
emotion_songs = []
current_song_index = 0
is_paused = False

# Home, Quiz, Result Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/result')
def result():
    personality_type = request.args.get('personality_type', '')
    description = mbti_results.get(personality_type, "Unknown type")
    return render_template('result.html', personality_type=personality_type, description=description)

# Emotion detection function
def run_emotion_detection(access_token):
    global detected_emotion, emotion_songs, current_song_index
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                detected_emotion = classNames[cls]
                cap.release()
                cv2.destroyAllWindows()
                emotion_songs = fetch_songs_for_emotion(detected_emotion, access_token)
                current_song_index = 0
                play_song(emotion_songs[current_song_index], access_token)  # Play the first song
                return

# Fetch songs based on emotion
def fetch_songs_for_emotion(emotion, access_token):
    headers = {'Authorization': f"Bearer {access_token}"}
    params = {'q': emotion, 'type': 'track', 'limit': 50}
    response = requests.get(f"{SPOTIFY_API_BASE_URL}/search", headers=headers, params=params)

    if response.status_code == 200:
        tracks = response.json().get('tracks', {}).get('items', [])
        return [{'id': track['id'], 'name': track['name'], 'artist': track['artists'][0]['name'],
                 'album': track['album']['name'],
                 'cover_url': track['album']['images'][0]['url'] if track['album']['images'] else ''
                 } for track in tracks]
    return []

# Function to play a song on the active device
def play_song(song, access_token):
    headers = {'Authorization': f"Bearer {access_token}"}
    devices_response = requests.get(f"{SPOTIFY_API_BASE_URL}/me/player/devices", headers=headers)
    devices = devices_response.json().get('devices', [])
    if not devices:
        print("No active devices found.")
        return

    device_id = devices[0]['id']  # Use the first available device
    track_uri = f"spotify:track:{song['id']}"
    play_url = f"{SPOTIFY_API_BASE_URL}/me/player/play?device_id={device_id}"

    # Activate the device
    requests.put(f"{SPOTIFY_API_BASE_URL}/me/player", headers=headers, json={"device_ids": [device_id]})

    # Start playback
    start_playback_response = requests.put(play_url, headers=headers, json={"uris": [track_uri]})
    if start_playback_response.status_code == 204:
        print(f"Playing: {song['name']} by {song['artist']} on device {devices[0]['name']}")
    else:
        print(f"Failed to play song: {start_playback_response.text}")

# Spotify login route
@app.route('/login_spotify')
def login_spotify():
    if 'access_token' in session:
        return redirect(url_for('spotify'))

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
            return redirect(url_for('spotify'))
    return "Error: Authorization failed."

# Spotify home route
@app.route('/spotify')
def spotify():
    if 'access_token' not in session:
        return redirect(url_for('login_spotify'))

    return render_template('spotify.html', access_token=session['access_token'])

# Start emotion detection
@app.route('/detect_emotion')
def detect_emotion():
    if 'access_token' not in session:
        return redirect(url_for('login_spotify'))

    access_token = session['access_token']
    threading.Thread(target=run_emotion_detection, args=(access_token,)).start()
    return redirect(url_for('spotify'))

# Control song playback
@app.route('/control/<action>')
def control(action):
    global current_song_index, is_paused

    if not emotion_songs:
        return jsonify({'error': 'No songs available to control'})

    access_token = session.get('access_token')
    if not access_token:
        return jsonify({'error': 'Access token is missing'})

    headers = {'Authorization': f'Bearer {access_token}'}

    if action == 'playpause':
        if is_paused:
            requests.put(f"{SPOTIFY_API_BASE_URL}/me/player/play", headers=headers)
            is_paused = False
        else:
            requests.put(f"{SPOTIFY_API_BASE_URL}/me/player/pause", headers=headers)
            is_paused = True
    elif action == 'next':
        current_song_index = (current_song_index + 1) % len(emotion_songs)
        play_song(emotion_songs[current_song_index], access_token)
    elif action == 'previous':
        current_song_index = (current_song_index - 1) % len(emotion_songs)
        play_song(emotion_songs[current_song_index], access_token)

    song = emotion_songs[current_song_index]
    return jsonify({
        'song': {
            'name': song['name'],
            'artist': song['artist'],
            'album': song['album'],
            'cover_url': song.get('cover_url', '')
        },
        'is_paused': is_paused
    })

@app.route('/get_detected_emotion')
def get_detected_emotion():
    return jsonify({'detected_emotion': detected_emotion})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Default to port 8080 if PORT is not set
    app.run(debug=False, host='0.0.0.0', port=port)



