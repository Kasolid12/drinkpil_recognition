from flask import Flask, request, jsonify
import pose_media as pm
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# Load your trained model
new_model = tf.keras.models.load_model('minum_obat_new.h5')  # Load the pre-trained model
pose = pm.mediapipe_pose()  # Initialize pose Mediapipe
pt = pose.mp_holistic.Holistic()  # Use holistic for pose detection

threshold = 0.5  # Threshold to determine action
actions = np.array(["DRINKPIL", "NOACT"])  # Action labels

# Flask route to process video input and return the counter
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    # Save the uploaded video file
    video_file = request.files['file']
    video_path = os.path.join('uploaded_videos', video_file.filename)
    video_file.save(video_path)

    # Initialize counter and sequence for action detection
    counter = 0  # Counter for the "DRINKPIL" action
    sequence = []
    sentence = []

    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return jsonify({'error': 'Error: Video cannot be accessed'}), 500

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame, results = pose.mediapipe_detection(frame, pt)
        except Exception as e:
            return jsonify({'error': f'Mediapipe detection error: {e}'}), 500

        keypoints = pose.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-10:]  # Store only the last 10 frames

        if len(sequence) == 10:
            res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                        if actions[np.argmax(res)] == "DRINKPIL":
                            counter += 1  # Increment counter if "DRINKPIL" action is detected
                else:
                    sentence.append(actions[np.argmax(res)])
                    if actions[np.argmax(res)] == "DRINKPIL":
                        counter += 1  # Increment counter if "DRINKPIL" action is detected
            if len(sentence) > 1:
                sentence = sentence[-1:]

        if counter == 3:  # Break if counter reaches 3
            break

    cap.release()
    os.remove(video_path)  # Optional: Remove the uploaded video after processing

    # Return the counter as JSON response
    return jsonify({'counter': counter})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8011, debug=True)
