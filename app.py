import cv2
import numpy as np
import pickle
import base64
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ====================================================
# LOAD MODEL & LABELS
# ====================================================
model = load_model("models/gesture_model.h5")

with open("models/gesture_labels.pkl", "rb") as f:
    label_map = pickle.load(f)

# ====================================================
# GLOBAL STATE
# ====================================================
gesture_history = []
MAX_HISTORY = 20

current_gesture = {
    "gesture": "NONE",
    "action": "Waiting...",
    "confidence": 0,
    "timestamp": "--:--:--"
}

# 🔥 SESSION START TIME
session_start_time = datetime.now()

# 🔒 CONFIDENCE THRESHOLD (IMPORTANT)
CONFIDENCE_THRESHOLD = 85

# ====================================================
# PAGE ROUTES
# ====================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/live")
def live():
    with open("models/gesture_actions.pkl", "rb") as f:
        actions = pickle.load(f)
    return render_template("live_recognition.html", actions=actions)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/manage")
def manage():
    with open("models/gesture_actions.pkl", "rb") as f:
        actions = pickle.load(f)

    return render_template(
        "manage_gestures.html",
        actions=actions,
        gestures=list(actions.keys())
    )

@app.route("/about")
def about():
    return render_template("about.html")

# ====================================================
# PREDICTION API
# ====================================================
@app.post("/predict")
def predict():
    global current_gesture, gesture_history

    with open("models/gesture_actions.pkl", "rb") as f:
        action_map = pickle.load(f)

    data = request.json["image"]
    img_bytes = base64.b64decode(data.split(",")[1])
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(frame, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 1)

    prediction = model.predict(img, verbose=0)[0]
    class_id = int(np.argmax(prediction))
    confidence = round(float(np.max(prediction)) * 100, 2)

    gesture_name = "NONE"
    action_name = "No Gesture"

    if confidence >= CONFIDENCE_THRESHOLD:
        gesture_name = label_map.get(class_id, "UNKNOWN").upper().strip()
        action_name = action_map.get(gesture_name, "N/A")

    current_gesture = {
        "gesture": gesture_name,
        "action": action_name,
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%I:%M:%S %p")
    }

    if gesture_name != "NONE":
        gesture_history.append(current_gesture.copy())
        gesture_history[:] = gesture_history[-MAX_HISTORY:]

    return jsonify(current_gesture)

# ====================================================
# DATA APIs
# ====================================================
@app.route("/api/gesture_history")
def api_gesture_history():
    return jsonify({"history": gesture_history})

@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    global session_start_time
    gesture_history.clear()
    session_start_time = datetime.now()
    return jsonify({"message": "Gesture history & session reset"})

@app.route("/api/statistics")
def api_statistics():
    total = len(gesture_history)

    if total == 0:
        avg_accuracy = 0
        most_used = "-"
    else:
        avg_accuracy = round(
            sum(g["confidence"] for g in gesture_history) / total, 2
        )
        counts = {}
        for g in gesture_history:
            counts[g["gesture"]] = counts.get(g["gesture"], 0) + 1
        most_used = max(counts, key=counts.get)

    duration = datetime.now() - session_start_time
    minutes, seconds = divmod(duration.seconds, 60)
    session_time = f"{minutes}m {seconds}s"

    return jsonify({
        "total_gestures": total,
        "most_used": most_used,
        "avg_accuracy": avg_accuracy,
        "session_time": session_time
    })

# ====================================================
# MANAGE ACTIONS
# ====================================================
@app.route("/api/add_gesture", methods=["POST"])
def api_add_gesture():
    data = request.get_json(force=True)
    gesture = data.get("gesture", "").upper().strip()
    action = data.get("action", "").strip()

    if not gesture or not action:
        return jsonify(success=False, message="Gesture and action required")

    with open("models/gesture_actions.pkl", "rb") as f:
        actions = pickle.load(f)

    actions[gesture] = action

    with open("models/gesture_actions.pkl", "wb") as f:
        pickle.dump(actions, f)

    return jsonify(success=True, message="Action saved successfully")

@app.route("/api/delete_gesture", methods=["POST"])
def api_delete_gesture():
    data = request.get_json(force=True)
    gesture = data.get("gesture", "").upper().strip()

    with open("models/gesture_actions.pkl", "rb") as f:
        actions = pickle.load(f)

    normalized = {k.upper().strip(): k for k in actions.keys()}

    if gesture in normalized:
        del actions[normalized[gesture]]
        with open("models/gesture_actions.pkl", "wb") as f:
            pickle.dump(actions, f)
        return jsonify(success=True, message="Gesture action removed")

    return jsonify(success=False, message="Gesture action not found")

# ====================================================
# RUN (RENDER READY)
# ====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
