from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time
import threading
from itertools import groupby
import hand_detection as hd
import gemini_predictions as gp

# ---- 1. Build the original CNN architecture ----
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(29, activation='softmax'))

# ---- 2. Load pre-trained weights ----
try:
    model.load_weights("CNN.h5")
    print("✅ Model architecture rebuilt and weights loaded successfully.")
except Exception as e:
    print("❌ Failed to load weights:", e)

# ---- 3. Define class labels ----
class_names = ["A", "B", "C", "D", "E", "F",
               "G", "H", "I", "J", "K", "L",
               "M", "N", "O", "P", "Q", "R",
               "S", "T", "U", "V", "W", "X",
               "Y", "Z", "del", "nothing", "space"]

# ---- 4. Initialize camera and hand detector ----
cap = cv2.VideoCapture(0)
detector = hd.handDetector()
samples_to_predict = []

# ---- Word buffer and Gemini prediction state ----
current_word = ""
last_committed_char = None
last_committed_count = 0
last_commit_time = 0.0
last_requested_prefix = None
current_predictions = []
predictions_lock = threading.Lock()
DEBOUNCE_MS = 0.3
CONSECUTIVE_FOR_COMMIT = 10


def listToString(s):
    """Convert list of strings into one concatenated string."""
    return "".join(s)


def commit_label(label: str):
    """Apply a confirmed label to current_word (letter, space, or del)."""
    global current_word, last_commit_time
    if label == "nothing":
        return
    if label == "space":
        current_word += " "
    elif label == "del":
        current_word = current_word[:-1]
    else:
        current_word += label
    last_commit_time = time.time()


def fetch_predictions_async(prefix: str):
    """Request word completions in a background thread; store top 3 for display."""
    global last_requested_prefix, current_predictions
    if len(prefix) < 2 or prefix == last_requested_prefix:
        return
    last_requested_prefix = prefix

    def task():
        words = gp.get_word_completions(prefix, top_k=5)
        with predictions_lock:
            current_predictions.clear()
            current_predictions.extend(words[:3])

    threading.Thread(target=task, daemon=True).start()


# ---- 5. Main loop ----
while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Failed to read from webcam.")
        break

    image = detector.findHands(img)
    landmark_list = detector.findPosition(img)

    # Define region of interest
    (startX, startY) = 50, 50
    (endX, endY) = 300, 300
    cv2.rectangle(img, (startX, startY), (endX, endY), 255, 4)

    cropped_video = img[50:300, 50:300]
    if len(landmark_list) != 0:
        if (startX <= landmark_list[0][1] <= endX) and (startY <= landmark_list[0][2] <= endY):
            image = cv2.resize(cropped_video, (64, 64))
            image = image.astype('float32') / 255.0
            x = img_to_array(image)
            x = np.expand_dims(image, axis=0)

            prediction = model.predict(x)
            label = class_names[np.argmax(prediction)]

            samples_to_predict.append(label)

            # ---- Confirmed letter: last run of CONSECUTIVE_FOR_COMMIT+ same label ----
            runs = [(k, len(list(g))) for k, g in groupby(samples_to_predict)]
            if runs:
                last_label, last_count = runs[-1]
                if last_count >= CONSECUTIVE_FOR_COMMIT:
                    if last_label != last_committed_char:
                        commit_label(last_label)
                        last_committed_char = last_label
                        if len(current_word) >= 2:
                            fetch_predictions_async(current_word)
                    last_committed_count = last_count

            string_labels = listToString(samples_to_predict)
            if len(string_labels) >= 10:
                import re
                consecutive = [match[1] for match in re.findall(r'((\w)\2{9,})', string_labels)]
                cv2.putText(img, format(consecutive), (90, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(img, label, (90, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    # ---- 300ms debounce: after pause, fetch predictions if prefix >= 2 ----
    if len(current_word) >= 2 and (time.time() - last_commit_time) >= DEBOUNCE_MS:
        if current_word != last_requested_prefix:
            fetch_predictions_async(current_word)

    # ---- Display current word and top 3 predictions ----
    cv2.putText(img, f"Word: {current_word or '(none)'}", (50, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    with predictions_lock:
        preds = list(current_predictions)
    for i, p in enumerate(preds[:3]):
        cv2.putText(img, f"{i + 1}. {p}", (50, 350 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
