"""
Person Appearance Tracker (face re-identification + visit logging)
====================================================================

v3: robust day/night re-identification via a multi-sample "gallery"
per person, and permanent, never-changing person IDs.

Why IDs were sometimes duplicated before
-----------------------------------------
A face embedding taken in bright daylight and one taken at night (or
under IR light) can look different enough numerically that a single
stored reference photo per person fails to match. That looked like
"the same person got a new ID" -- but the ID logic itself was already
correct (an ID is only ever created once, at registration, and is
never replaced). The fix is giving each person MULTIPLE reference
embeddings (day, night, different angles) so a new detection just
needs to match ANY one of them, not one single snapshot.

What this version does
------------------------
- Each person has exactly ONE permanent ID, created the first time
  they are ever seen. It is never regenerated or replaced. Every
  future visit -- day or night -- reuses that same ID; only the
  visit COUNT increases.
- Maintains a gallery of up to MAX_ENCODINGS_PER_PERSON face
  embeddings per person, captured across different visits/lighting,
  so matching is robust to day vs. night appearance changes.
- Applies CLAHE (adaptive contrast enhancement) to dark frames before
  detection/recognition, which meaningfully helps in low light --
  though it cannot manufacture detail from a frame with no light in
  it at all. If your camera has an IR/night mode, this pipeline works
  with it; if there is truly no light source, no software can detect
  a face because there is no image signal to work with.
- Everything else (visit color by visit count, per-visit elapsed
  seconds, SQLite logging by day/month/year/hour, --report) is
  unchanged from the previous version.

Requirements
------------
    pip install opencv-python numpy

Run
---
    python person_appearance_tracker.py
    python person_appearance_tracker.py --report
"""

import os
import sys
import sqlite3
import time
import urllib.request
import uuid
from datetime import datetime

import cv2
import numpy as np

# ------------------------- CONFIG -------------------------
CAMERA_INDEX = 0
DB_PATH = "appearances.db"
SESSION_GAP_SECONDS = 8.0          # how long someone can be out of frame before the visit is "over"
MAX_VISITS_FOR_FULL_GREEN = 10     # visit count at which color maxes out at green
MATCH_SIMILARITY_THRESHOLD = 0.363 # SFace cosine similarity threshold for "same person" (higher = same)
DETECTOR_SCORE_THRESHOLD = 0.75    # slightly relaxed vs. default to help catch low-light/IR faces
MIN_FACE_SIZE_FOR_GALLERY = 60     # px; only add an embedding to the gallery from a reasonably large/clear face
MAX_ENCODINGS_PER_PERSON = 20      # cap gallery size per person
NEW_SAMPLE_MIN_DISTINCTIVENESS = 0.85  # only add a new gallery sample if it's not near-identical to one already stored
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 90    # mean pixel brightness (0-255) below which CLAHE enhancement kicks in
MODEL_DIR = "models"
YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
# ------------------------------------------------------------


def download_models_if_missing():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for path, url in ((YUNET_PATH, YUNET_URL), (SFACE_PATH, SFACE_URL)):
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)} ...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"Could not download {url}: {e}")
                print(f"Download it manually and place it at: {os.path.abspath(path)}")
                sys.exit(1)


# ============================================================
# Low-light enhancement
# ============================================================

def enhance_if_dark(frame):
    """If the frame is dim, boost local contrast with CLAHE on the
    luminance channel. Helps a lot with dim-but-not-dark scenes and
    IR footage; cannot recover detail from a frame with no signal."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if brightness >= LOW_LIGHT_BRIGHTNESS_THRESHOLD:
        return frame, brightness

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced, brightness


# ============================================================
# Database
# ============================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            first_seen TEXT,
            total_visits INTEGER DEFAULT 0
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            encoding BLOB,
            added_at TEXT,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS visits (
            visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            start_time TEXT,
            end_time TEXT,
            duration_seconds REAL,
            visit_number INTEGER,
            color_hex TEXT,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            hour INTEGER
        )"""
    )
    # migrate old single-encoding schema if present
    cols = [r[1] for r in conn.execute("PRAGMA table_info(persons)")]
    if "encoding" in cols:
        rows = conn.execute("SELECT person_id, encoding FROM persons WHERE encoding IS NOT NULL")
        for person_id, enc_blob in rows.fetchall():
            existing = conn.execute(
                "SELECT COUNT(*) FROM encodings WHERE person_id = ?", (person_id,)
            ).fetchone()[0]
            if existing == 0 and enc_blob is not None:
                conn.execute(
                    "INSERT INTO encodings (person_id, encoding, added_at) VALUES (?, ?, ?)",
                    (person_id, enc_blob, datetime.now().isoformat()),
                )
        conn.commit()
    conn.commit()
    return conn


def load_gallery(conn):
    """Returns dict person_id -> list of embeddings (each shape (1,128) float32)."""
    gallery = {}
    for person_id, enc_blob in conn.execute("SELECT person_id, encoding FROM encodings"):
        gallery.setdefault(person_id, []).append(
            np.frombuffer(enc_blob, dtype=np.float32).reshape(1, -1)
        )
    return gallery


def register_new_person(conn, embedding):
    """Creates exactly one permanent ID for a never-before-seen person.
    This is the ONLY place a person_id is ever generated."""
    person_id = uuid.uuid4().hex[:10]
    conn.execute(
        "INSERT INTO persons (person_id, first_seen, total_visits) VALUES (?, ?, 0)",
        (person_id, datetime.now().isoformat()),
    )
    conn.execute(
        "INSERT INTO encodings (person_id, encoding, added_at) VALUES (?, ?, ?)",
        (person_id, embedding.astype(np.float32).tobytes(), datetime.now().isoformat()),
    )
    conn.commit()
    return person_id


def add_gallery_sample(conn, person_id, embedding):
    conn.execute(
        "INSERT INTO encodings (person_id, encoding, added_at) VALUES (?, ?, ?)",
        (person_id, embedding.astype(np.float32).tobytes(), datetime.now().isoformat()),
    )
    # trim oldest if over cap
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM encodings WHERE person_id = ? ORDER BY added_at", (person_id,)
    )]
    if len(ids) > MAX_ENCODINGS_PER_PERSON:
        for old_id in ids[: len(ids) - MAX_ENCODINGS_PER_PERSON]:
            conn.execute("DELETE FROM encodings WHERE id = ?", (old_id,))
    conn.commit()


def increment_visit_count(conn, person_id):
    conn.execute("UPDATE persons SET total_visits = total_visits + 1 WHERE person_id = ?", (person_id,))
    conn.commit()
    row = conn.execute("SELECT total_visits FROM persons WHERE person_id = ?", (person_id,)).fetchone()
    return row[0]


def log_visit(conn, person_id, start_dt, end_dt, duration, visit_number, color_bgr):
    color_hex = "#{:02x}{:02x}{:02x}".format(color_bgr[2], color_bgr[1], color_bgr[0])
    conn.execute(
        """INSERT INTO visits
           (person_id, start_time, end_time, duration_seconds, visit_number,
            color_hex, year, month, day, hour)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            person_id, start_dt.isoformat(), end_dt.isoformat(), round(duration, 2),
            visit_number, color_hex, start_dt.year, start_dt.month, start_dt.day, start_dt.hour,
        ),
    )
    conn.commit()


# ============================================================
# Color: red (1st visit) -> shades -> green (frequent visitor)
# ============================================================

def color_for_visit(visit_number, max_visits=MAX_VISITS_FOR_FULL_GREEN):
    t = min(max(visit_number - 1, 0) / max(max_visits - 1, 1), 1.0)
    hue = int(t * 60)
    value = int(140 + 90 * np.sin(t * np.pi))
    hsv_pixel = np.uint8([[[hue, 255, value]]])
    bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ============================================================
# Matching against the whole gallery
# ============================================================

def best_match(recognizer, embedding, gallery):
    """Compares embedding against every stored sample of every known
    person; returns (person_id_or_None, best_score)."""
    best_person, best_score = None, -1.0
    for person_id, samples in gallery.items():
        for sample in samples:
            score = recognizer.match(embedding, sample, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                if score >= MATCH_SIMILARITY_THRESHOLD:
                    best_person = person_id
    return best_person, best_score


# ============================================================
# Main tracking loop
# ============================================================

def run_tracker():
    download_models_if_missing()

    detector = cv2.FaceDetectorYN.create(
        YUNET_PATH, "", (320, 320), DETECTOR_SCORE_THRESHOLD, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

    conn = get_db()
    gallery = load_gallery(conn)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    detector.setInputSize((w, h))

    active_sessions = {}  # person_id -> dict
    print("Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()

        detect_frame, brightness = enhance_if_dark(frame)
        _, faces = detector.detect(detect_frame)
        seen_this_frame = set()

        if faces is not None:
            for face in faces:
                x, y, fw, fh = face[:4].astype(int)
                aligned = recognizer.alignCrop(detect_frame, face)
                embedding = recognizer.feature(aligned)

                person_id, score = best_match(recognizer, embedding, gallery)

                if person_id is None:
                    # brand-new person: exactly one ID created, forever
                    person_id = register_new_person(conn, embedding)
                    gallery[person_id] = [embedding]
                elif min(fw, fh) >= MIN_FACE_SIZE_FOR_GALLERY and score < NEW_SAMPLE_MIN_DISTINCTIVENESS:
                    # good clear match, but a meaningfully different sample
                    # (different lighting/angle) -- add it to the gallery so
                    # future day/night recognition gets stronger over time
                    add_gallery_sample(conn, person_id, embedding)
                    gallery[person_id].append(embedding)

                seen_this_frame.add(person_id)

                if person_id not in active_sessions:
                    visit_number = increment_visit_count(conn, person_id)
                    color = color_for_visit(visit_number)
                    active_sessions[person_id] = {
                        "first_seen": now,
                        "first_seen_dt": datetime.now(),
                        "last_seen": now,
                        "visit_number": visit_number,
                        "color": color,
                    }
                else:
                    active_sessions[person_id]["last_seen"] = now

                sess = active_sessions[person_id]
                elapsed = now - sess["first_seen"]
                color = sess["color"]

                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 3)
                label = f"ID {person_id} | visit #{sess['visit_number']} | {elapsed:.1f}s"
                cv2.putText(frame, label, (x, max(y - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for person_id in list(active_sessions.keys()):
            sess = active_sessions[person_id]
            if person_id not in seen_this_frame and (now - sess["last_seen"]) > SESSION_GAP_SECONDS:
                duration = sess["last_seen"] - sess["first_seen"]
                log_visit(conn, person_id, sess["first_seen_dt"], datetime.now(),
                          duration, sess["visit_number"], sess["color"])
                del active_sessions[person_id]

        cv2.putText(frame, f"Known people so far: {len(gallery)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Currently in frame: {len(seen_this_frame)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Scene brightness: {brightness:.0f}/255",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Person Appearance Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    now = time.time()
    for person_id, sess in active_sessions.items():
        duration = sess["last_seen"] - sess["first_seen"]
        log_visit(conn, person_id, sess["first_seen_dt"], datetime.now(),
                  duration, sess["visit_number"], sess["color"])

    cap.release()
    cv2.destroyAllWindows()
    conn.close()


# ============================================================
# Reporting
# ============================================================

def print_report():
    conn = get_db()
    print("\n=== People ===")
    for person_id, first_seen, total_visits in conn.execute(
        "SELECT person_id, first_seen, total_visits FROM persons ORDER BY first_seen"
    ):
        n_samples = conn.execute(
            "SELECT COUNT(*) FROM encodings WHERE person_id = ?", (person_id,)
        ).fetchone()[0]
        print(f"  {person_id}  first seen: {first_seen}  total visits: {total_visits}  gallery samples: {n_samples}")

    print("\n=== Visits ===")
    for row in conn.execute(
        """SELECT person_id, start_time, duration_seconds, visit_number, color_hex,
                  year, month, day, hour FROM visits ORDER BY start_time"""
    ):
        pid, start, dur, vnum, color, y, m, d, h = row
        print(f"  [{start}] {pid}  visit #{vnum}  {dur}s  color={color}  ({y}-{m:02d}-{d:02d} {h:02d}:00)")

    print("\n=== Visits per day ===")
    for date, cnt in conn.execute(
        "SELECT date(start_time), COUNT(*) FROM visits GROUP BY date(start_time) ORDER BY date(start_time)"
    ):
        print(f"  {date}: {cnt} visit(s)")

    print("\n=== Visits per month ===")
    for y, m, cnt in conn.execute(
        "SELECT year, month, COUNT(*) FROM visits GROUP BY year, month ORDER BY year, month"
    ):
        print(f"  {y}-{m:02d}: {cnt} visit(s)")

    print("\n=== Visits per year ===")
    for y, cnt in conn.execute("SELECT year, COUNT(*) FROM visits GROUP BY year ORDER BY year"):
        print(f"  {y}: {cnt} visit(s)")

    print("\n=== Busiest hours (across all days) ===")
    for h, cnt in conn.execute("SELECT hour, COUNT(*) FROM visits GROUP BY hour ORDER BY hour"):
        print(f"  {h:02d}:00 - {cnt} visit(s)")

    conn.close()


if __name__ == "__main__":
    if "--report" in sys.argv:
        print_report()
    else:
        run_tracker()