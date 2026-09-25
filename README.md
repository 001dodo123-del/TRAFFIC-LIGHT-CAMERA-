# Person Appearance Tracker

A camera-based system that recognizes people by face, gives each person a
**permanent ID** the first time they're seen, and tracks every time they
come back — logging how long each visit lasted and coloring their on-screen
box from **red (1st visit)** toward **green (frequent visitor)** as their
visit count grows.

---

## What it does

- **Detects faces** in a live camera feed (OpenCV's YuNet detector).
- **Recognizes returning people** using face embeddings (OpenCV's SFace
  model) compared against a stored gallery — so the same person is
  recognized whether they last visited an hour ago or a month ago.
- **Assigns one permanent ID per person**, created exactly once, the first
  time they're ever seen. That ID is never replaced or regenerated —
  every later visit just increments that person's visit count.
- **Colors the bounding box by visit count, not time.** 1st-ever visit is
  red; each additional visit shifts a step further toward green. The
  color stays fixed for the whole time someone is in frame during one
  visit — it only changes between separate visits.
- **Shows live elapsed seconds** for the person's current, in-progress
  visit.
- **Logs every completed visit** to a local SQLite database
  (`appearances.db`): person ID, start/end time, duration in seconds,
  visit number, color, and the year/month/day/hour it happened.
- **Handles low light / night footage** by enhancing dim frames (CLAHE
  contrast boost) before detection, and builds a multi-sample "gallery"
  of each person's face over time so day and night appearances of the
  same person are both recognized.
- **Prints a report** (`--report`) of every person, every visit, and
  breakdowns by day, month, year, and hour.

---

## Requirements

- Python 3.8+
- `opencv-python`
- `numpy`

```bash
pip install -r requirements.txt
```

No compiler, CMake, or dlib required — everything runs on plain OpenCV
with prebuilt wheels.

On first run, the script automatically downloads two small model files
(a face detector and a face recognizer, ~10 MB total) into a local
`models/` folder. If your machine has no internet access at runtime,
it will print the exact URLs and the local path to place them manually.

---

## Usage

**Start live tracking:**

```bash
python person_appearance_tracker.py
```

A window opens showing your camera feed with a colored box around each
detected face, labeled with their ID, visit number, and elapsed seconds.
Press **`q`** to quit.

**View the visit history / stats:**

```bash
python person_appearance_tracker.py --report
```

Prints every known person (with first-seen date and total visit count),
every logged visit, and counts grouped by day, month, year, and hour.

---

## How recognition and ID permanence work

1. The first time a face is seen, it gets a new, permanent, randomly
   generated ID (a short hex string) and is registered in the database.
   This is the *only* place in the code that ever creates an ID.
2. On every later detection, the new face is compared against a
   **gallery** of previously stored face samples for every known person.
   If it matches an existing person closely enough, that person's
   existing ID is reused and their visit count goes up by one — the ID
   itself is never changed.
3. To stay accurate across different lighting (daylight, evening, IR/
   night mode), each person accumulates up to 20 reference face samples
   over time, captured automatically from their visits. A new detection
   only needs to match *any one* of those samples, not a single fixed
   reference photo — this is what keeps someone recognized as "the same
   person" even as their appearance shifts with lighting.
4. A visit is considered "over" once the person hasn't been seen for
   `SESSION_GAP_SECONDS` (default 8s), at which point it's written to
   the database with its final duration.

---

## Configuration

All settings are constants near the top of `person_appearance_tracker.py`:

| Setting | Default | What it controls |
|---|---|---|
| `CAMERA_INDEX` | `0` | Which camera to use, if you have more than one |
| `DB_PATH` | `appearances.db` | SQLite database file location |
| `SESSION_GAP_SECONDS` | `8.0` | How long someone can be out of frame before their visit is closed out |
| `MAX_VISITS_FOR_FULL_GREEN` | `10` | Visit count at which the color reaches full green |
| `MATCH_SIMILARITY_THRESHOLD` | `0.363` | How closely a face must match to count as "the same person" — lower catches more matches but risks merging different people; higher is stricter but risks creating duplicate IDs for the same person |
| `DETECTOR_SCORE_THRESHOLD` | `0.75` | Face detector confidence cutoff |
| `MIN_FACE_SIZE_FOR_GALLERY` | `60` | Minimum face size (px) before a sample is added to someone's gallery |
| `MAX_ENCODINGS_PER_PERSON` | `20` | Cap on how many reference samples are kept per person |
| `LOW_LIGHT_BRIGHTNESS_THRESHOLD` | `90` | Mean frame brightness (0–255) below which low-light enhancement kicks in |

---

## Database schema

`appearances.db` (SQLite):

- **persons** — `person_id` (permanent, primary key), `first_seen`, `total_visits`
- **encodings** — the face-sample gallery: `person_id`, `encoding`, `added_at`
- **visits** — one row per completed visit: `person_id`, `start_time`,
  `end_time`, `duration_seconds`, `visit_number`, `color_hex`, `year`,
  `month`, `day`, `hour`

You can query it directly with any SQLite tool if you want custom
reports beyond what `--report` prints.

---

## Limitations — please read

- **No detection system is 100% accurate.** Lighting, camera angle,
  occlusion (masks, hats, hands over the face), motion blur, and image
  quality all affect results. The multi-sample gallery and low-light
  enhancement in this project meaningfully improve day/night
  reliability, but they don't make it infallible — treat
  `MATCH_SIMILARITY_THRESHOLD` as a tuning knob for your environment,
  not a guarantee.
- **Software can't create light that isn't there.** In genuine darkness
  with no IR illumination, there's no image signal for any algorithm to
  work with — that's a hardware requirement (an IR/night-vision camera),
  not something code can solve.
- **This stores biometric (facial) data** to re-identify specific
  people over time. Only use it on people who've consented to being
  recognized and logged this way, and check the biometric/privacy laws
  that apply where you are (e.g. GDPR in the EU, Illinois BIPA in the
  US) before using it on anyone other than yourself.

---

## Troubleshooting

- **Same person keeps getting a new ID:** lower `MATCH_SIMILARITY_THRESHOLD`
  slightly, and give the system a few visits in different lighting so
  its gallery for that person builds up.
- **Two different people are merged into one ID:** raise
  `MATCH_SIMILARITY_THRESHOLD`.
- **Nothing detected at all in low light:** confirm your camera actually
  has a night/IR mode enabled; check `Scene brightness` shown on-screen
  to see how dim the feed actually is.
- **Model download fails on first run:** download the two files manually
  from the URLs printed in the console and place them in the `models/`
  folder using the exact filenames shown.
