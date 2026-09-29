# Face Mask Detector

Real-time face mask detection with a MobileNetV2 classifier and OpenCV. Runs two ways: as a browser app you can point a camera at, or as a local webcam script.

Each detected face is boxed and labelled with a confidence score — green for **Mask**, red for **No Mask**.

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| With mask | 0.97 | 0.99 | 0.98 |
| Without mask | 0.99 | 0.97 | 0.98 |
| **Overall** | **0.98** | **0.98** | **0.98** |

## How it works

Detection is two models in sequence, which is what keeps it accurate on faces the classifier never saw during training:

1. **Face detection** — an OpenCV DNN SSD (`res10_300x300`) finds every face in the frame and returns its bounding box. Only boxes above 50% confidence are kept.
2. **Mask classification** — each face is cropped, resized to 224×224, preprocessed and passed to a MobileNetV2 classifier fine-tuned on masked and unmasked faces.

Splitting the job this way means the classifier only ever sees a tight face crop, so background clutter does not affect it.

## Run it

```bash
pip install -r requirements.txt
```

**Browser app** — works on a laptop or a phone, no local webcam setup needed:

```bash
streamlit run app.py
```

**Local webcam script** — opens a live OpenCV window, press `q` to quit:

```bash
python detect_mask.py
```

## Project structure

| Path | Role |
|---|---|
| `app.py` | Streamlit app — takes a camera snapshot and labels every face found |
| `detect_mask.py` | Live webcam detection in an OpenCV window |
| `mask_detector.h5` | Trained MobileNetV2 classifier (12 MB) |
| `face_detector/` | OpenCV SSD face-detection model and its prototxt |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Python version pin for deployment |

## Tech stack

Python · TensorFlow / Keras · MobileNetV2 (transfer learning) · OpenCV DNN · Streamlit · NumPy

## Author

**Md. Mahabubul Hasan** — [@mahabub-automation](https://github.com/mahabub-automation) · [LinkedIn](https://linkedin.com/in/mahabubulhasan)
