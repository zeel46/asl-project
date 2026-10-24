# ASL Project – Real-Time Sign Language Recognition

A web app that recognises American Sign Language from a webcam in real time and turns it into text. The main classifier is a **K-Nearest Neighbours model written from scratch** (`knn_scratch.py`); no machine learning library is used. MediaPipe is used only to find hand and body points.

## What it recognises

**Letters (A–Z)**
24 static letters are classified frame by frame with KNN, by comparing the hand's 21 points with 36,715 stored hands. Hold a letter still for 1 second to type it. J and Z are moving letters: draw them in the air and they are detected from the fingertip's path. Accuracy on people the model never saw: **97.1%**.

**Numbers (1–9)**
Recognised the same way as letters, in their own tab because some numbers share a handshape with letters (2/V, 6/W, 9/F). Accuracy on unseen people: **100%**.

**Words (20 common words)**
Press Record and sign one word. The 2.5-second movement is compared with 194 stored word videos using Dynamic Time Warping + KNN, so the same sign matches even at different speeds. The app also suggests replies and shows a video of how to sign them back. Accuracy on unseen signers: **77.3%**.

Words: thank you, sorry, please, yes, no, help, good, bad, what, where, who, you, eat, drink, water, doctor, bathroom, fine, sick, family.

## How to run

Requires **Python 3.10** and a webcam.

```bash
git clone https://github.com/zeel46/asl-project.git
cd asl-project
python -m venv aslenv
aslenv\Scripts\pip install -r requirements.txt
aslenv\Scripts\python app.py
```

Then open **http://127.0.0.1:5000** in Chrome or Edge and click **Turn camera on**.
(On macOS/Linux use `aslenv/bin/pip` and `aslenv/bin/python`.)

The trained models are included, so no training is needed. To retrain: `python train_knn.py` (letters and numbers) and `python train_words.py` (words).

## Datasets

- [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Kaggle)
- [ASL-HG](https://data.mendeley.com/datasets/j4y5w2c8w9/1) (Mendeley Data, CC BY 4.0)
- [WLASL](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) (academic use only)
