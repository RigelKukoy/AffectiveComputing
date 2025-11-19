# -------------------------
# Install / Import libs
# -------------------------
!pip install -q mediapipe tensorflow==2.12.0 tqdm scikit-learn matplotlib pandas

import os, sys, math, time, random, glob
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
plt.rcParams.update({'figure.max_open_warning': 0})

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -------------------------
# Configuration (tweak if needed)
# -------------------------
DATA_CSV = "fer2013.csv"      # expected CSV file in working directory
SAMPLE_PER_CLASS = 500        # set None to use entire dataset (may be slow). Use 500 for reasonable run time.
USE_CNN_TRACK = True          # True to run Track B embeddings (slower)
CNN_SIZE = 224                # input size for MobileNetV2
MAX_IMAGES_TOTAL = None       # None or integer to cap total images processed (across classes)
SHOW_N_SAMPLES_PER_CLASS = 3  # for visual checks
KNN_K = 5                     # neighbors for kNN (you can change)
VERBOSE = True

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]
EMO2IDX = {e:i for i,e in enumerate(EMOTIONS)}
IDX2EMO = {i:e for e,i in EMO2IDX.items()}

print("Python", sys.version)
print("TF", tf.__version__)
print("Mediapipe", mp.__version__)

# -------------------------
# Load FER2013 CSV (download via Kaggle if needed)
# -------------------------
def ensure_fer_csv():
    if Path(DATA_CSV).exists():
        print(f"Found {DATA_CSV}.")
        return True
    print(f"{DATA_CSV} not found. Attempting Kaggle download (you'll be prompted to upload kaggle.json).")
    try:
        from google.colab import files
        print("Please upload kaggle.json (Kaggle API credentials). If you don't have it, upload fer2013.csv manually.")
        uploaded = files.upload()
        # copy uploaded kaggle.json to ~/.kaggle if present
        if 'kaggle.json' in uploaded:
            !mkdir -p ~/.kaggle
            !cp kaggle.json ~/.kaggle/
            !chmod 600 ~/.kaggle/kaggle.json
            # download dataset
            !kaggle datasets download -d msambare/fer2013 -q
            !unzip -o fer2013.zip
            if Path(DATA_CSV).exists():
                print("Downloaded fer2013.csv.")
                return True
            else:
                print("Download attempted but fer2013.csv not found after unzip.")
                return False
        else:
            # maybe user uploaded fer2013.csv directly
            if DATA_CSV in uploaded:
                print("fer2013.csv uploaded.")
                return True
            print("No kaggle.json or fer2013.csv uploaded.")
            return False
    except Exception as e:
        print("Automatic Kaggle download failed:", e)
        return False

ok = ensure_fer_csv()
if not ok:
    raise FileNotFoundError("fer2013.csv required. Upload it to Colab and re-run.")

df = pd.read_csv(DATA_CSV)
print("CSV loaded, total rows:", len(df))
df.head()

# -------------------------
# Preprocess pixel string -> numpy 48x48
# -------------------------
def pixels_to_image(pixels_str):
    arr = np.fromstring(pixels_str, dtype=int, sep=' ')
    if arr.size != 48*48:
        # sometimes trailing spaces -> fallback
        arr = np.array(pixels_str.split(), dtype=int)
    img = arr.reshape(48,48).astype(np.uint8)
    return img

df['img'] = df['pixels'].apply(pixels_to_image)
df['label'] = df['emotion'].apply(lambda x: EMOTIONS[x] if isinstance(x, (int,np.integer)) else x)

# -------------------------
# Build balanced small dataset (optional)
# -------------------------
def build_split_df(df, per_class=None):
    groups=[]
    for idx, emo in enumerate(EMOTIONS):
        subset = df[df['emotion']==idx]
        if per_class is None:
            groups.append(subset)
        else:
            if len(subset) <= per_class:
                groups.append(subset)
            else:
                groups.append(subset.sample(per_class, random_state=RANDOM_STATE))
    out = pd.concat(groups).reset_index(drop=True)
    return out

df_small = build_split_df(df, per_class=SAMPLE_PER_CLASS)
print("Using dataset rows:", len(df_small))
display(df_small['emotion'].value_counts().sort_index().rename(index=IDX2EMO))

# -------------------------
# Milestone 1: Face detection and visual check (show samples with bounding boxes)
# -------------------------
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)

def detect_faces_and_draw(img_gray):
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    res = face_detector.process(rgb)
    out = rgb.copy()
    boxes = []
    if res.detections:
        for det in res.detections:
            bb = det.location_data.relative_bounding_box
            h,w = img_gray.shape[:2]
            x1 = int(max(0, bb.xmin * w))
            y1 = int(max(0, bb.ymin * h))
            x2 = int(min(w, x1 + bb.width * w))
            y2 = int(min(h, y1 + bb.height * h))
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 1)
            boxes.append((x1,y1,x2,y2))
    return out, boxes, res

# Show samples per class
def plot_samples_with_boxes(df_small, n=SHOW_N_SAMPLES_PER_CLASS):
    fig_h = len(EMOTIONS)
    fig, axes = plt.subplots(len(EMOTIONS), n, figsize=(3*n, 3*len(EMOTIONS)))
    for i, emo in enumerate(EMOTIONS):
        rows = df_small[df_small['label']==emo]
        samples = rows.sample(min(n, len(rows)), random_state=RANDOM_STATE).reset_index(drop=True)
        for j in range(n):
            ax = axes[i,j] if len(EMOTIONS)>1 else axes[j]
            if j < len(samples):
                img = samples.loc[j,'img']
                out, boxes, _ = detect_faces_and_draw(img)
                ax.imshow(out)
                ax.set_title(f"{emo}")
            ax.axis('off')
    plt.tight_layout()
    plt.suptitle("Sample images with detected face boxes (Milestone 1)", y=1.02)
    plt.show()

plot_samples_with_boxes(df_small)

# -------------------------
# Milestone 2 - Track A: Landmark features using MediaPipe FaceMesh
# - Normalize landmarks by inter-pupillary distance
# - Flatten + include EAR/MAR approximations
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                  refine_landmarks=True, min_detection_confidence=0.4)

# indices for approximate eyes and mouth (MediaPipe FaceMesh 468 landmarks)
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

def extract_facemesh_features(img_gray):
    # returns flattened normalized landmarks + ear + mar, or None if no face
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark  # 468 landmarks
    h,w = img_gray.shape[:2]
    coords = np.array([[p.x*w, p.y*h] for p in lm])  # (468,2)
    # normalize: center and scale by inter-pupillary distance
    left = coords[LEFT_EYE_OUTER]
    right = coords[RIGHT_EYE_OUTER]
    ipd = np.linalg.norm(left-right) if np.linalg.norm(left-right) > 1e-6 else 1.0
    center = coords.mean(axis=0)
    norm_coords = (coords - center) / ipd
    flat = norm_coords.flatten()  # length 936
    # EAR approx
    def ear(eye_idxs):
        p_v1 = coords[eye_idxs[1]]
        p_v2 = coords[eye_idxs[2]]
        p_h1 = coords[eye_idxs[0]]
        p_h2 = coords[eye_idxs[3]]
        vert = np.linalg.norm(p_v1 - p_v2) + 1e-6
        horiz = np.linalg.norm(p_h1 - p_h2) + 1e-6
        return vert / horiz
    left_ear = ear(LEFT_EYE_IDXS)
    right_ear = ear(RIGHT_EYE_IDXS)
    avg_ear = (left_ear + right_ear)/2.0
    mar = np.linalg.norm(coords[MOUTH_TOP_IDX]-coords[MOUTH_BOTTOM_IDX]) / (ipd + 1e-6)
    features = np.concatenate([flat, [avg_ear, mar]])
    return features

# Bulk extract Track A features (careful with runtime)
print("\nExtracting Track A (FaceMesh) features...")
X_A = []
y_A = []
paths_A = []

for idx, row in tqdm(df_small.iterrows(), total=len(df_small)):
    img = row['img']
    feat = extract_facemesh_features(img)
    if feat is None:
        continue
    X_A.append(feat)
    y_A.append(row['emotion'])
    paths_A.append(idx)
X_A = np.array(X_A)
y_A = np.array(y_A)
print("Track A features shape:", X_A.shape, "labels shape:", y_A.shape)

# -------------------------
# Milestone 2 - Track B: CNN embeddings (MobileNetV2)
# - Crop using face detector (if available) then resize to 224x224
# -------------------------
if USE_CNN_TRACK:
    print("\nPreparing MobileNetV2 backbone (this can be slow on CPU)...")
    backbone = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(CNN_SIZE,CNN_SIZE,3))
    def image_to_mobilenet_embedding(img_gray):
        # crop by face detection if possible
        out, boxes, det = detect_faces_and_draw(img_gray)
        if len(boxes)>0:
            x1,y1,x2,y2 = boxes[0]
            face = img_gray[y1:y2, x1:x2] if y2>y1 and x2>x1 else img_gray
        else:
            face = img_gray
        # convert to 3-channel and resize
        rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(rgb, (CNN_SIZE, CNN_SIZE))
        arr = img_to_array(resized)
        arr = np.expand_dims(arr, axis=0)
        arr = mobilenet_preprocess(arr)
        emb = backbone.predict(arr, verbose=0)
        return emb.flatten()
    print("Extracting Track B (CNN) embeddings...")
    X_B = []
    y_B = []
    paths_B = []
    for idx, row in tqdm(df_small.iterrows(), total=len(df_small)):
        img = row['img']
        try:
            emb = image_to_mobilenet_embedding(img)
            X_B.append(emb)
            y_B.append(row['emotion'])
            paths_B.append(idx)
        except Exception as e:
            # skip problematic samples
            continue
    X_B = np.array(X_B)
    y_B = np.array(y_B)
    print("Track B embeddings shape:", X_B.shape, "labels shape:", y_B.shape)
else:
    X_B, y_B = np.array([]), np.array([])

# -------------------------
# Train/Test split (use a stratified split on each track separately)
# -------------------------
def train_test_for_track(X, y, test_size=0.2):
    if X.shape[0] == 0:
        return None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    return X_tr, X_te, y_tr, y_te

split_A = train_test_for_track(X_A, y_A)
split_B = train_test_for_track(X_B, y_B) if USE_CNN_TRACK else None

# -------------------------
# Model training + evaluation function
# -------------------------
def train_eval_models(X_tr, y_tr, X_te, y_te, track_name="track"):
    out = {}
    # Standardize
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    # Logistic Regression
    print(f"\nTraining Logistic Regression ({track_name}) ...")
    clf_lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, multi_class='multinomial', solver='saga')
    clf_lr.fit(X_tr_s, y_tr)
    y_pred_lr = clf_lr.predict(X_te_s)
    acc_lr = accuracy_score(y_te, y_pred_lr)
    f1_lr = f1_score(y_te, y_pred_lr, average='macro')
    cr_lr = classification_report(y_te, y_pred_lr, target_names=EMOTIONS, zero_division=0)
    cm_lr = confusion_matrix(y_te, y_pred_lr)
    out['lr'] = dict(model=clf_lr, scaler=scaler, acc=acc_lr, f1_macro=f1_lr, report=cr_lr, cm=cm_lr, y_pred=y_pred_lr)

    # k-NN
    print(f"\nTraining k-NN (k={KNN_K}) ({track_name}) ...")
    clf_knn = KNeighborsClassifier(n_neighbors=KNN_K)
    clf_knn.fit(X_tr_s, y_tr)
    y_pred_knn = clf_knn.predict(X_te_s)
    acc_knn = accuracy_score(y_te, y_pred_knn)
    f1_knn = f1_score(y_te, y_pred_knn, average='macro')
    cr_knn = classification_report(y_te, y_pred_knn, target_names=EMOTIONS, zero_division=0)
    cm_knn = confusion_matrix(y_te, y_pred_knn)
    out['knn'] = dict(model=clf_knn, scaler=scaler, acc=acc_knn, f1_macro=f1_knn, report=cr_knn, cm=cm_knn, y_pred=y_pred_knn)

    # Print summaries
    print(f"\n=== {track_name.upper()} RESULTS ===")
    print("Logistic Regression - acc:", out['lr']['acc'], "macro-F1:", out['lr']['f1_macro'])
    print(out['lr']['report'])
    print("k-NN - acc:", out['knn']['acc'], "macro-F1:", out['knn']['f1_macro'])
    print(out['knn']['report'])
    return out

res_A = None
res_B = None

if split_A:
    X_tr_A, X_te_A, y_tr_A, y_te_A = split_A
    res_A = train_eval_models(X_tr_A, y_tr_A, X_te_A, y_te_A, track_name="Track A (Landmarks)")
else:
    print("No Track A data available (no faces detected).")

if USE_CNN_TRACK and split_B:
    X_tr_B, X_te_B, y_tr_B, y_te_B = split_B
    res_B = train_eval_models(X_tr_B, y_tr_B, X_te_B, y_te_B, track_name="Track B (CNN embeddings)")
else:
    print("No Track B run (disabled or no embeddings).")

# -------------------------
# Visualization helpers: confusion matrices, normalized confusion, barplot for per-class F1
# -------------------------
def plot_confusion(cm, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax, cmap='Blues')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.show()

def plot_confusion_normalized(cm, title="Normalized confusion matrix"):
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax, cmap='Reds')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.show()

def per_class_f1_from_report(report_text):
    # parse classification_report text into dict of f1 scores
    lines = report_text.splitlines()
    class_scores = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] in EMOTIONS:
            class_scores[parts[0]] = float(parts[-2])  # F1-score column
    return class_scores

# Plot for Track A
if res_A:
    for name, info in res_A.items():
        plot_confusion(info['cm'], title=f"Track A ({name}) Confusion Matrix")
        plot_confusion_normalized(info['cm'], title=f"Track A ({name}) Normalized Confusion Matrix")
        f1s = per_class_f1_from_report(info['report'])
        # bar plot of per-class F1
        fig, ax = plt.subplots(figsize=(8,3))
        sns.barplot(x=list(f1s.keys()), y=list(f1s.values()), ax=ax)
        ax.set_ylim(0,1)
        ax.set_title(f"Track A ({name}) per-class F1")
        plt.xticks(rotation=45)
        plt.show()

# Plot for Track B
if res_B:
    for name, info in res_B.items():
        plot_confusion(info['cm'], title=f"Track B ({name}) Confusion Matrix")
        plot_confusion_normalized(info['cm'], title=f"Track B ({name}) Normalized Confusion Matrix")
        f1s = per_class_f1_from_report(info['report'])
        fig, ax = plt.subplots(figsize=(8,3))
        sns.barplot(x=list(f1s.keys()), y=list(f1s.values()), ax=ax)
        ax.set_ylim(0,1)
        ax.set_title(f"Track B ({name}) per-class F1")
        plt.xticks(rotation=45)
        plt.show()

# -------------------------
# Show some misclassified examples (2-3 per model)
# -------------------------
def show_misclassified(X_test_idx_list, y_true, y_pred, model_label, df_source):
    idxs = np.where(y_true != y_pred)[0]
    n_show = min(6, len(idxs))
    if n_show == 0:
        print(f"No misclassified examples for {model_label}.")
        return
    chosen = idxs[:n_show]
    fig, axes = plt.subplots(1, n_show, figsize=(3*n_show,3))
    for i, idx in enumerate(chosen):
        global_idx = X_test_idx_list[idx]
        row = df_source.loc[global_idx]
        img = row['img']
        true_label = EMOTIONS[y_true[idx]]
        pred_label = EMOTIONS[y_pred[idx]]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"T:{true_label}\nP:{pred_label}")
        axes[i].axis('off')
    plt.suptitle(f"Misclassified examples - {model_label}")
    plt.show()

# Need mapping from test split indices back to original df_small indices.
# For Track A:
if split_A:
    # we kept 'paths_A' as original indices (row indices of df_small)
    # But we built paths_A earlier as 'paths_A' referencing df_small index; re-create mapping
    # Reconstruct test index list for A:
    # We have X_A built from df_small indices in paths_A sequence; but we stored only positions
    # To reconstruct, rebuild the index list in same order as X_A creation:
    indices_A = []
    # Recreate: iterate df_small and extract features again to get indices in same order (cheap since small)
    for idx, row in df_small.iterrows():
        if extract_facemesh_features(row['img']) is not None:
            indices_A.append(idx)
    # Now indices_A maps each row in X_A to df_small index.
    # During split, we created X_tr_A, X_te_A by train_test_split — we can recalc test idxs:
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    for tr_idx, te_idx in sss.split(np.zeros(len(indices_A)), y_A):
        test_idx_list_A = [indices_A[i] for i in te_idx]
    # Show misclassified for best models
    show_misclassified(test_idx_list_A, y_te_A, res_A['lr']['y_pred'], "Track A - LR", df_small)
    show_misclassified(test_idx_list_A, y_te_A, res_A['knn']['y_pred'], "Track A - kNN", df_small)

# For Track B:
if USE_CNN_TRACK and split_B:
    indices_B = []
    for idx, row in df_small.iterrows():
        # attempt embedding generation quickly to reconstruct order (we did earlier)
        try:
            _ = image_to_mobilenet_embedding(row['img'])
            indices_B.append(idx)
        except:
            continue
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    for tr_idx, te_idx in sss.split(np.zeros(len(indices_B)), y_B):
        test_idx_list_B = [indices_B[i] for i in te_idx]
    show_misclassified(test_idx_list_B, y_te_B, res_B['lr']['y_pred'], "Track B - LR", df_small)
    show_misclassified(test_idx_list_B, y_te_B, res_B['knn']['y_pred'], "Track B - kNN", df_small)

# -------------------------
# Predict wrapper (simple) - runs chosen pipeline on a new FER-style image (48x48 grayscale)
# -------------------------
# Save best model objects in memory (we don't write to disk)
best_pipelines = {}
if res_A:
    # pick best by macro F1
    best_name_A = 'lr' if res_A['lr']['f1_macro'] >= res_A['knn']['f1_macro'] else 'knn'
    best_pipelines['A'] = (res_A[best_name_A]['model'], res_A[best_name_A]['scaler'], best_name_A)
    print("Best Track A pipeline:", best_name_A, "F1:", res_A[best_name_A]['f1_macro'])
if res_B:
    best_name_B = 'lr' if res_B['lr']['f1_macro'] >= res_B['knn']['f1_macro'] else 'knn'
    best_pipelines['B'] = (res_B[best_name_B]['model'], res_B[best_name_B]['scaler'], best_name_B)
    print("Best Track B pipeline:", best_name_B, "F1:", res_B[best_name_B]['f1_macro'])

def predict_fer_from_image(img_gray, pipeline='A'):
    """
    pipeline: 'A' for landmarks pipeline, 'B' for CNN pipeline
    returns (pred_label_str, probs_or_None)
    """
    if pipeline=='A':
        feat = extract_facemesh_features(img_gray)
        if feat is None:
            return None, None
        model, scaler, name = best_pipelines['A']
        x = scaler.transform(feat.reshape(1,-1))
        pred = model.predict(x)[0]
        probs = model.predict_proba(x) if hasattr(model, "predict_proba") else None
        return EMOTIONS[int(pred)], probs
    else:
        emb = image_to_mobilenet_embedding(img_gray)
        model, scaler, name = best_pipelines['B']
        x = scaler.transform(emb.reshape(1,-1))
        pred = model.predict(x)[0]
        probs = model.predict_proba(x) if hasattr(model, "predict_proba") else None
        return EMOTIONS[int(pred)], probs

# Demonstrate predict on demo images from demo_rivas folder
print("\nDemonstrating predict function on demo images from demo_rivas folder:")
demo_folder = "demo_rivas"
demo_files = {
    "sad": os.path.join(demo_folder, "demo_sad.jpg"),
    "fear": os.path.join(demo_folder, "demo_fear.jpg"),
    "happy": os.path.join(demo_folder, "demo_happy.jpg")
}

fig, axes = plt.subplots(1, len(demo_files), figsize=(5*len(demo_files), 5))
if len(demo_files) == 1:
    axes = [axes]

for idx, (expected_emotion, filepath) in enumerate(demo_files.items()):
    if os.path.exists(filepath):
        # Load image and convert to grayscale
        img_color = cv2.imread(filepath)
        if img_color is None:
            print(f"Could not load {filepath}")
            continue
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # Try predict A and B if available
        out_A, probA = (None, None)
        out_B, probB = (None, None)
        if 'A' in best_pipelines:
            out_A, probA = predict_fer_from_image(img_gray, pipeline='A')
        if 'B' in best_pipelines:
            out_B, probB = predict_fer_from_image(img_gray, pipeline='B')
        
        print(f"Demo {filepath} expected:{expected_emotion} -> Landmark:{out_A} ; CNN:{out_B}")
        
        # Display image with predictions
        axes[idx].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"Expected: {expected_emotion}\nLandmark: {out_A}\nCNN: {out_B}")
        axes[idx].axis('off')
    else:
        print(f"Demo file not found: {filepath}")
        axes[idx].axis('off')

plt.suptitle("Demo Images Predictions")
plt.tight_layout()
plt.show()

# -------------------------
# Comparative summary table (accuracy, macro-F1) inline
# -------------------------
rows = []
if res_A:
    for k,v in res_A.items():
        rows.append(dict(track='landmark', model=k, accuracy=v['acc'], macro_f1=v['f1_macro']))
if res_B:
    for k,v in res_B.items():
        rows.append(dict(track='cnn', model=k, accuracy=v['acc'], macro_f1=v['f1_macro']))
summary_df = pd.DataFrame(rows).sort_values(['track','model']).reset_index(drop=True)
print("\nComparative summary (accuracy + macro-F1):")
display(summary_df)

# -------------------------
# Short textual reflection (printed inline)
# -------------------------
print("\nReflection suggestions (copy into your report):")
print("- Check confusion matrices: classes with mutual confusion (e.g., happy vs neutral) are typical.")
print("- If class imbalance hurts performance, use class weights or oversampling.")
print("- Track B (CNN embeddings) often outperforms raw landmarks when there is enough data; Track A is more interpretable.")
print("- Improvements: data augmentation, fine-tuning the backbone, ensembling both pipelines.")
print("\nNotebook complete. All visualizations displayed above inline.")