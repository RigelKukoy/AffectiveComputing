
# Required libraries
import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# --------------------------
# Config / Paths / Classes
# --------------------------
DATA_DIR = "data"  # Assumes a 'data' folder with 'train' and 'test' subfolders
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Create main data directories if they don't exist to avoid errors
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTIONS)
EMO2IDX = {e:i for i,e in enumerate(EMOTIONS)}
IDX2EMO = {i:e for e,i in EMO2IDX.items()}

# Create emotion subdirectories for train and test splits
for split in [TRAIN_DIR, TEST_DIR]:
    for emo in EMOTIONS:
        os.makedirs(os.path.join(split, emo), exist_ok=True)

# Directories for caching features and saving models
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# CNN input size for MobileNetV2
CNN_SIZE = 224

# --------------------------
# Sanity checks: folder structure and counts
# --------------------------
def sanity_check_folders(train_dir=TRAIN_DIR, test_dir=TEST_DIR):
    print("Sanity check: class folders and image counts\n")
    for split_dir in [train_dir, test_dir]:
        print(f"Split: {os.path.basename(split_dir)}")
        for emo in EMOTIONS:
            folder = os.path.join(split_dir, emo)
            if not os.path.isdir(folder):
                print(f"  MISSING: {folder}")
            else:
                # Correctly count image files, not directories
                count = len(glob.glob(os.path.join(folder, "*")))
                print(f"  {emo:9s}: {count} images")
        print("")

sanity_check_folders()

# --------------------------
# Utility: show sample images with detected face boxes
# --------------------------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

def show_samples_with_faceboxes(split_dir=TRAIN_DIR, samples_per_class=3):
    fig, axes = plt.subplots(len(EMOTIONS), samples_per_class, figsize=(3*samples_per_class,3*len(EMOTIONS)))
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    for i, emo in enumerate(EMOTIONS):
        # Correctly find image files
        imgs = glob.glob(os.path.join(split_dir, emo, "*"))[:samples_per_class]
        for j in range(samples_per_class):
            ax = axes[i, j] if len(EMOTIONS)>1 else axes[j]
            if j < len(imgs):
                img_path = imgs[j]
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # FER images are grayscale
                if img is None: continue
                h,w = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                results = face_detector.process(rgb)
                if results.detections:
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        x1 = int(bbox.xmin * w)
                        y1 = int(bbox.ymin * h)
                        bw = int(bbox.width * w)
                        bh = int(bbox.height * h)
                        cv2.rectangle(rgb, (x1,y1), (x1+bw, y1+bh), (0,255,0), 1)
                ax.imshow(rgb, cmap='gray')
                ax.set_title(f"{emo}")
            else:
                ax.axis('off')
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    face_detector.close()

# Visual sample check (Milestone 1)
print("\n--- Milestone 1: Visual Face Detection Check ---")
show_samples_with_faceboxes(samples_per_class=3)

# --------------------------
# Face detection + landmark extraction (MediaPipe FaceMesh)
# Track A feature extraction: normalized flattened landmarks + EAR/MAR
# --------------------------
# MediaPipe landmark indices commonly used for eye/mouth:
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]   # approximate
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380] # approximate
MOUTH_IDXS = [13, 14, 78, 308, 82, 312]         # approximate top/bottom/left/right

def extract_landmark_features_from_image(image_gray, return_landmarks=False):
    """
    Input: grayscale image (48x48) or larger
    Output: flattened normalized landmarks (x,y) and simple ratios (EAR, MAR)
    If no face detected -> None
    """
    # Convert to RGB for MediaPipe
    h, w = image_gray.shape[:2]
    rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        coords = np.array([(pt.x * w, pt.y * h) for pt in lm.landmark])
        
        # Center & scale normalization by inter-pupillary distance
        left_eye_center = coords[33]
        right_eye_center = coords[263]
        ipd = np.linalg.norm(left_eye_center - right_eye_center)
        if ipd < 1e-6: ipd = 1.0 # Avoid division by zero
            
        center = coords.mean(axis=0)
        norm_coords = (coords - center) / ipd
        flat = norm_coords.flatten()
        
        # EAR calculation
        def ear(eye_idxs):
            p1, p2, p3, p4 = coords[eye_idxs[1]], coords[eye_idxs[2]], coords[eye_idxs[0]], coords[eye_idxs[3]]
            vert = np.linalg.norm(p1-p2)
            horiz = np.linalg.norm(p3-p4)
            return vert / (horiz + 1e-6)

        avg_ear = (ear(LEFT_EYE_IDXS) + ear(RIGHT_EYE_IDXS)) / 2.0
        
        # MAR calculation
        mouth_top, mouth_bottom, mouth_left, mouth_right = coords[MOUTH_IDXS[0]], coords[MOUTH_IDXS[1]], coords[MOUTH_IDXS[2]], coords[MOUTH_IDXS[3]]
        mar = np.linalg.norm(mouth_top-mouth_bottom) / (np.linalg.norm(mouth_left-mouth_right) + 1e-6)
        
        features = np.concatenate([flat, [avg_ear, mar]])
        
        if return_landmarks:
            return features, coords
        return features

def build_landmark_dataset(split_dir, cache_path):
    if cache_path.exists():
        print(f"Loading cached landmark features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['X'], data['y'], data['paths']
    
    X_list, y_list, paths = [], [], []
    missing = 0
    print(f"Extracting landmarks from {split_dir} ...")
    for emo in EMOTIONS:
        files = glob.glob(os.path.join(split_dir, emo, "*"))
        for p in tqdm(files, desc=f"{os.path.basename(split_dir)} - {emo}"):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            feat = extract_landmark_features_from_image(img)
            if feat is None:
                missing += 1
                continue
            X_list.append(feat)
            y_list.append(EMO2IDX[emo])
            paths.append(p)
    
    X = np.vstack(X_list) if X_list else np.zeros((0, 938))
    y = np.array(y_list, dtype=int)
    np.savez_compressed(cache_path, X=X, y=y, paths=paths)
    print(f"Done. Missing detections: {missing}")
    return X, y, paths

# Build landmark datasets
print("\n--- Milestone 2 (Track A): Building Landmark Dataset ---")
landmark_train_cache = CACHE_DIR / "landmarks_train.npz"
landmark_test_cache  = CACHE_DIR / "landmarks_test.npz"
X_land_train, y_land_train, train_paths_land = build_landmark_dataset(TRAIN_DIR, landmark_train_cache)
X_land_test,  y_land_test,  test_paths_land  = build_landmark_dataset(TEST_DIR, landmark_test_cache)

print("Landmark train shape:", X_land_train.shape, "test shape:", X_land_test.shape)

# --------------------------
# Track B: CNN Embeddings via MobileNetV2 (pretrained)
# --------------------------
print("\n--- Milestone 2 (Track B): Building CNN Embeddings Dataset ---")
backbone = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(CNN_SIZE,CNN_SIZE,3))

def image_to_mobilenet_embedding(img_gray):
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (CNN_SIZE, CNN_SIZE))
    arr = img_to_array(resized)
    arr = np.expand_dims(arr, axis=0)
    arr = mobilenet_preprocess(arr)
    emb = backbone.predict(arr, verbose=0)
    return emb.flatten()

def build_cnn_embeddings(split_dir, cache_path):
    if cache_path.exists():
        print(f"Loading cached CNN embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['X'], data['y'], data['paths']
        
    X_list, y_list, paths = [], [], []
    missing = 0
    print(f"Extracting CNN embeddings from {split_dir} ...")
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4) as det:
        for emo in EMOTIONS:
            files = glob.glob(os.path.join(split_dir, emo, "*"))
            for p in tqdm(files, desc=f"{os.path.basename(split_dir)} - {emo}"):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                res = det.process(rgb)
                face = img
                if res.detections:
                    bb = res.detections[0].location_data.relative_bounding_box
                    h,w = img.shape[:2]
                    x1, y1 = max(0, int(bb.xmin * w)), max(0, int(bb.ymin * h))
                    x2, y2 = min(w, x1 + int(bb.width * w)), min(h, y1 + int(bb.height * h))
                    if x2 > x1 and y2 > y1: face = img[y1:y2, x1:x2]
                else:
                    missing += 1
                
                emb = image_to_mobilenet_embedding(face)
                X_list.append(emb)
                y_list.append(EMO2IDX[emo])
                paths.append(p)
                
    X = np.vstack(X_list) if X_list else np.zeros((0, backbone.output_shape[-1]))
    y = np.array(y_list, dtype=int)
    np.savez_compressed(cache_path, X=X, y=y, paths=paths)
    print(f"Done. Missing face detections: {missing}")
    return X, y, paths

cnn_train_cache = CACHE_DIR / "cnn_train.npz"
cnn_test_cache  = CACHE_DIR / "cnn_test.npz"
X_cnn_train, y_cnn_train, train_paths_cnn = build_cnn_embeddings(TRAIN_DIR, cnn_train_cache)
X_cnn_test,  y_cnn_test,  test_paths_cnn  = build_cnn_embeddings(TEST_DIR, cnn_test_cache)

print("CNN train shape:", X_cnn_train.shape, "test shape:", X_cnn_test.shape)

# --------------------------
# Model training function
# --------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test, feature_type="landmark"):
    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    print(f"\n--- Training Logistic Regression on {feature_type} features ---")
    clf_lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    clf_lr.fit(X_train_scaled, y_train)
    y_pred_lr = clf_lr.predict(X_test_scaled)
    
    results['lr'] = {
        'model': clf_lr, 'scaler': scaler,
        'acc': accuracy_score(y_test, y_pred_lr),
        'f1_macro': f1_score(y_test, y_pred_lr, average='macro'),
        'report': classification_report(y_test, y_pred_lr, target_names=EMOTIONS, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr),
        'y_pred': y_pred_lr
    }

    # k-NN
    print(f"\n--- Training k-NN on {feature_type} features ---")
    k = int(np.sqrt(len(y_train)))
    if k % 2 == 0: k += 1
    clf_knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf_knn.fit(X_train_scaled, y_train)
    y_pred_knn = clf_knn.predict(X_test_scaled)
    
    results['knn'] = {
        'model': clf_knn, 'scaler': scaler, 'k': k,
        'acc': accuracy_score(y_test, y_pred_knn),
        'f1_macro': f1_score(y_test, y_pred_knn, average='macro'),
        'report': classification_report(y_test, y_pred_knn, target_names=EMOTIONS, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_knn),
        'y_pred': y_pred_knn
    }
    
    return results

# --------------------------
# Milestone 3: Evaluation & Reflection
# --------------------------
print("\n\n--- Milestone 3: Model Evaluation ---")

# Train & evaluate for Track A (Landmarks)
res_land = {}
if X_land_train.shape[0] > 0:
    res_land = train_and_evaluate(X_land_train, y_land_train, X_land_test, y_land_test, feature_type="landmark")
else:
    print("Skipping Landmark models: No features were extracted.")

# Train & evaluate for Track B (CNN)
res_cnn = {}
if X_cnn_train.shape[0] > 0:
    res_cnn = train_and_evaluate(X_cnn_train, y_cnn_train, X_cnn_test, y_cnn_test, feature_type="cnn")
else:
    print("Skipping CNN models: No features were extracted.")

# Helper to plot confusion matrix
def plot_confusion_matrix(cm, title="Confusion matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Show all reports and confusion matrices
print("\n\n--- DETAILED EVALUATION REPORTS ---")
if res_land:
    print("\n=== Track A: Landmark Feature Results ===")
    print("\nLogistic Regression Report:")
    print(res_land['lr']['report'])
    plot_confusion_matrix(res_land['lr']['confusion_matrix'], title="Landmark + Logistic Regression")
    
    print("\nk-NN Report:")
    print(res_land['knn']['report'])
    plot_confusion_matrix(res_land['knn']['confusion_matrix'], title=f"Landmark + k-NN (k={res_land['knn']['k']})")

if res_cnn:
    print("\n=== Track B: CNN Embedding Results ===")
    print("\nLogistic Regression Report:")
    print(res_cnn['lr']['report'])
    plot_confusion_matrix(res_cnn['lr']['confusion_matrix'], title="CNN + Logistic Regression")
    
    print("\nk-NN Report:")
    print(res_cnn['knn']['report'])
    plot_confusion_matrix(res_cnn['knn']['confusion_matrix'], title=f"CNN + k-NN (k={res_cnn['knn']['k']})")

# --------------------------
# Save best pipelines and create predict function
# --------------------------
def pick_best_model(results):
    return max(results.items(), key=lambda item: item[1]['f1_macro'])[0] if results else None

best_land_model_name = pick_best_model(res_land)
best_cnn_model_name = pick_best_model(res_cnn)

print(f"\nBest Landmark Model (by Macro F1): {best_land_model_name}")
print(f"Best CNN Model (by Macro F1): {best_cnn_model_name}")

if best_land_model_name:
    joblib.dump(res_land[best_land_model_name], MODEL_DIR / "best_landmark_pipeline.joblib")
    print(f"Saved best landmark pipeline to {MODEL_DIR / 'best_landmark_pipeline.joblib'}")
if best_cnn_model_name:
    joblib.dump(res_cnn[best_cnn_model_name], MODEL_DIR / "best_cnn_pipeline.joblib")
    print(f"Saved best cnn pipeline to {MODEL_DIR / 'best_cnn_pipeline.joblib'}")

# --------------------------
# Final Predict Function
# --------------------------
def predict_emotion(image_path, pipeline_path):
    pipeline = joblib.load(pipeline_path)
    model = pipeline['model']
    scaler = pipeline['scaler']
    feature_type = 'landmark' if 'landmark' in str(pipeline_path) else 'cnn'
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    if feature_type == 'landmark':
        features = extract_landmark_features_from_image(img)
        if features is None:
            return "No face detected", None
        features = features.reshape(1, -1)
    else: # cnn
        features = image_to_mobilenet_embedding(img).reshape(1, -1)

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    return IDX2EMO[prediction], img

# --- Demonstrate Predict Function on Demo Images ---
print("\n--- Prediction Function Demonstration on Demo Images ---")

# Use demo images from demo_rivas folder
demo_folder = "demo_rivas"
demo_files = {
    "sad": os.path.join(demo_folder, "demo_sad.jpg"),
    "fear": os.path.join(demo_folder, "demo_fear.jpg"),
    "happy": os.path.join(demo_folder, "demo_happy.jpg")
}

# Determine the best pipeline to use
best_pipeline_path = None
f1_land = res_land[best_land_model_name]['f1_macro'] if best_land_model_name else -1
f1_cnn = res_cnn[best_cnn_model_name]['f1_macro'] if best_cnn_model_name else -1

if f1_cnn > f1_land:
    best_pipeline_path = MODEL_DIR / "best_cnn_pipeline.joblib"
    print("Using the best CNN pipeline for demonstration.")
elif f1_land != -1:
    best_pipeline_path = MODEL_DIR / "best_landmark_pipeline.joblib"
    print("Using the best Landmark pipeline for demonstration.")

if best_pipeline_path and os.path.exists(best_pipeline_path):
    fig, axes = plt.subplots(1, len(demo_files), figsize=(5*len(demo_files), 5))
    if len(demo_files) == 1:
        axes = [axes]
    
    for idx, (expected_emotion, img_path) in enumerate(demo_files.items()):
        if os.path.exists(img_path):
            try:
                predicted_emotion, img_arr = predict_emotion(img_path, best_pipeline_path)
                
                print(f"Demo Image: {os.path.basename(img_path)}")
                print(f"  -> Expected Emotion:  {expected_emotion}")
                print(f"  -> Predicted Emotion: {predicted_emotion}")
                
                # Load original color image for display
                img_color = cv2.imread(img_path)
                if img_color is not None:
                    img_display = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                else:
                    img_display = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
                
                axes[idx].imshow(img_display)
                axes[idx].set_title(f"Expected: {expected_emotion}\nPredicted: {predicted_emotion}")
                axes[idx].axis('off')
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                axes[idx].axis('off')
        else:
            print(f"Demo file not found: {img_path}")
            axes[idx].axis('off')
    
    plt.suptitle("Demo Images Predictions")
    plt.tight_layout()
    plt.show()
else:
    print("No trained pipeline available for demonstration.")

# --------------------------
# Final Comparative Summary Table
# --------------------------
summary_rows = []
for track, results in [("Landmark", res_land), ("CNN", res_cnn)]:
    if results:
        for model_name, info in results.items():
            summary_rows.append({
                'Track': track,
                'Model': model_name.upper(),
                'Accuracy': f"{info['acc']:.4f}",
                'Macro F1-Score': f"{info['f1_macro']:.4f}"
            })

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    print("\n\n--- FINAL COMPARATIVE SUMMARY ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("model_summary.csv", index=False)
    print("\nSummary table saved to model_summary.csv")

print("\nScript finished successfully.")
```