import os
import io
import time
import base64
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads_svc'
app.config['RESULTS_FOLDER'] = 'results_svc'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

CURRENT_DATA = None
SIMULATION_INDEX = 0
TRAINING_STATE = {"busy": False, "progress": 0, "status": "Idle", "eta": "0.0", "result": None}

def run_svc_task(df, save_root):
    global TRAINING_STATE
    try:
        rows = len(df)
        # SVC is slower than GMM, so we adjust time estimation
        est_time = max(8.0, 5.0 + (rows / 5000))
        start_t = time.time()
        
        def update(p, s):
            rem = max(0.0, est_time - (time.time() - start_t))
            TRAINING_STATE.update({"progress": p, "status": s, "eta": f"{rem:.1f}"})

        # --- STEP 1: FEATURE ENGINEERING (Same as GMM) ---
        update(10, "Extracting Features...")
        
        feature_set = ['Rate', 'Size']
        
        # Burstiness
        if 'Jitter' in df.columns: df['Burstiness'] = df['Jitter']
        else: df['Burstiness'] = df['Rate'].rolling(10).std().fillna(0) / (df['Rate'].rolling(10).mean().fillna(1) + 1e-9)
        feature_set.append('Burstiness')

        # Latency
        if 'Duration' in df.columns: df['Latency'] = df['Duration'] * 1000 
        else: df['Latency'] = (1 / (1000 - df['Rate'].clip(upper=990))) * 1000 + (df['Size'] * 0.01)

        # Loss
        if 'Loss' in df.columns:
            if df['Loss'].max() < 1.0: df['Loss'] = df['Loss'] * 100 
        else:
            util = df['Rate'] / 1000
            df['Loss'] = (util**4) * (1 + df['Burstiness']) * 100
            df['Loss'] = df['Loss'].clip(0, 100)

        # --- STEP 2: GROUND TRUTH GENERATION ---
        # SVC needs labels to learn. We generate them using the "Expert Rules".
        update(25, "Generating Ground Truth...")
        
        conditions = [
            (df['Rate'] > 80) & (df['Burstiness'] > 0.1),       # eMBB
            (df['Latency'] < 10) & (df['Loss'] < 0.1),          # uRLLC
            (df['Rate'] <= 80)                                  # mMTC
        ]
        ALL_LABELS = ['eMBB', 'mMTC', 'uRLLC']
        df['GroundTruth'] = np.select(conditions, ALL_LABELS, default='mMTC')
        
        # --- STEP 3: TRAIN/TEST SPLIT ---
        update(40, "Splitting Data (80/20)...")
        X = df[feature_set].values
        y = df['GroundTruth'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split: Train on 80%, Test on 20%
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # --- STEP 4: SVC TRAINING ---
        update(60, "Training Support Vector Classifier...")
        
        # RBF Kernel is standard for non-linear boundaries
        svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
        svm_model.fit(X_train, y_train)
        
        # Count Support Vectors (Unique metric for SVC)
        num_support_vectors = len(svm_model.support_)

        # --- STEP 5: PREDICTION & EVALUATION ---
        update(80, "Evaluating Model...")
        
        # Predict on the whole dataset for visualization
        all_preds = svm_model.predict(X_scaled)
        df['Predicted'] = all_preds
        
        acc = accuracy_score(df['GroundTruth'], all_preds)
        _, _, f1, _ = precision_recall_fscore_support(df['GroundTruth'], all_preds, average='weighted', zero_division=0)
        
        avg_thru = df['Rate'].mean()
        avg_lat = df['Latency'].mean()
        avg_loss = df['Loss'].mean()

        # --- STEP 6: PLOTTING ---
        update(90, "Generating Visuals...")
        
        run_id = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(save_root, run_id)
        os.makedirs(save_dir, exist_ok=True)

        # Plot 1: Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        cm_data = confusion_matrix(df['GroundTruth'], df['Predicted'], labels=ALL_LABELS)
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Oranges', xticklabels=ALL_LABELS, yticklabels=ALL_LABELS, ax=ax_cm)
        ax_cm.set_title('SVC Confusion Matrix')
        ax_cm.set_ylabel('Ground Truth')
        ax_cm.set_xlabel('SVC Prediction')
        
        # Save CM
        buf_cm = io.BytesIO()
        fig_cm.savefig(buf_cm, format='png', bbox_inches='tight', facecolor='white')
        fig_cm.savefig(os.path.join(save_dir, "svc_confusion_matrix.png"), bbox_inches='tight', facecolor='white')
        plt.close(fig_cm)
        cm_b64 = base64.b64encode(buf_cm.getvalue()).decode()

        # Plot 2: Scatter (Log Scale)
        fig_sc, ax_sc = plt.subplots(figsize=(8, 6))
        colors = {'eMBB': '#e74c3c', 'uRLLC': '#f1c40f', 'mMTC': '#3498db'}
        for label in ALL_LABELS:
            subset = df[df['Predicted'] == label] # Use Predicted labels
            if len(subset) > 0:
                ax_sc.scatter(subset['Burstiness'], subset['Rate'], c=colors[label], label=label, alpha=0.6, edgecolors='w', linewidth=0.5, s=60)
        ax_sc.set_yscale('log')
        ax_sc.set_xlabel('Burstiness')
        ax_sc.set_ylabel('Rate (Mbps) [Log Scale]')
        ax_sc.set_title(f'SVC Classification Results (Support Vectors: {num_support_vectors})')
        ax_sc.legend()
        fig_sc.savefig(os.path.join(save_dir, "svc_distribution.png"), bbox_inches='tight', facecolor='white', dpi=300)
        plt.close(fig_sc)

        # Save Metrics
        pd.DataFrame({
            "Timestamp": [time.ctime()],
            "Model": ["SVC (RBF)"],
            "Accuracy": [acc],
            "F1_Score": [f1],
            "Support_Vectors": [num_support_vectors]
        }).to_csv(os.path.join(save_dir, "svc_metrics.csv"), index=False)
        
        TRAINING_STATE['result'] = {
            "throughput": f"{avg_thru:.2f} Mbps",
            "latency": f"{avg_lat:.2f} ms",
            "loss": f"{avg_loss:.3f}%",
            "accuracy": f"{acc*100:.1f}%",
            "f1_score": f"{f1*100:.1f}%",
            "extra_metric_name": "SUPPORT VECTORS", # Dynamic label for Frontend
            "extra_metric_val": num_support_vectors,
            "cm_image": cm_b64,
            "scatter": [{'x': float(r['Burstiness']), 'y': float(r['Rate']), 'c': 0 if r['Predicted']=='eMBB' else (1 if r['Predicted']=='uRLLC' else 2)} for i,r in df.sample(min(800, len(df))).iterrows()],
            "folder": save_dir
        }
        TRAINING_STATE['status'] = "Complete"
        TRAINING_STATE['eta'] = "0.0"

    except Exception as e:
        print(f"Error: {e}")
        TRAINING_STATE.update({"status": "Error", "error": str(e)})
    finally:
        TRAINING_STATE['busy'] = False

# --- FLASK ROUTES (Same as before) ---
@app.route('/')
def index(): return render_template('index_svc.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_DATA, SIMULATION_INDEX
    files = request.files.getlist('files')
    chunks = []
    for f in files:
        if f.filename=='': continue
        p = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(p)
        try:
            try: df = pd.read_csv(p, encoding='utf-8-sig')
            except: df = pd.read_csv(p, encoding='latin1')
            df.columns = df.columns.str.strip()
            rename_map = {}
            for col in df.columns:
                c = col.lower().strip()
                if any(x in c for x in ['rate', 'bps', 'spd']): rename_map[col] = 'Rate'
                elif any(x in c for x in ['size', 'sz', 'pkt']): rename_map[col] = 'Size'
            df = df.rename(columns=rename_map)
            if 'Rate' in df.columns: chunks.append(df)
        except: pass
    if chunks:
        CURRENT_DATA = pd.concat(chunks, ignore_index=True).fillna(0)
        SIMULATION_INDEX = 0
        return jsonify({"status": "success", "msg": f"Loaded {len(CURRENT_DATA)} flows."})
    return jsonify({"status": "error", "msg": "Invalid data."})

@app.route('/stream_data')
def stream():
    global SIMULATION_INDEX
    if CURRENT_DATA is None: return jsonify({"done": True})
    end = min(SIMULATION_INDEX+500, len(CURRENT_DATA))
    batch = CURRENT_DATA.iloc[SIMULATION_INDEX:end]
    SIMULATION_INDEX = end
    return jsonify({"done": SIMULATION_INDEX>=len(CURRENT_DATA), "rate": batch['Rate'].tolist()})

@app.route('/start_training', methods=['POST'])
def start():
    if CURRENT_DATA is not None and not TRAINING_STATE['busy']:
        TRAINING_STATE.update({"busy": True, "progress": 0, "status": "Starting SVC..."})
        threading.Thread(target=run_svc_task, args=(CURRENT_DATA.copy(), app.config['RESULTS_FOLDER']), daemon=True).start()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})

@app.route('/get_status')
def status(): return jsonify(TRAINING_STATE)

if __name__ == '__main__': app.run(debug=True, port=5001) # Runs on Port 5001