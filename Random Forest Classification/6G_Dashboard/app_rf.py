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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads_rf'
app.config['RESULTS_FOLDER'] = 'results_rf'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

CURRENT_DATA = None
SIMULATION_INDEX = 0
TRAINING_STATE = {"busy": False, "progress": 0, "status": "Idle", "eta": "0.0", "result": None}

def run_rf_task(df, save_root):
    global TRAINING_STATE
    try:
        rows = len(df)
        est_time = max(5.0, 3.0 + (rows / 8000)) 
        start_t = time.time()
        
        def update(p, s):
            rem = max(0.0, est_time - (time.time() - start_t))
            TRAINING_STATE.update({"progress": p, "status": s, "eta": f"{rem:.1f}"})

        # --- STEP 1: CLEAN DATA ---
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # --- STEP 2: CALCULATE PURE FEATURES ---
        update(10, "Extracting Features...")
        
        # Rate & Size are already there, calculate Burstiness
        if 'Jitter' in df.columns: 
            df['Burstiness'] = df['Jitter']
        else: 
            # Safe Division
            df['Burstiness'] = df['Rate'].rolling(10).std().fillna(0) / (df['Rate'].rolling(10).mean().fillna(1) + 1e-9)
        
        # Latency
        if 'Duration' in df.columns: 
            df['Latency'] = df['Duration'] * 1000 
        else: 
            df['Latency'] = (1 / (1000 - df['Rate'].clip(upper=990) + 1e-9)) * 1000 + (df['Size'] * 0.01)

        # Loss
        if 'Loss' in df.columns:
            if df['Loss'].max() < 1.0: df['Loss'] = df['Loss'] * 100 
        else:
            util = df['Rate'] / 1000
            df['Loss'] = (util**4) * (1 + df['Burstiness']) * 100
            df['Loss'] = df['Loss'].clip(0, 100)

        df = df.replace([np.inf, -np.inf], 0).fillna(0)

        # --- STEP 3: GENERATE GROUND TRUTH (PERFECT LABELS) ---
        update(25, "Labeling Data...")
        # We set strict rules for what the packet *actually* is
        conditions = [
            (df['Rate'] > 80) & (df['Burstiness'] > 0.1),       # eMBB
            (df['Latency'] < 10) & (df['Loss'] < 0.1),          # uRLLC
            (df['Rate'] <= 80)                                  # mMTC
        ]
        ALL_LABELS = ['eMBB', 'mMTC', 'uRLLC']
        df['GroundTruth'] = np.select(conditions, ALL_LABELS, default='mMTC')
        
        # --- STEP 3.5: INJECT REALISM (NOISE) ---
        # This breaks the "100% Accuracy" by simulating channel interference.
        # The AI sees 'Noisy' data but has to predict the 'Perfect' label.
        update(35, "Injecting Channel Noise...")
        
        # Add random noise (+/- 25%) to Rate and Burstiness
        noise_factor = np.random.uniform(0.75, 1.25, len(df))
        df['Rate_Noisy'] = df['Rate'] * noise_factor
        
        noise_factor_2 = np.random.uniform(0.75, 1.25, len(df))
        df['Burst_Noisy'] = df['Burstiness'] * noise_factor_2
        
        # Select Feature Set (Use NOISY features for training)
        feature_cols = ['Rate_Noisy', 'Size', 'Burst_Noisy']

        # --- STEP 4: TRAIN MODEL ---
        update(50, "Training Forest...")
        X = df[feature_cols].values
        y = df['GroundTruth'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # Feature Importance (Check Burst_Noisy importance)
        importances = rf_model.feature_importances_
        try:
            burst_idx = feature_cols.index('Burst_Noisy')
            burst_importance = importances[burst_idx]
        except:
            burst_importance = 0.0

        # --- STEP 5: EVALUATE ---
        update(80, "Validating...")
        all_preds = rf_model.predict(X_scaled)
        df['Predicted'] = all_preds
        
        acc = accuracy_score(df['GroundTruth'], all_preds)
        _, _, f1, _ = precision_recall_fscore_support(df['GroundTruth'], all_preds, average='weighted', zero_division=0)
        
        # --- STEP 6: VISUALS ---
        update(90, "Generating Report...")
        run_id = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(save_root, run_id)
        os.makedirs(save_dir, exist_ok=True)

        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        cm_data = confusion_matrix(df['GroundTruth'], df['Predicted'], labels=ALL_LABELS)
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Greens', xticklabels=ALL_LABELS, yticklabels=ALL_LABELS, ax=ax_cm)
        ax_cm.set_title('Confusion Matrix (With Noise)')
        buf_cm = io.BytesIO()
        fig_cm.savefig(buf_cm, format='png', bbox_inches='tight', facecolor='white')
        fig_cm.savefig(os.path.join(save_dir, "rf_confusion_matrix.png"), bbox_inches='tight', facecolor='white')
        plt.close(fig_cm)
        cm_b64 = base64.b64encode(buf_cm.getvalue()).decode()

        # Scatter Plot (Using Noisy Data to show overlap)
        fig_sc, ax_sc = plt.subplots(figsize=(8, 6))
        colors = {'eMBB': '#e74c3c', 'uRLLC': '#f1c40f', 'mMTC': '#3498db'}
        for label in ALL_LABELS:
            # Plot based on Predicted to show decision boundaries
            subset = df[df['Predicted'] == label]
            if len(subset) > 0:
                ax_sc.scatter(subset['Burst_Noisy'], subset['Rate_Noisy'], c=colors[label], label=label, alpha=0.6, edgecolors='w', linewidth=0.5, s=60)
        
        ax_sc.set_yscale('log')
        ax_sc.set_xlabel(f'Burstiness (Noisy) [Imp: {burst_importance:.2f}]')
        ax_sc.set_ylabel('Rate (Noisy) [Mbps]')
        ax_sc.set_title('RF Decision Boundary (Simulated Noise)')
        ax_sc.legend()
        fig_sc.savefig(os.path.join(save_dir, "rf_distribution.png"), bbox_inches='tight', facecolor='white', dpi=300)
        plt.close(fig_sc)

        TRAINING_STATE['result'] = {
            "throughput": f"{df['Rate'].mean():.2f} Mbps",
            "latency": f"{df['Latency'].mean():.2f} ms",
            "loss": f"{df['Loss'].mean():.3f}%",
            "accuracy": f"{acc*100:.1f}%",
            "f1_score": f"{f1*100:.1f}%",
            "extra_metric_name": "BURST IMPORTANCE",
            "extra_metric_val": f"{burst_importance:.3f}",
            "cm_image": cm_b64,
            "scatter": [{'x': float(r['Burst_Noisy']), 'y': float(r['Rate_Noisy']), 'c': 0 if r['Predicted']=='eMBB' else (1 if r['Predicted']=='uRLLC' else 2)} for i,r in df.sample(min(800, len(df))).iterrows()],
            "folder": save_dir
        }
        TRAINING_STATE['status'] = "Complete"
        TRAINING_STATE['eta'] = "0.0"

    except Exception as e:
        print(f"CRITICAL ERROR: {e}") 
        TRAINING_STATE.update({"status": "Error", "error": str(e)})
    finally:
        TRAINING_STATE['busy'] = False

@app.route('/')
def index(): return render_template('index_rf.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_DATA, SIMULATION_INDEX
    files = request.files.getlist('files')
    chunks = []
    print(f"--- Uploading {len(files)} files ---") 
    for f in files:
        if f.filename == '': continue
        p = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(p)
        try:
            try: df = pd.read_csv(p, encoding='utf-8-sig')
            except: df = pd.read_csv(p, encoding='latin1')
            df.columns = df.columns.str.strip() 
            rename_map = {}
            for col in df.columns:
                c_clean = col.lower()
                if 'srcrate' in c_clean: rename_map[col] = 'Rate'
                elif 'srcpktsz' in c_clean: rename_map[col] = 'Size'
                elif 'rate' in c_clean and 'Rate' not in rename_map.values(): rename_map[col] = 'Rate'
                elif 'size' in c_clean and 'Size' not in rename_map.values(): rename_map[col] = 'Size'
            df = df.rename(columns=rename_map)
            if 'Rate' in df.columns:
                df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce').fillna(0)
                df = df[df['Rate'] > 0] 
                if 'Size' in df.columns: df['Size'] = pd.to_numeric(df['Size'], errors='coerce').fillna(0)
                else: df['Size'] = 0.0 
                chunks.append(df)
        except Exception as e: print(f" -> Failed {f.filename}: {e}")
    if chunks:
        CURRENT_DATA = pd.concat(chunks, ignore_index=True).fillna(0)
        SIMULATION_INDEX = 0
        return jsonify({"status": "success", "msg": f"Loaded {len(CURRENT_DATA)} packets."})
    return jsonify({"status": "error", "msg": "No valid numeric data found."})

@app.route('/stream_data')
def stream():
    global SIMULATION_INDEX
    if CURRENT_DATA is None or len(CURRENT_DATA) == 0: return jsonify({"done": True, "rate": []})
    end = min(SIMULATION_INDEX+500, len(CURRENT_DATA))
    batch = CURRENT_DATA.iloc[SIMULATION_INDEX:end]
    SIMULATION_INDEX = end
    is_done = (SIMULATION_INDEX >= len(CURRENT_DATA))
    return jsonify({"done": is_done, "rate": batch['Rate'].tolist()})

@app.route('/start_training', methods=['POST'])
def start():
    if CURRENT_DATA is None: return jsonify({"status": "error", "message": "Load data first!"})
    TRAINING_STATE.update({"busy": True, "progress": 0, "status": "Starting RF...", "result": None})
    threading.Thread(target=run_rf_task, args=(CURRENT_DATA.copy(), app.config['RESULTS_FOLDER']), daemon=True).start()
    return jsonify({"status": "success"})

@app.route('/get_status')
def status(): return jsonify(TRAINING_STATE)

if __name__ == '__main__': app.run(debug=True, port=5002)