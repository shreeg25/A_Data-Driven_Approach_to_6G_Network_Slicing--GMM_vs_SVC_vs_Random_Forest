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

# Global Storage
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

        # --- STEP 2: FEATURES ---
        update(10, "Extracting Features...")
        
        # 2a. Burstiness
        if 'Jitter' in df.columns: 
            df['Burstiness'] = df['Jitter']
        else: 
            df['Burstiness'] = df['Rate'].rolling(10).std().fillna(0) / (df['Rate'].rolling(10).mean().fillna(1) + 1e-9)
        
        # 2b. Latency
        if 'Duration' in df.columns: 
            df['Latency'] = df['Duration'] * 1000 
        else: 
            df['Latency'] = (1 / (1000 - df['Rate'].clip(upper=990) + 1e-9)) * 1000 + (df['Size'] * 0.01)

        # 2c. Loss
        if 'Loss' in df.columns:
            if df['Loss'].max() < 1.0: df['Loss'] = df['Loss'] * 100 
        else:
            util = df['Rate'] / 1000
            df['Loss'] = (util**4) * (1 + df['Burstiness']) * 100
            df['Loss'] = df['Loss'].clip(0, 100)

        df = df.replace([np.inf, -np.inf], 0).fillna(0)

        # --- STEP 3: GROUND TRUTH (Clean Rules) ---
        update(25, "Labeling Data...")
        conditions = [
            (df['Rate'] > 80) & (df['Burstiness'] > 0.1),       # eMBB
            (df['Latency'] < 10) & (df['Loss'] < 0.1),          # uRLLC
            (df['Rate'] <= 80)                                  # mMTC
        ]
        ALL_LABELS = ['eMBB', 'mMTC', 'uRLLC']
        df['GroundTruth'] = np.select(conditions, ALL_LABELS, default='mMTC')
        
        # Select Features (Clean, No Noise)
        feature_cols = ['Rate', 'Size', 'Burstiness']

        # --- STEP 4: TRAIN ---
        update(50, "Training Forest...")
        X = df[feature_cols].values
        y = df['GroundTruth'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        importances = rf_model.feature_importances_
        try:
            burst_idx = feature_cols.index('Burstiness')
            burst_importance = importances[burst_idx]
        except: burst_importance = 0.0

        # --- STEP 5: EVALUATE ---
        update(80, "Validating...")
        all_preds = rf_model.predict(X_scaled)
        df['Predicted'] = all_preds
        
        acc = accuracy_score(df['GroundTruth'], all_preds)
        _, _, f1, _ = precision_recall_fscore_support(df['GroundTruth'], all_preds, average='weighted', zero_division=0)
        
        # --- STEP 6: SAVE VISUALS ---
        update(90, "Saving Results...")
        run_id = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(save_root, run_id)
        os.makedirs(save_dir, exist_ok=True)

        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        cm_data = confusion_matrix(df['GroundTruth'], df['Predicted'], labels=ALL_LABELS)
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Greens', xticklabels=ALL_LABELS, yticklabels=ALL_LABELS, ax=ax_cm)
        ax_cm.set_title('RF Confusion Matrix')
        buf_cm = io.BytesIO()
        fig_cm.savefig(buf_cm, format='png', bbox_inches='tight', facecolor='white')
        fig_cm.savefig(os.path.join(save_dir, "rf_confusion_matrix.png"), bbox_inches='tight', facecolor='white')
        plt.close(fig_cm)
        cm_b64 = base64.b64encode(buf_cm.getvalue()).decode()

        # Scatter Plot
        fig_sc, ax_sc = plt.subplots(figsize=(8, 6))
        colors = {'eMBB': '#e74c3c', 'uRLLC': '#f1c40f', 'mMTC': '#3498db'}
        for label in ALL_LABELS:
            subset = df[df['Predicted'] == label]
            if len(subset) > 0:
                ax_sc.scatter(subset['Burstiness'], subset['Rate'], c=colors[label], label=label, alpha=0.6, edgecolors='w', linewidth=0.5, s=60)
        ax_sc.set_yscale('log')
        ax_sc.set_xlabel('Burstiness')
        ax_sc.set_ylabel('Rate (Mbps)')
        ax_sc.set_title('RF Decision Boundary (Clean)')
        ax_sc.legend()
        fig_sc.savefig(os.path.join(save_dir, "rf_distribution.png"), bbox_inches='tight', facecolor='white', dpi=300)
        plt.close(fig_sc)

        # Summary CSV
        summary_data = {
            "Timestamp": [run_id],
            "Accuracy": [acc],
            "F1_Score": [f1],
            "Avg_Throughput_Mbps": [df['Rate'].mean()],
            "Avg_Latency_ms": [df['Latency'].mean()],
            "Avg_Loss_Pct": [df['Loss'].mean()],
            "Burst_Importance": [burst_importance]
        }
        pd.DataFrame(summary_data).to_csv(os.path.join(save_dir, "simulation_summary.csv"), index=False)

        TRAINING_STATE['result'] = {
            "throughput": f"{df['Rate'].mean():.2f} Mbps",
            "latency": f"{df['Latency'].mean():.2f} ms",
            "loss": f"{df['Loss'].mean():.3f}%",
            "accuracy": f"{acc*100:.1f}%",
            "f1_score": f"{f1*100:.1f}%",
            "extra_metric_name": "BURST IMPORTANCE",
            "extra_metric_val": f"{burst_importance:.3f}",
            "cm_image": cm_b64,
            "scatter": [{'x': float(r['Burstiness']), 'y': float(r['Rate']), 'c': 0 if r['Predicted']=='eMBB' else (1 if r['Predicted']=='uRLLC' else 2)} for i,r in df.sample(min(800, len(df))).iterrows()],
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
            # 1. ROBUST READ (Auto-detect separator)
            try: 
                df = pd.read_csv(p, sep=None, engine='python', encoding='utf-8-sig')
            except: 
                df = pd.read_csv(p, sep=None, engine='python', encoding='latin1')

            # 2. MAP COLUMNS
            df.columns = df.columns.str.strip() 
            rename_map = {}
            for col in df.columns:
                c_clean = col.lower()
                if 'srcrate' in c_clean: rename_map[col] = 'Rate'
                elif 'srcpktsz' in c_clean: rename_map[col] = 'Size'
                elif 'rate' in c_clean and 'Rate' not in rename_map.values(): rename_map[col] = 'Rate'
                elif 'size' in c_clean and 'Size' not in rename_map.values(): rename_map[col] = 'Size'

            df = df.rename(columns=rename_map)
            
            # 3. FORCE NUMERIC (Preserve Zeros)
            if 'Rate' in df.columns:
                df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce').fillna(0)
                
                if 'Size' in df.columns:
                    df['Size'] = pd.to_numeric(df['Size'], errors='coerce').fillna(0)
                else:
                    df['Size'] = 0.0

                chunks.append(df)
                
        except Exception as e:
            print(f" -> Failed to load {f.filename}: {e}")

    if chunks:
        CURRENT_DATA = pd.concat(chunks, ignore_index=True).fillna(0)
        SIMULATION_INDEX = 0
        print(f"--- Data Loaded: {len(CURRENT_DATA)} packets. Max Rate: {CURRENT_DATA['Rate'].max()} ---")
        return jsonify({"status": "success", "msg": f"Loaded {len(CURRENT_DATA)} packets."})
    
    return jsonify({"status": "error", "msg": "No valid numeric data found."})

@app.route('/stream_data')
def stream():
    global SIMULATION_INDEX, CURRENT_DATA
    if CURRENT_DATA is None or len(CURRENT_DATA) == 0: 
        return jsonify({"done": True, "rate": []})
    
    batch_size = 200
    end = SIMULATION_INDEX + batch_size
    
    # Infinite Loop Logic
    if end < len(CURRENT_DATA):
        batch = CURRENT_DATA.iloc[SIMULATION_INDEX:end]
        SIMULATION_INDEX = end
    else:
        remaining = CURRENT_DATA.iloc[SIMULATION_INDEX:]
        needed = batch_size - len(remaining)
        start_chunk = CURRENT_DATA.iloc[0:needed]
        batch = pd.concat([remaining, start_chunk])
        SIMULATION_INDEX = needed 
    
    # VISUALIZATION SCALING
    # If values are in Mbps (e.g., 500) but chart expects bits (500,000,000)
    raw_rates = batch['Rate'].tolist()
    avg_rate = np.mean(raw_rates) if len(raw_rates) > 0 else 0
    
    # Smart Scale: If average is small (<10k), multiply by 1M for visibility
    # This only affects the graph stream, not the backend analysis
    final_rates = raw_rates
    if avg_rate < 10000 and avg_rate > 0:
        final_rates = [x * 1000000 for x in raw_rates]
    
    return jsonify({"done": False, "rate": final_rates})

@app.route('/start_training', methods=['POST'])
def start():
    if CURRENT_DATA is None: return jsonify({"status": "error", "message": "Load data first!"})
    TRAINING_STATE.update({"busy": True, "progress": 0, "status": "Starting RF...", "result": None})
    threading.Thread(target=run_rf_task, args=(CURRENT_DATA.copy(), app.config['RESULTS_FOLDER']), daemon=True).start()
    return jsonify({"status": "success"})

@app.route('/get_status')
def status(): return jsonify(TRAINING_STATE)

if __name__ == '__main__': app.run(debug=True, port=5002)