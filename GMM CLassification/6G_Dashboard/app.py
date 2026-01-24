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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

CURRENT_DATA = None
SIMULATION_INDEX = 0
TRAINING_STATE = {"busy": False, "progress": 0, "status": "Idle", "eta": "0.0", "result": None}

def run_training_task(df, save_root):
    global TRAINING_STATE
    try:
        rows = len(df)
        est_time = max(5.0, 3.0 + (rows / 15000))
        start_t = time.time()
        
        def update(p, s):
            rem = max(0.0, est_time - (time.time() - start_t))
            TRAINING_STATE.update({"progress": p, "status": s, "eta": f"{rem:.1f}"})

        # --- STEP 1: FEATURE ENGINEERING ---
        update(10, "Processing Network Features...")
        
        feature_set = ['Rate', 'Size']
        
        # 1. Burstiness (Real or Calc)
        if 'Jitter' in df.columns:
            df['Burstiness'] = df['Jitter']
            feature_set.append('Burstiness')
        else:
            # Calculate Burstiness from Rate Variance
            df['Burstiness'] = df['Rate'].rolling(10).std().fillna(0) / (df['Rate'].rolling(10).mean().fillna(1) + 1e-9)
            feature_set.append('Burstiness')

        # 2. Latency (Real or Calc)
        if 'Duration' in df.columns:
            df['Latency'] = df['Duration'] * 1000 
        else:
            # Physics-based Latency: Serialization + Queueing
            # Queueing spikes as Rate -> 1000Mbps
            df['Latency'] = (1 / (1000 - df['Rate'].clip(upper=990))) * 1000 + (df['Size'] * 0.01)

        # 3. Loss (Real or Calc)
        if 'Loss' in df.columns:
            if df['Loss'].max() < 1.0: df['Loss'] = df['Loss'] * 100 
        else:
            # Loss probability increases with Burstiness
            util = df['Rate'] / 1000
            df['Loss'] = (util**4) * (1 + df['Burstiness']) * 100
            df['Loss'] = df['Loss'].clip(0, 100)

        # --- STEP 2: LABELING ---
        update(30, "Classifying Traffic Types...")
        
        # Ground Truth Logic
        conditions = [
            (df['Rate'] > 100) & (df['Burstiness'] > df['Burstiness'].mean()), # eMBB
            (df['Latency'] < 10) & (df['Loss'] < 0.1),                         # uRLLC
            (df['Rate'] < 50)                                                  # mMTC
        ]
        df['Label'] = np.select(conditions, ['eMBB', 'uRLLC', 'mMTC'], default='mMTC')
        time.sleep(1.0)

        # --- STEP 3: TRAINING ---
        update(50, f"Training GMM on {feature_set}...")
        
        X = df[feature_set].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(X_scaled)
        df['Cluster'] = gmm.predict(X_scaled)
        iterations = gmm.n_iter_
        
        update(70, "Mapping Clusters...")
        cluster_map = {}
        for c in range(3):
            sub = df[df['Cluster'] == c]
            if not sub.empty: cluster_map[c] = sub['Label'].mode()[0]
            else: cluster_map[c] = "Unknown"
        preds = df['Cluster'].map(cluster_map)
        
        # --- RESULTS ---
        update(90, "Finalizing...")
        acc = accuracy_score(df['Label'], preds)
        _, _, f1, _ = precision_recall_fscore_support(df['Label'], preds, average='weighted', zero_division=0)
        
        # Plot
        fig = plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(df['Label'], preds), annot=True, fmt='d', cmap='viridis')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        cm_b64 = base64.b64encode(buf.getvalue()).decode()
        
        # Save
        run_id = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(save_root, run_id)
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, "results.csv"), index=False)
        
        TRAINING_STATE['result'] = {
            "throughput": f"{df['Rate'].mean():.2f} Mbps",
            "latency": f"{df['Latency'].mean():.2f} ms",
            "loss": f"{df['Loss'].mean():.3f}%",
            "accuracy": f"{acc*100:.1f}%",
            "f1_score": f"{f1*100:.1f}%",
            "iterations": iterations,
            "cm_image": cm_b64,
            "scatter": [{'x': float(r['Burstiness']), 'y': float(r['Rate']), 'c': int(r['Cluster'])} for i,r in df.sample(min(800, len(df))).iterrows()],
            "folder": save_dir
        }
        TRAINING_STATE['status'] = "Complete"
        TRAINING_STATE['eta'] = "0.0"

    except Exception as e:
        print(f"Error: {e}")
        TRAINING_STATE.update({"status": "Error", "error": str(e)})
    finally:
        TRAINING_STATE['busy'] = False

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_DATA
    files = request.files.getlist('files')
    chunks = []
    
    print(">>> SERVER: Processing Upload...")
    
    for f in files:
        if f.filename=='': continue
        p = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(p)
        try:
            # 'utf-8-sig' handles Excel CSVs correctly
            try:
                df = pd.read_csv(p, encoding='utf-8-sig')
            except:
                df = pd.read_csv(p, encoding='latin1')
                
            df.columns = df.columns.str.strip()
            
            # --- ROBUST COLUMN MAPPING ---
            rename_map = {}
            for col in df.columns:
                c = col.lower().strip()
                
                # 1. RATE
                if any(x in c for x in ['rate', 'bps', 'spd']): 
                    rename_map[col] = 'Rate'
                
                # 2. SIZE (Crucial Fix: Added 'sz' and 'pkt' and 'len')
                elif any(x in c for x in ['size', 'sz', 'pkt', 'len', 'byte']): 
                    rename_map[col] = 'Size'
                
                # 3. JITTER/BURSTINESS
                elif 'jit' in c or 'var' in c: 
                    rename_map[col] = 'Jitter'
                
                # 4. LOSS
                elif 'loss' in c or 'drop' in c: 
                    rename_map[col] = 'Loss'
                
                # 5. DURATION
                elif 'dur' in c or 'time' in c: 
                    rename_map[col] = 'Duration'
            
            df = df.rename(columns=rename_map)
            
            # Check mandatory columns
            if 'Rate' in df.columns and 'Size' in df.columns:
                chunks.append(df)
            else:
                print(f">>> ERROR: {f.filename} columns not recognized. Found: {df.columns.tolist()}")
                
        except Exception as e:
            print(f">>> CRITICAL ERROR: {e}")
        
    if chunks:
        CURRENT_DATA = pd.concat(chunks, ignore_index=True).fillna(0)
        print(f">>> SUCCESS: Loaded {len(CURRENT_DATA)} rows.")
        return jsonify({"status": "success", "msg": f"Loaded {len(CURRENT_DATA)} flows."})
    return jsonify({"status": "error", "msg": "No valid data. Required: 'Rate' & 'Size/Sz/Pkt'"})

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
    if not TRAINING_STATE['busy'] and CURRENT_DATA is not None:
        TRAINING_STATE.update({"busy": True, "progress": 0, "status": "Starting..."})
        threading.Thread(target=run_training_task, args=(CURRENT_DATA.copy(), app.config['RESULTS_FOLDER']), daemon=True).start()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})

@app.route('/get_status')
def status(): return jsonify(TRAINING_STATE)

if __name__ == '__main__': app.run(debug=True, port=5000)