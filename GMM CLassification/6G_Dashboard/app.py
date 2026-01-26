import os
import io
import time
import base64
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # backend for non-GUI server
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
        
        if 'Jitter' in df.columns:
            df['Burstiness'] = df['Jitter']
            feature_set.append('Burstiness')
        else:
            # Burstiness Calculation
            df['Burstiness'] = df['Rate'].rolling(10).std().fillna(0) / (df['Rate'].rolling(10).mean().fillna(1) + 1e-9)
            feature_set.append('Burstiness')

        if 'Duration' in df.columns:
            df['Latency'] = df['Duration'] * 1000 
        else:
            # Latency Calculation
            df['Latency'] = (1 / (1000 - df['Rate'].clip(upper=990))) * 1000 + (df['Size'] * 0.01)

        if 'Loss' in df.columns:
            if df['Loss'].max() < 1.0: df['Loss'] = df['Loss'] * 100 
        else:
            # Loss Calculation
            util = df['Rate'] / 1000
            df['Loss'] = (util**4) * (1 + df['Burstiness']) * 100
            df['Loss'] = df['Loss'].clip(0, 100)

        # --- STEP 2: LABELING ---
        update(30, "Classifying Traffic Types...")
        
        conditions = [
            (df['Rate'] > 80) & (df['Burstiness'] > 0.1),       # eMBB
            (df['Latency'] < 10) & (df['Loss'] < 0.1),          # uRLLC
            (df['Rate'] <= 80)                                  # mMTC
        ]
        ALL_LABELS = ['eMBB', 'mMTC', 'uRLLC']
        df['Label'] = np.select(conditions, ALL_LABELS, default='mMTC')
        
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
            if not sub.empty: 
                cluster_map[c] = sub['Label'].mode()[0]
            else: 
                cluster_map[c] = "Unknown"
        preds = df['Cluster'].map(cluster_map)
        
        # --- RESULTS & SAVING ---
        update(90, "Finalizing & Saving...")
        
        acc = accuracy_score(df['Label'], preds)
        _, _, f1, _ = precision_recall_fscore_support(df['Label'], preds, average='weighted', zero_division=0)
        
        avg_thru = df['Rate'].mean()
        avg_lat = df['Latency'].mean()
        avg_loss = df['Loss'].mean()

        run_id = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(save_root, run_id)
        os.makedirs(save_dir, exist_ok=True)

        # PLOT 1: CONFUSION MATRIX
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        cm_data = confusion_matrix(df['Label'], preds, labels=ALL_LABELS)
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=ALL_LABELS, yticklabels=ALL_LABELS, ax=ax_cm)
        ax_cm.set_title('Confusion Matrix')
        fig_cm.savefig(os.path.join(save_dir, "confusion_matrix.png"), bbox_inches='tight', facecolor='white')
        
        buf_cm = io.BytesIO()
        fig_cm.savefig(buf_cm, format='png', bbox_inches='tight', facecolor='white')
        plt.close(fig_cm)
        cm_b64 = base64.b64encode(buf_cm.getvalue()).decode()

        # PLOT 2: SCATTER PLOT (LOG SCALE)
        fig_sc, ax_sc = plt.subplots(figsize=(8, 6))
        colors = {'eMBB': '#e74c3c', 'uRLLC': '#f1c40f', 'mMTC': '#3498db'}
        for label in ALL_LABELS:
            subset = df[df['Label'] == label]
            if len(subset) > 0:
                ax_sc.scatter(subset['Burstiness'], subset['Rate'], c=colors[label], label=label, alpha=0.6, edgecolors='w', linewidth=0.5, s=60)
        ax_sc.set_yscale('log') 
        ax_sc.set_xlabel('Burstiness')
        ax_sc.set_ylabel('Rate (Mbps) [Log Scale]')
        ax_sc.legend()
        fig_sc.savefig(os.path.join(save_dir, "slice_distribution_paper.png"), bbox_inches='tight', facecolor='white', dpi=300)
        plt.close(fig_sc)

        # SAVE CSV
        summary_data = {
            "Timestamp": [time.ctime()],
            "Overall_Throughput_Mbps": [avg_thru],
            "Overall_Latency_ms": [avg_lat],
            "Overall_Loss_Percent": [avg_loss],
            "Model_Accuracy": [acc],
            "F1_Score": [f1],
            "Iterations_to_Converge": [iterations],
            "Total_Flows": [rows]
        }
        pd.DataFrame(summary_data).to_csv(os.path.join(save_dir, "summary_metrics.csv"), index=False)
        
        TRAINING_STATE['result'] = {
            "throughput": f"{avg_thru:.2f} Mbps",
            "latency": f"{avg_lat:.2f} ms",
            "loss": f"{avg_loss:.3f}%",
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
                elif any(x in c for x in ['size', 'sz', 'pkt', 'len', 'byte']): rename_map[col] = 'Size'
                elif 'jit' in c or 'var' in c: rename_map[col] = 'Jitter'
                elif 'loss' in c or 'drop' in c: rename_map[col] = 'Loss'
                elif 'dur' in c or 'time' in c: rename_map[col] = 'Duration'
            df = df.rename(columns=rename_map)
            if 'Rate' in df.columns and 'Size' in df.columns: chunks.append(df)
        except Exception as e: print(f"Error: {e}")
        
    if chunks:
        CURRENT_DATA = pd.concat(chunks, ignore_index=True).fillna(0)
        SIMULATION_INDEX = 0 # <--- RESET SIMULATION
        return jsonify({"status": "success", "msg": f"Loaded {len(CURRENT_DATA)} flows."})
    return jsonify({"status": "error", "msg": "Invalid data."})

@app.route('/stream_data')
def stream():
    global SIMULATION_INDEX
    if CURRENT_DATA is None: return jsonify({"done": True})
    
    end = min(SIMULATION_INDEX+500, len(CURRENT_DATA))
    batch = CURRENT_DATA.iloc[SIMULATION_INDEX:end]
    SIMULATION_INDEX = end
    
    # Check if we reached the end
    is_done = (SIMULATION_INDEX >= len(CURRENT_DATA))
    return jsonify({"done": is_done, "rate": batch['Rate'].tolist()})

@app.route('/start_training', methods=['POST'])
def start():
    if CURRENT_DATA is None:
        return jsonify({"status": "error", "message": "No data loaded."})
    if TRAINING_STATE['busy']:
        return jsonify({"status": "error", "message": "Training already in progress."})

    TRAINING_STATE.update({"busy": True, "progress": 0, "status": "Starting..."})
    threading.Thread(target=run_training_task, args=(CURRENT_DATA.copy(), app.config['RESULTS_FOLDER']), daemon=True).start()
    return jsonify({"status": "success"})

@app.route('/get_status')
def status(): return jsonify(TRAINING_STATE)

if __name__ == '__main__': app.run(debug=True, port=5000) 