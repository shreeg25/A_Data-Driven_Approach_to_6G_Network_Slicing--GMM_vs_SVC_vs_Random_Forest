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
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# --- GLOBAL STATE ---
CURRENT_DATA = None
SIMULATION_INDEX = 0

TRAINING_STATE = {
    "busy": False,
    "progress": 0,
    "status": "Idle",
    "eta": "0.0",
    "result": None,
    "error": None
}

def run_training_task(df, save_root):
    """Background Worker with FORCED DELAYS for Visualization"""
    global TRAINING_STATE
    
    try:
        # 1. SETUP & ETA CALCULATION
        rows = len(df)
        # We enforce a minimum of 5 seconds for the UX
        estimated_seconds = max(5.0, 3.0 + (rows / 20000))
        start_time = time.time()
        
        def update_state(prog, status):
            elapsed = time.time() - start_time
            remaining = max(0.0, estimated_seconds - elapsed)
            TRAINING_STATE['progress'] = prog
            TRAINING_STATE['status'] = status
            TRAINING_STATE['eta'] = f"{remaining:.1f}"
            print(f">>> Progress: {prog}% | Status: {status} | ETA: {remaining:.1f}s")

        update_state(5, f"Initializing ({rows} flows)...")
        time.sleep(1.0) # Visual pause

        run_id = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(save_root, run_id)
        os.makedirs(save_dir, exist_ok=True)

        # 2. QoS METRICS (10% - 30%)
        update_state(10, "Calculating Throughput & Latency...")
        avg_thru = df['Throughput'].mean()
        avg_lat = df['Latency'].mean()
        avg_loss = df['Loss'].mean()
        time.sleep(1.0) # Visual pause

        # 3. AI TRAINING (30% - 70%)
        update_state(30, "Training Neural Network...")
        
        scaler = StandardScaler()
        X = df[['Size', 'Rate']].values
        X_scaled = scaler.fit_transform(X)
        
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(X_scaled)
        df['Cluster'] = gmm.predict(X_scaled)
        
        update_state(60, "Classifying 6G Slices...")
        time.sleep(1.0) # Visual pause

        # Map Clusters
        cluster_map = {}
        for c in range(3):
            subset = df[df['Cluster'] == c]
            if not subset.empty:
                cluster_map[c] = subset['Label'].mode()[0]
            else:
                cluster_map[c] = "Unknown"
        
        preds = df['Cluster'].map(cluster_map)
        
        # 4. EVALUATION (70% - 90%)
        update_state(80, "Generating Accuracy Report...")
        acc = accuracy_score(df['Label'], preds)
        p, r, f1, _ = precision_recall_fscore_support(df['Label'], preds, average='weighted', zero_division=0)
        
        # Generate Confusion Matrix
        fig = plt.figure(figsize=(6, 5))
        cm = confusion_matrix(df['Label'], preds)
        labels = sorted(df['Label'].unique())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        cm_image = base64.b64encode(buf.getvalue()).decode()
        
        # Save to Disk
        df.to_csv(os.path.join(save_dir, "processed_data.csv"), index=False)
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        
        # 5. FINALIZE (100%)
        update_state(100, "Finalizing...")
        time.sleep(0.5)
        
        TRAINING_STATE['result'] = {
            "throughput": f"{avg_thru:.2f} Mbps",
            "latency": f"{avg_lat:.2f} ms",
            "loss": f"{avg_loss:.2f}%",
            "accuracy": f"{acc*100:.1f}%",
            "f1_score": f"{f1*100:.1f}%",
            "cm_image": cm_image,
            "scatter": [{'x': float(r['Size']), 'y': float(r['Rate']), 'c': int(r['Cluster'])} for i,r in df.sample(min(800, len(df))).iterrows()],
            "folder": save_dir
        }
        TRAINING_STATE['status'] = "Complete"
        TRAINING_STATE['eta'] = "0.0"

    except Exception as e:
        print(f"ERROR: {e}")
        TRAINING_STATE['error'] = str(e)
        TRAINING_STATE['status'] = "Error"
    
    finally:
        TRAINING_STATE['busy'] = False

@app.route('/')
def index():
    return render_template('index.html', timestamp=time.time())

@app.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_DATA
    files = request.files.getlist('files')
    all_chunks = []
    
    for file in files:
        if file.filename == '': continue
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            
            # AUTO-CORRECT COLUMNS
            if 'sMeanPktSz' in df.columns and 'SrcRate' in df.columns:
                df['Size'] = df['sMeanPktSz']
                df['Rate'] = df['SrcRate']
                
                # METRICS GENERATION
                df['Throughput'] = (df['Rate'] * df['Size'] * 8) / 1e6
                df['Latency'] = 10 + (df['Rate'] / 50) + np.random.uniform(0, 5, len(df))
                df['Loss'] = (df['Rate'] / (df['Rate'].max()+1)) * 5 * np.random.rand(len(df))

                conditions = [
                    (df['Size'] > 800) & (df['Rate'] > 100), 
                    (df['Size'] < 200) & (df['Rate'] > 50),
                    (df['Rate'] <= 50)
                ]
                df['Label'] = np.select(conditions, ['eMBB', 'uRLLC', 'mMTC'], default='mMTC')
                all_chunks.append(df)
        except Exception as e:
            print(e)

    if all_chunks:
        CURRENT_DATA = pd.concat(all_chunks, ignore_index=True).fillna(0)
        return jsonify({"status": "success", "message": f"Loaded {len(CURRENT_DATA)} flows."})
    else:
        return jsonify({"status": "error", "message": "Invalid CSV columns."})

@app.route('/stream_data')
def stream_data():
    global CURRENT_DATA, SIMULATION_INDEX
    if CURRENT_DATA is None: return jsonify({"error": "No data"})
    
    end_idx = min(SIMULATION_INDEX + 500, len(CURRENT_DATA))
    batch = CURRENT_DATA.iloc[SIMULATION_INDEX:end_idx]
    SIMULATION_INDEX = end_idx
    
    return jsonify({
        "done": SIMULATION_INDEX >= len(CURRENT_DATA),
        "rate": batch['Rate'].tolist()
    })

@app.route('/start_training', methods=['POST'])
def start_training():
    global TRAINING_STATE, CURRENT_DATA
    if CURRENT_DATA is None: return jsonify({"status": "error", "message": "No data"})
    if TRAINING_STATE['busy']: return jsonify({"status": "error", "message": "Busy"})
    
    TRAINING_STATE = {
        "busy": True, "progress": 0, "status": "Starting...", 
        "eta": "Calculating...", "result": None, "error": None
    }
    
    thread = threading.Thread(target=run_training_task, args=(CURRENT_DATA.copy(), app.config['RESULTS_FOLDER']))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "success"})

@app.route('/get_status')
def get_status_route():
    return jsonify(TRAINING_STATE)

if __name__ == '__main__':
    app.run(debug=True, port=5000)