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

# --- GLOBAL STATE (Thread-Safe) ---
CURRENT_DATA = None
SIMULATION_INDEX = 0

# Training State Dictionary
TRAINING_STATE = {
    "busy": False,
    "progress": 0,
    "status": "Idle",
    "eta": 0,
    "result": None,
    "error": None
}

def calculate_eta(rows):
    """Estimates processing time based on dataset size."""
    # Rough benchmark: 100,000 rows takes ~3 seconds on average CPU
    return round((rows / 100000) * 3.5 + 2, 1)

def run_training_task(df, save_root):
    """Background Worker Thread for AI Training"""
    global TRAINING_STATE
    
    try:
        start_t = time.time()
        rows = len(df)
        TRAINING_STATE['progress'] = 5
        TRAINING_STATE['status'] = f"Initializing (Dataset: {rows} flows)..."
        
        # Create Run Folder
        run_id = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(save_root, run_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # --- PHASE 1: QOS CALCULATION (10-30%) ---
        TRAINING_STATE['progress'] = 10
        TRAINING_STATE['status'] = "Calculating QoS Metrics..."
        
        # Recalculate averages just to be sure
        avg_thru = df['Throughput'].mean()
        avg_lat = df['Latency'].mean()
        avg_loss = df['Loss'].mean()
        time.sleep(0.5) # UI smoothing
        
        # --- PHASE 2: AI TRAINING (30-70%) ---
        TRAINING_STATE['progress'] = 30
        TRAINING_STATE['status'] = "Training GMM Neural Network..."
        
        scaler = StandardScaler()
        X = df[['Size', 'Rate']].values
        X_scaled = scaler.fit_transform(X)
        
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(X_scaled)
        df['Cluster'] = gmm.predict(X_scaled)
        
        TRAINING_STATE['progress'] = 60
        TRAINING_STATE['status'] = "Classifying Slices..."
        
        # Map Clusters to Labels (Self-Supervised Logic)
        cluster_map = {}
        for c in range(3):
            subset = df[df['Cluster'] == c]
            if not subset.empty:
                cluster_map[c] = subset['Label'].mode()[0]
            else:
                cluster_map[c] = "Unknown"
        
        preds = df['Cluster'].map(cluster_map)
        df['Predicted'] = preds
        
        # --- PHASE 3: EVALUATION (70-90%) ---
        TRAINING_STATE['progress'] = 75
        TRAINING_STATE['status'] = "Calculating Accuracy Scores..."
        
        acc = accuracy_score(df['Label'], preds)
        p, r, f1, _ = precision_recall_fscore_support(df['Label'], preds, average='weighted', zero_division=0)
        
        # Generate Confusion Matrix
        fig = plt.figure(figsize=(6, 5))
        cm = confusion_matrix(df['Label'], preds)
        labels = sorted(df['Label'].unique())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save Plot
        img_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(img_path)
        
        # Encode for Web
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        cm_image = base64.b64encode(buf.getvalue()).decode()
        
        # --- PHASE 4: SAVING (90-100%) ---
        TRAINING_STATE['progress'] = 90
        TRAINING_STATE['status'] = "Archiving Results..."
        
        # Save CSV
        df.to_csv(os.path.join(save_dir, "processed_data.csv"), index=False)
        
        # Save Report
        with open(os.path.join(save_dir, "analysis_report.txt"), "w") as f:
            f.write(f"6G AI CONTROLLER - ANALYSIS REPORT\n")
            f.write(f"==================================\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Dataset Size: {rows} flows\n\n")
            f.write(f"[NETWORK PERFORMANCE]\n")
            f.write(f"Avg Throughput: {avg_thru:.2f} Mbps\n")
            f.write(f"Avg Latency:    {avg_lat:.2f} ms\n")
            f.write(f"Avg Loss:       {avg_loss:.2f} %\n\n")
            f.write(f"[AI MODEL PERFORMANCE]\n")
            f.write(f"Accuracy: {acc*100:.2f}%\n")
            f.write(f"F1-Score: {f1*100:.2f}%\n")
        
        # Generate Scatter Sample
        sample = df.sample(min(800, len(df)))
        scatter = [{'x': float(r['Size']), 'y': float(r['Rate']), 'c': int(r['Cluster'])} for i,r in sample.iterrows()]
        
        duration = time.time() - start_t
        
        # --- FINALIZE ---
        TRAINING_STATE['result'] = {
            "throughput": f"{avg_thru:.2f} Mbps",
            "latency": f"{avg_lat:.2f} ms",
            "loss": f"{avg_loss:.2f}%",
            "accuracy": f"{acc*100:.1f}%",
            "f1_score": f"{f1*100:.1f}%",
            "train_time": f"{duration:.2f}s",
            "cm_image": cm_image,
            "scatter": scatter,
            "folder": save_dir
        }
        TRAINING_STATE['progress'] = 100
        TRAINING_STATE['status'] = "Complete"
        
    except Exception as e:
        print(f"TRAINING ERROR: {e}")
        TRAINING_STATE['error'] = str(e)
        TRAINING_STATE['status'] = "Error"
    
    finally:
        TRAINING_STATE['busy'] = False

@app.route('/')
def index():
    return render_template('index.html', timestamp=time.time())

@app.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_DATA, SIMULATION_INDEX
    files = request.files.getlist('files')
    all_chunks = []
    
    for file in files:
        if file.filename == '': continue
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            
            # AUTO-CORRECT & GENERATE COLUMNS
            if 'sMeanPktSz' in df.columns and 'SrcRate' in df.columns:
                df['Size'] = df['sMeanPktSz']
                df['Rate'] = df['SrcRate']
                
                # Logic: Throughput = Rate * Size * 8 / 1e6
                df['Throughput'] = (df['Rate'] * df['Size'] * 8) / 1e6
                # Logic: Latency increases with Rate
                df['Latency'] = 10 + (df['Rate'] / 50) + np.random.uniform(0, 5, len(df))
                # Logic: Loss increases with Rate saturation
                df['Loss'] = (df['Rate'] / (df['Rate'].max()+1)) * 5 * np.random.rand(len(df))

                # Logic: Labels based on 3GPP characteristics
                conditions = [
                    (df['Size'] > 800) & (df['Rate'] > 100), # eMBB
                    (df['Size'] < 200) & (df['Rate'] > 50),  # uRLLC
                    (df['Rate'] <= 50)                       # mMTC
                ]
                df['Label'] = np.select(conditions, ['eMBB', 'uRLLC', 'mMTC'], default='mMTC')
                
                all_chunks.append(df)
        except Exception as e:
            print(f"Error: {e}")

    if all_chunks:
        CURRENT_DATA = pd.concat(all_chunks, ignore_index=True).fillna(0)
        SIMULATION_INDEX = 0
        return jsonify({"status": "success", "message": f"Loaded {len(CURRENT_DATA)} flows."})
    else:
        return jsonify({"status": "error", "message": "Invalid CSV. Need 'sMeanPktSz' & 'SrcRate'."})

@app.route('/stream_data')
def stream_data():
    global CURRENT_DATA, SIMULATION_INDEX
    if CURRENT_DATA is None: return jsonify({"error": "No data"})
    
    end_idx = min(SIMULATION_INDEX + 500, len(CURRENT_DATA))
    batch = CURRENT_DATA.iloc[SIMULATION_INDEX:end_idx]
    SIMULATION_INDEX = end_idx
    
    return jsonify({
        "done": SIMULATION_INDEX >= len(CURRENT_DATA),
        "rate": batch['Rate'].tolist(),
        "processed": SIMULATION_INDEX
    })

@app.route('/start_training', methods=['POST'])
def start_training():
    global TRAINING_STATE, CURRENT_DATA
    
    if CURRENT_DATA is None:
        return jsonify({"status": "error", "message": "No data loaded"})
        
    if TRAINING_STATE['busy']:
        return jsonify({"status": "error", "message": "Training already in progress"})
    
    # Reset State
    TRAINING_STATE = {
        "busy": True,
        "progress": 0,
        "status": "Starting...",
        "eta": calculate_eta(len(CURRENT_DATA)),
        "result": None,
        "error": None
    }
    
    # Start Background Thread
    thread = threading.Thread(target=run_training_task, args=(CURRENT_DATA.copy(), app.config['RESULTS_FOLDER']))
    thread.daemon = True # Thread dies if server dies
    thread.start()
    
    return jsonify({"status": "success", "message": "Background task started"})

@app.route('/get_status')
def get_status_route():
    return jsonify(TRAINING_STATE)

if __name__ == '__main__':
    app.run(debug=True, port=5000)