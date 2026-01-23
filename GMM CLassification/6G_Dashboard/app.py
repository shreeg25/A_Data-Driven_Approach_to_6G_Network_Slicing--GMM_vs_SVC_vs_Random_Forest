import os
import io
import time
import base64
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
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CURRENT_DATA = None
SIMULATION_INDEX = 0

@app.route('/')
def index():
    # Timestamp forces browser to reload fresh HTML
    return render_template('index.html', timestamp=time.time())

@app.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_DATA, SIMULATION_INDEX
    files = request.files.getlist('files')
    all_chunks = []
    
    print(">>> SERVER: Loading Data...")
    
    for file in files:
        if file.filename == '': continue
        try:
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            df.columns = df.columns.str.strip()
            
            # 1. Feature Matching
            if 'sMeanPktSz' in df.columns and 'SrcRate' in df.columns:
                df['Size'] = df['sMeanPktSz']
                df['Rate'] = df['SrcRate']
                
                # --- SYNTHETIC METRIC GENERATION ---
                # Since the file lacks these columns, we calculate them logically:
                
                # A. Throughput (Mbps) = Rate (pps) * Size (Bytes) * 8 / 1,000,000
                df['Throughput'] = (df['Rate'] * df['Size'] * 8) / 1e6
                
                # B. Latency (ms) = Inverse of Rate (Congestion Model) + Random Jitter
                # (Simple simulation for demo purposes)
                df['Latency'] = (1000 / (df['Rate'] + 10)) + np.random.uniform(1, 5, len(df))
                
                # C. Packet Loss (%) = Higher rate -> Higher probability of drop
                # Sigmoid-like function to simulate congestion at high rates
                df['Loss'] = 1 / (1 + np.exp(-(df['Rate'] - df['Rate'].mean())/1000)) * 5 

                # D. AUTO-LABELING (Ground Truth Generation)
                # We create "True Labels" based on 3GPP definitions so we can measure Accuracy
                # Slice 0 (IoT/mMTC): Small Size, Low Rate
                # Slice 1 (Video/eMBB): Large Size, High Rate
                # Slice 2 (Critical/uRLLC): Small Size, High Rate
                conditions = [
                    (df['Size'] > 500) & (df['Rate'] > 100), # eMBB
                    (df['Size'] < 500) & (df['Rate'] > 100), # uRLLC
                    (df['Rate'] <= 100)                      # mMTC
                ]
                choices = ['eMBB', 'uRLLC', 'mMTC']
                df['Label'] = np.select(conditions, choices, default='mMTC')
                
                all_chunks.append(df)
        except Exception as e:
            print(f"Error: {e}")

    if all_chunks:
        CURRENT_DATA = pd.concat(all_chunks, ignore_index=True)
        # Handle NaNs
        CURRENT_DATA = CURRENT_DATA.fillna(0)
        SIMULATION_INDEX = 0
        return jsonify({"status": "success", "message": f"Loaded {len(CURRENT_DATA)} flows with Generated Metrics."})
    else:
        return jsonify({"status": "error", "message": "Columns 'sMeanPktSz' and 'SrcRate' required."})

@app.route('/stream_data')
def stream_data():
    global CURRENT_DATA, SIMULATION_INDEX
    if CURRENT_DATA is None: return jsonify({"error": "No data"})
    
    # Send 500 packets per batch for speed
    end_idx = min(SIMULATION_INDEX + 500, len(CURRENT_DATA))
    batch = CURRENT_DATA.iloc[SIMULATION_INDEX:end_idx]
    SIMULATION_INDEX = end_idx
    
    return jsonify({
        "done": SIMULATION_INDEX >= len(CURRENT_DATA),
        "rate": batch['Rate'].tolist(),
        "processed": SIMULATION_INDEX
    })

@app.route('/train_model', methods=['POST'])
def train_model():
    print(">>> SERVER: Training AI Model...")
    global CURRENT_DATA
    if CURRENT_DATA is None: return jsonify({"status": "error"})

    try:
        start_t = time.time()
        
        # --- 1. AI TRAINING (Unsupervised GMM) ---
        scaler = StandardScaler()
        X = CURRENT_DATA[['Size', 'Rate']].values
        X_scaled = scaler.fit_transform(X)
        
        # Train GMM
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(X_scaled)
        CURRENT_DATA['Cluster'] = gmm.predict(X_scaled)
        
        # --- 2. MAP CLUSTERS TO LABELS ---
        # We align the GMM Clusters (0,1,2) to our Synthetic Labels ('eMBB', etc.)
        # to maximize accuracy (Hungarian Algorithm logic simplified)
        cluster_map = {}
        for c in range(3):
            subset = CURRENT_DATA[CURRENT_DATA['Cluster'] == c]
            if not subset.empty:
                cluster_map[c] = subset['Label'].mode()[0] # Map to most frequent label
            else:
                cluster_map[c] = 'Unknown'
        
        CURRENT_DATA['Predicted'] = CURRENT_DATA['Cluster'].map(cluster_map)

        # --- 3. CALCULATE METRICS ---
        # ML Metrics
        acc = accuracy_score(CURRENT_DATA['Label'], CURRENT_DATA['Predicted'])
        p, r, f1, _ = precision_recall_fscore_support(CURRENT_DATA['Label'], CURRENT_DATA['Predicted'], average='weighted', zero_division=0)
        
        # Network QoS Metrics
        avg_throughput = CURRENT_DATA['Throughput'].mean()
        avg_latency = CURRENT_DATA['Latency'].mean()
        avg_loss = CURRENT_DATA['Loss'].mean()
        
        duration = time.time() - start_t

        # --- 4. GENERATE VISUALS ---
        # Confusion Matrix
        fig = plt.figure(figsize=(6, 5))
        cm = confusion_matrix(CURRENT_DATA['Label'], CURRENT_DATA['Predicted'])
        labels = sorted(CURRENT_DATA['Label'].unique())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix (GMM vs Rules)')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        cm_image = base64.b64encode(buf.getvalue()).decode()

        # Scatter Plot Sample
        sample = CURRENT_DATA.sample(min(800, len(CURRENT_DATA)))
        scatter = [{'x': float(r['Size']), 'y': float(r['Rate']), 'c': int(r['Cluster'])} for i,r in sample.iterrows()]

        return jsonify({
            "status": "success",
            # Metrics
            "throughput": f"{avg_throughput:.2f} Mbps",
            "latency": f"{avg_latency:.2f} ms",
            "loss": f"{avg_loss:.2f} %",
            "accuracy": f"{acc*100:.1f}%",
            "f1_score": f"{f1*100:.1f}%",
            "time": f"{duration:.2f}s",
            # Images
            "cm_image": cm_image,
            "scatter": scatter
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)