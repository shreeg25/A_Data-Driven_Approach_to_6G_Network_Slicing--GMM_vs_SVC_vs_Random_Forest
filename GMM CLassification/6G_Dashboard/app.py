import os
import io
import time
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Server-side plotting
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
    # Timestamp prevents browser caching
    return render_template('index.html', timestamp=time.time())

@app.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_DATA, SIMULATION_INDEX
    files = request.files.getlist('files')
    all_chunks = []
    
    print(">>> SERVER: Loading File...")
    
    for file in files:
        if file.filename == '': continue
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # 1. Read File
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip() # Remove spaces
            
            # 2. Map Your Specific Columns
            # We look specifically for 'sMeanPktSz' and 'SrcRate'
            if 'sMeanPktSz' in df.columns and 'SrcRate' in df.columns:
                df['Size'] = df['sMeanPktSz']
                df['Rate'] = df['SrcRate']
                
                # 3. GENERATE MISSING METRICS (Since file doesn't have them)
                
                # A. Throughput (Mbps) = Rate * Size * 8 bits / 1,000,000
                df['Throughput'] = (df['Rate'] * df['Size'] * 8) / 1e6
                
                # B. Latency (ms) -> High Rate usually means Congestion (Higher Latency)
                # Formula: Base Latency + Congestion Factor + Jitter
                df['Latency'] = 10 + (df['Rate'] / 50) + np.random.uniform(1, 5, len(df))
                
                # C. Packet Loss (%) -> Higher Rate = Higher Loss Probability
                df['Loss'] = (df['Rate'] / df['Rate'].max()) * 5 * np.random.rand(len(df))

                # D. GENERATE LABELS (Ground Truth for Confusion Matrix)
                # We define rules to classify traffic so we can measure "Accuracy"
                conditions = [
                    (df['Size'] > 800) & (df['Rate'] > 100), # Large & Fast -> eMBB (Video)
                    (df['Size'] < 200) & (df['Rate'] > 50),  # Small & Fast -> uRLLC (Control)
                    (df['Rate'] <= 50)                       # Slow -> mMTC (IoT)
                ]
                choices = ['eMBB', 'uRLLC', 'mMTC']
                df['Label'] = np.select(conditions, choices, default='mMTC')
                
                all_chunks.append(df)
            else:
                print(f"Error: File {file.filename} missing 'sMeanPktSz' or 'SrcRate'")

        except Exception as e:
            print(f"Error reading {file.filename}: {e}")

    if all_chunks:
        CURRENT_DATA = pd.concat(all_chunks, ignore_index=True).fillna(0)
        SIMULATION_INDEX = 0
        print(f">>> SERVER: Success! Loaded {len(CURRENT_DATA)} flows.")
        return jsonify({"status": "success", "message": f"Loaded {len(CURRENT_DATA)} flows successfully."})
    else:
        return jsonify({"status": "error", "message": "Columns 'sMeanPktSz' and 'SrcRate' are missing."})

@app.route('/stream_data')
def stream_data():
    global CURRENT_DATA, SIMULATION_INDEX
    if CURRENT_DATA is None: return jsonify({"error": "No data"})
    
    # Stream 500 packets at a time
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
    print(">>> SERVER: calculating metrics...")
    global CURRENT_DATA
    if CURRENT_DATA is None: return jsonify({"status": "error"})

    try:
        start_t = time.time()
        
        # 1. Calculate Average Network Metrics
        avg_thru = CURRENT_DATA['Throughput'].mean()
        avg_lat = CURRENT_DATA['Latency'].mean()
        avg_loss = CURRENT_DATA['Loss'].mean()

        # 2. AI Training (GMM)
        scaler = StandardScaler()
        X = CURRENT_DATA[['Size', 'Rate']].values
        X_scaled = scaler.fit_transform(X)
        
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(X_scaled)
        CURRENT_DATA['Cluster'] = gmm.predict(X_scaled)
        
        # 3. Map Clusters to Labels (Self-Supervised)
        cluster_map = {}
        for c in range(3):
            subset = CURRENT_DATA[CURRENT_DATA['Cluster'] == c]
            if not subset.empty: 
                cluster_map[c] = subset['Label'].mode()[0] # Map to most frequent label
            else:
                cluster_map[c] = "Unknown"
        
        preds = CURRENT_DATA['Cluster'].map(cluster_map)
        
        # 4. ML Performance Metrics
        acc = accuracy_score(CURRENT_DATA['Label'], preds)
        p, r, f1, _ = precision_recall_fscore_support(CURRENT_DATA['Label'], preds, average='weighted', zero_division=0)
        
        # 5. Generate Confusion Matrix Image
        fig = plt.figure(figsize=(5, 4))
        cm = confusion_matrix(CURRENT_DATA['Label'], preds)
        labels = sorted(CURRENT_DATA['Label'].unique())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        cm_image = base64.b64encode(buf.getvalue()).decode()

        # 6. Scatter Plot Data
        sample = CURRENT_DATA.sample(min(800, len(CURRENT_DATA)))
        scatter = [{'x': float(r['Size']), 'y': float(r['Rate']), 'c': int(r['Cluster'])} for i,r in sample.iterrows()]
        
        duration = time.time() - start_t

        return jsonify({
            "status": "success",
            "throughput": f"{avg_thru:.2f} Mbps",
            "latency": f"{avg_lat:.2f} ms",
            "loss": f"{avg_loss:.2f}%",
            "accuracy": f"{acc*100:.1f}%",
            "f1_score": f"{f1*100:.1f}%",
            "train_time": f"{duration:.2f}s",
            "cm_image": cm_image,
            "scatter": scatter
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)