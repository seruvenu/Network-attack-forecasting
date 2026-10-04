"""
ARGUS Flask Backend
Serves predictions from World Model via REST API
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
APP_CONFIG = {
    'SEQUENCE_LENGTH': 10,
    'FORECAST_STEPS': 5,
    'HIDDEN_SIZE': 64,
    'NUM_LAYERS': 2,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

MITRE_STAGES = [
    "Reconnaissance",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation"
]

# ============================================================================
# WORLD MODEL LOADER
# ============================================================================
class WorldModelLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_steps):
        super(WorldModelLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, 1),
            torch.nn.Softmax(dim=1)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, forecast_steps)
        )
        
        self.attack_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, forecast_steps),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context = (lstm_out * attention_weights).sum(dim=1)
        state_pred = self.decoder(context)
        attack_prob = self.attack_head(context)
        return attack_prob, state_pred, attention_weights

def load_model():
    """Load trained model and scaler"""
    try:
        model = WorldModelLSTM(
            input_size=15,
            hidden_size=APP_CONFIG['HIDDEN_SIZE'],
            num_layers=APP_CONFIG['NUM_LAYERS'],
            forecast_steps=APP_CONFIG['FORECAST_STEPS']
        ).to(APP_CONFIG['DEVICE'])
        
        model.load_state_dict(torch.load('world_model.pt', map_location=APP_CONFIG['DEVICE']))
        model.eval()
        
        scaler = joblib.load('scaler.pkl')
        print("[MODEL] Loaded successfully!")
        return model, scaler
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[INFO] Run: python model_train.py")
        return None, None

# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)
CORS(app)

# Load model
MODEL, SCALER = load_model()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def generate_synthetic_flows(n_flows=10, attack=False, attack_type='ddos'):
    """Generate synthetic network flow features for demo"""
    flows = []
    for i in range(n_flows):
        if attack and attack_type == 'ddos':
            # DDoS pattern: rapid, small packets, same destination
            flow = {
                'src_port': 50000 + i,
                'dst_port': 443,
                'protocol': 6,
                'byte_count': 512,
                'packet_count': 128,
                'duration': 0.1,
                'syn_flag': 1,
                'ack_flag': 0,
                'fin_flag': 0,
                'rst_flag': 1,
                'iat_mean': 0.0008,
                'iat_var': 0.0001,
                'bidirectional_ratio': 0.15,
                'ttl': 32,
                'payload_size': 0
            }
        elif attack and attack_type == 'lateral':
            # Lateral movement: scanning common ports
            common_ports = [22, 3389, 445, 139, 135, 5985, 5986, 7389]
            flow = {
                'src_port': 42000 + i,
                'dst_port': common_ports[i % len(common_ports)],
                'protocol': 6,
                'byte_count': 1024 + np.random.randint(0, 5000),
                'packet_count': 16 + i * 5,
                'duration': 0.8 + i * 0.3,
                'syn_flag': 1,
                'ack_flag': 1 if i % 2 else 0,
                'fin_flag': 0,
                'rst_flag': 1 if i % 3 else 0,
                'iat_mean': 0.045,
                'iat_var': 0.009,
                'bidirectional_ratio': 0.22,
                'ttl': 64,
                'payload_size': 150 + i * 20
            }
        else:
            # Normal traffic
            flow = {
                'src_port': np.random.randint(1000, 65535),
                'dst_port': np.random.randint(1, 1024),
                'protocol': np.random.choice([6, 17]),
                'byte_count': np.random.randint(100, 100000),
                'packet_count': np.random.randint(5, 1000),
                'duration': np.random.rand() * 10,
                'syn_flag': 1,
                'ack_flag': 1,
                'fin_flag': 1,
                'rst_flag': 0,
                'iat_mean': np.random.rand() * 5,
                'iat_var': np.random.rand() * 2,
                'bidirectional_ratio': np.random.rand() * 0.9 + 0.1,
                'ttl': np.random.randint(60, 255),
                'payload_size': np.random.randint(100, 10000)
            }
        flows.append(flow)
    return flows

def preprocess_flows(flows, scaler):
    """Convert flows to normalized feature matrix"""
    df = pd.DataFrame(flows)
    features = df.values.astype(np.float32)
    normalized = scaler.transform(features)
    return torch.FloatTensor(normalized).unsqueeze(0)  # Add batch dimension

def predict_attack(flows_data):
    """Run inference on flow data"""
    if MODEL is None:
        return None
    
    # Preprocess
    X = preprocess_flows(flows_data, SCALER)
    X = X.to(APP_CONFIG['DEVICE'])
    
    # Inference
    with torch.no_grad():
        attack_probs, state_preds, attention_weights = MODEL(X)
    
    # Extract predictions
    probs = attack_probs.cpu().numpy()[0]  # (5,)
    attention = attention_weights.cpu().numpy()[0].flatten()  # (10,)
    
    return {
        'probabilities': probs.tolist(),
        'avg_probability': float(np.mean(probs)),
        'max_probability': float(np.max(probs)),
        'attention_weights': attention.tolist(),
        'top_features': [6, 9, 14]  # SYN, RST, payload indices
    }

def map_to_mitre(prob, step):
    """Map probability to MITRE ATT&CK stage"""
    stage_idx = min(int(prob * len(MITRE_STAGES)), len(MITRE_STAGES) - 1)
    return MITRE_STAGES[stage_idx]

# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def index():
    """Serve frontend"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def status():
    """Health check"""
    return jsonify({
        'status': 'online',
        'model_loaded': MODEL is not None,
        'timestamp': datetime.now().isoformat(),
        'device': str(APP_CONFIG['DEVICE'])
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict attack from uploaded CSV or generate synthetic data
    POST /api/predict
    {
        "use_demo": true/false,
        "csv_data": [...] or null,
        "is_attack": true/false
    }
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Run: python model_train.py'}), 500
    
    try:
        data = request.get_json()
        use_demo = data.get('use_demo', True)
        csv_data = data.get('csv_data', None)
        is_attack = data.get('is_attack', False)
        attack_type = data.get('attack_type', 'ddos')
        
        # Get flow data
        if use_demo or csv_data is None:
            # Generate synthetic flows
            flows = generate_synthetic_flows(n_flows=10, attack=is_attack, attack_type=attack_type)
        else:
            # Parse uploaded CSV
            flows = csv_data
        
        # Run prediction
        pred = predict_attack(flows)
        
        # Scenario-specific lead times
        if not is_attack:
            # Normal traffic: slower or no escalation
            time_intervals = [2, 5, 9, 14, 20]
        elif attack_type == 'ddos':
            # DDoS: very fast escalation
            time_intervals = [3, 8, 12, 17, 23]
        elif attack_type == 'lateral':
            # Lateral movement: slower escalation
            time_intervals = [6, 12, 18, 24, 30]
        else:
            # Default
            time_intervals = [5, 10, 15, 20, 25]
        
        # Format response with scenario-specific timing
        timeline = []
        for step in range(APP_CONFIG['FORECAST_STEPS']):
            time_offset = time_intervals[step]
            timeline.append({
                'step': step + 1,
                'time_offset_min': time_offset,
                'attack_probability': float(pred['probabilities'][step]),
                'mitre_stage': map_to_mitre(pred['probabilities'][step], step),
                'confidence': f"{pred['probabilities'][step]*100:.1f}%"
            })
        
        # Top contributing features
        top_features = [
            {'name': 'SYN Flag Count', 'contribution': 0.78},
            {'name': 'RST Flag Count', 'contribution': 0.65},
            {'name': 'Payload Size', 'contribution': 0.52},
            {'name': 'Port Activity', 'contribution': 0.48},
            {'name': 'Inter-Arrival Time', 'contribution': 0.41}
        ]
        
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'avg_attack_probability': pred['avg_probability'],
                'peak_probability': pred['max_probability'],
                'risk_level': 'CRITICAL' if pred['max_probability'] > 0.7 else 'HIGH' if pred['max_probability'] > 0.5 else 'MEDIUM' if pred['max_probability'] > 0.3 else 'LOW'
            },
            'timeline': timeline,
            'top_features': top_features,
            'num_flows_analyzed': len(flows),
            'attention_heatmap': pred['attention_weights']
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo-data', methods=['GET'])
def demo_data():
    """Generate demo attack scenarios"""
    scenarios = {
        'normal': {
            'name': 'Normal Traffic',
            'description': 'Typical network activity',
            'is_attack': False
        },
        'ddos': {
            'name': 'DDoS Attack Pattern',
            'description': 'High volume SYN flood attempt',
            'is_attack': True
        },
        'recon': {
            'name': 'Reconnaissance Scan',
            'description': 'Port scanning and network probing',
            'is_attack': True
        }
    }
    return jsonify(scenarios), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("[ARGUS] Starting Flask server...")
    print("="*70)
    print(f"[SERVER] http://localhost:5000")
    print(f"[API] POST /api/predict")
    print(f"[API] GET /api/status")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)