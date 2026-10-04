"""Generate labeled test traffic for ARGUS. Run: python make_test_data.py
Every file is 600 seconds of flows. Columns are a common flow schema:
timestamp, src_ip, dst_ip, src_port, dst_port, protocol, duration_s,
fwd_packets, bwd_packets, fwd_bytes, bwd_bytes, syn_flags, ack_flags, rst_flags, label
"""
import numpy as np, pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(26153)
T0 = datetime(2026, 9, 29, 10, 0, 0)
CLIENTS = [f"10.0.1.{i}" for i in range(10, 60)]
SERVERS = ["10.0.0.5", "10.0.0.6", "10.0.0.7", "10.0.0.20", "10.0.0.53"]
COLS = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "duration_s",
        "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes", "syn_flags", "ack_flags", "rst_flags", "label"]

def rows(n, t_start, t_end, src, dst, dport, proto="TCP", label="benign", dur=(0.05, 2.0),
         fwd=(4, 25), bwd=(4, 40), fbytes=(300, 4000), bbytes=(500, 30000), syn=1, rst=0, ack=None):
    """n flows spread uniformly in [t_start, t_end]; src/dst/dport may be lists (sampled per flow)."""
    pick = lambda x: x if callable(x) else (lambda: x[rng.integers(len(x))] if isinstance(x, (list, tuple)) else x)
    s, d, p = pick(src), pick(dst), pick(dport)
    out = []
    for t in np.sort(rng.uniform(t_start, t_end, n)):
        fp = int(rng.integers(*fwd)) if fwd[1] > fwd[0] else fwd[0]
        bp = int(rng.integers(*bwd)) if bwd[1] > bwd[0] else bwd[0]
        out.append([(T0 + timedelta(seconds=float(t))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    s(), d(), int(rng.integers(1025, 65535)), p(), proto,
                    round(float(rng.uniform(*dur)), 3), fp, bp,
                    int(rng.integers(*fbytes)), int(rng.integers(*bbytes)) if bp else 0,
                    syn, (fp + bp - 2) if ack is None else ack, rst, label])
    return out

def background(n=3000):
    web = rows(int(n * .7), 0, 600, CLIENTS, SERVERS[:3], [443, 80, 443])
    dns = rows(int(n * .2), 0, 600, CLIENTS, "10.0.0.53", 53, "UDP", dur=(0.01, 0.2),
               fwd=(1, 3), bwd=(1, 3), fbytes=(60, 120), bbytes=(100, 400), syn=0, ack=0)
    ssh = rows(int(n * .1), 0, 600, CLIENTS[:8], SERVERS[3], 22, dur=(5, 120), fwd=(30, 400), bwd=(30, 400),
               fbytes=(2000, 40000), bbytes=(2000, 40000))
    return web + dns + ssh

def save(name, data):
    df = pd.DataFrame(data, columns=COLS).sort_values("timestamp")
    df.to_csv(name, index=False)
    print(f"{name:32s} {len(df):6d} flows   labels: {dict(df.label.value_counts())}")

# 1. Easy normal
save("1_normal_baseline.csv", background(4000))

# 2. Easy attack: SYN flood from spoofed sources starting at t=300
spoof = lambda: f"{rng.integers(11, 220)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
save("2_syn_flood_easy.csv", background(2000) + rows(7000, 300, 600, spoof, "10.0.0.5", 80, label="ddos_syn",
     dur=(0.0, 0.01), fwd=(1, 3), bwd=(0, 0), fbytes=(40, 80), bbytes=(0, 1), syn=1, ack=0))

# 3. HARD attack: slowloris. Few flows, tiny bytes, very long connections; volume barely moves
atk = [f"172.16.{rng.integers(0, 5)}.{i}" for i in range(20, 60)]
save("3_slowloris_hard.csv", background(3000) + rows(260, 200, 600, atk, "10.0.0.5", 80, label="ddos_slowloris",
     dur=(60, 300), fwd=(5, 15), bwd=(1, 4), fbytes=(200, 900), bbytes=(100, 400), syn=1))

# 4. Multi-stage intrusion for forecasting: recon -> lateral movement -> exfiltration
bad = "10.0.1.37"
recon = rows(900, 150, 220, bad, "10.0.0.20", lambda: int(rng.integers(1, 10000)), label="recon_portscan",
             dur=(0.0, 0.05), fwd=(1, 3), bwd=(0, 2), fbytes=(40, 90), bbytes=(0, 60), syn=1, rst=1, ack=0)
lateral = rows(160, 300, 430, bad, [f"10.0.1.{i}" for i in range(10, 60)], [445, 3389, 22, 5985], label="lateral_movement",
               dur=(2, 40), fwd=(20, 200), bwd=(20, 200), fbytes=(3000, 60000), bbytes=(3000, 60000))
exfil = rows(35, 480, 600, "10.0.1.42", "203.0.113.77", 443, label="exfiltration",
             dur=(30, 120), fwd=(3000, 9000), bwd=(100, 400), fbytes=(4_000_000, 20_000_000), bbytes=(5000, 30000))
save("4_multistage_intrusion.csv", background(3500) + recon + lateral + exfil)

# 5. HARD benign: flash crowd. Traffic spikes 5x with real handshakes and normal responses
save("5_flash_crowd_benign_hard.csv", background(2500) +
     rows(9000, 300, 420, CLIENTS + [f"192.168.{rng.integers(0, 4)}.{rng.integers(2, 250)}" for _ in range(300)],
          "10.0.0.5", 443, label="benign"))

# Ground truth for scoring
pd.DataFrame([
    ["1_normal_baseline.csv", "none", "", "LOW", "must not alert"],
    ["2_syn_flood_easy.csv", "ddos_syn", 300, "CRITICAL", "alert soon after t=300"],
    ["3_slowloris_hard.csv", "ddos_slowloris", 200, "HIGH", "looks quiet by volume; needs duration and byte features"],
    ["4_multistage_intrusion.csv", "recon>lateral>exfil", 150, "HIGH", "should forecast lateral movement from recon, exfil from lateral"],
    ["5_flash_crowd_benign_hard.csv", "none", "", "LOW", "big spike but legitimate: false-positive test"],
], columns=["file", "attack", "attack_start_s", "expected_risk", "what_it_tests"]).to_csv("ground_truth.csv", index=False)
