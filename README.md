# ARGUS: AI-Based Network Attack Forecasting

Smart India Hackathon 2026, Problem Statement SIH26153.

ARGUS reads network traffic data, groups it into time windows, and forecasts the attack probability and likely attack stage (MITRE ATT&CK) for the next 5 to 25 minutes. A web dashboard shows the peak risk, the forecast, the predicted progression, and the top contributing features.

## Project structure

```
argus-sih/
  app.py                  Flask server and /api/predict route
  model_train.py          model training script
  world_model.pt          trained model
  scaler.pkl              feature scaler
  requirements.txt        Python dependencies
  templates/index.html    dashboard
  test_data/              labeled test traffic (easy and hard scenarios)
  docs/                   presentation
```

## Run locally

```
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000.

## Test data

`test_data/make_test_data.py` generates five labeled CSV files with a fixed seed:

| File | Tests |
|---|---|
| 1_normal_baseline.csv | normal traffic, must stay low risk |
| 2_syn_flood_easy.csv | SYN flood DDoS |
| 3_slowloris_hard.csv | low-volume slow attack |
| 4_multistage_intrusion.csv | recon, then lateral movement, then exfiltration |
| 5_flash_crowd_benign_hard.csv | harmless traffic spike, false-positive test |

`ground_truth.csv` lists the expected result for each file.

## Status

- Working: dashboard with risk gauge, forecast chart, stage timeline, and feature view.
- In progress: running the model on uploaded CSV files. Uploads currently map to sample scenarios.
- To do: score the model on real datasets (CIC-IDS2017, UNSW-NB15) and report precision, recall, false-positive rate, and lead time.

## References

- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- MITRE ATT&CK: https://attack.mitre.org
