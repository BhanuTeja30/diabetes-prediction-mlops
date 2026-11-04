# AI-Powered Diabetes Risk Prediction System with MLOps

An end-to-end, educational machine-learning system that estimates diabetes risk from eight routine
health measurements. It includes a Streamlit web app, a FastAPI backend, a reproducible DVC pipeline,
MLflow experiment tracking and model registry, Evidently drift monitoring, Prometheus metrics, a
Grafana dashboard, controlled retraining/promotion, Docker Compose deployment and a Pytest suite.

> **Medical disclaimer.** This project is for education and research only. Its output is a
> *model-estimated risk*, **not a medical diagnosis**. The model can be wrong in both directions: a
> "risk not indicated" result does not mean you do not have diabetes. Always consult a qualified
> healthcare professional for testing (e.g. HbA1c, fasting plasma glucose, OGTT) and interpretation.

---

## Contents
1. [Features](#features) · 2. [Technology stack](#technology-stack) · 3. [Dataset](#dataset-source-and-licensing)
4. [Architecture](#system-architecture) · 5. [Prerequisites](#prerequisites) · 6. [Installation](#installation)
7. [Dataset setup](#dataset-setup) · 8. [Training & evaluation](#model-training-and-evaluation)
9. [Promotion](#controlled-promotion-and-rollback) · 10. [Run the API](#fastapi-backend) · 11. [Run the frontend](#streamlit-frontend)
12. [MLflow](#mlflow) · 13. [DVC](#dvc) · 14. [Monitoring](#monitoring-evidently-prometheus-grafana)
15. [Retraining](#controlled-retraining) · 16. [Docker Compose](#docker-compose) · 17. [Testing](#testing)
18. [Troubleshooting](#troubleshooting) · 19. [Limitations](#limitations)

---

## Features
- **Separate users and admins**: everyone signs in. Users register, sign in and get predictions; admins additionally see model information and monitoring, reload the model, run drift checks on uploaded batches and manage accounts. Roles are enforced by the API, not just hidden in the UI.
- **Automatic risk prediction**: enter 8 values → click **Predict** → label, probability, threshold and plain-language interpretation.
- **Shared validation**: input limits live in `configs/model_config.yaml` and are enforced by both the Streamlit form and the Pydantic API schema.
- **Leak-free preprocessing**: zero-coded missing values (glucose, blood pressure, skin thickness, insulin, BMI only – *not* pregnancies) are converted to NaN and imputed **inside** the scikit-learn pipeline, fitted on the training split only.
- **Screening-oriented model selection**: Logistic Regression, Random Forest and XGBoost are compared on the validation set. Each model's threshold is tuned to reach ≥ 80 % recall, then the model with the best specificity wins (Brier score tie-break). Accuracy is never the selection criterion.
- **Honest evaluation**: the test split is used only once, by `src/evaluate.py`; ROC, PR, calibration curves and confusion matrix are saved.
- **Single versioned artifact**: `models/production/model.joblib` (full pipeline) + `metadata.json` + `evaluation.json`.
- **No fake predictions**: if no model is present the API returns HTTP 503 with setup instructions.
- **MLOps**: MLflow tracking + registry (aliases `candidate` / `production`), DVC pipeline, Evidently drift reports, Prometheus metrics and alert rules, provisioned Grafana dashboard, gated promotion with human approval and rollback.
- **Privacy by default**: no request bodies in logs; metrics carry no health values; only coarse operational metadata is stored. Raw-feature logging for drift analysis is **off** unless explicitly approved.

## Technology stack
Python 3.12+ (verified on 3.14) · scikit-learn 1.9 · XGBoost 3.4 · pandas 2.3 · NumPy 2.5 · Matplotlib / Seaborn ·
Streamlit 1.63 · FastAPI 0.141 + Uvicorn 0.52 · Pydantic 2 · MLflow 3.15 · DVC 3.67 · Evidently 0.7 ·
prometheus_client · Prometheus 3.5 · Grafana 12.1 · Docker / Docker Compose · Joblib · Pytest · Ruff · YAML + env vars.
Exact pins: `requirements*.txt`.

## Dataset source and licensing
- **Pima Indians Diabetes Database** – 768 records of female patients of Pima Indian heritage aged ≥ 21, collected by the
  US National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). Original reference: Smith J.W. et al. (1988),
  *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*, Proc. Symposium on Computer Applications in Medical Care.
- Historically distributed by the UCI Machine Learning Repository; widely mirrored (e.g. Kaggle, where it is listed as CC0).
  This project downloads the mirror at `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
  and verifies its SHA-256 (`params.yaml → data.expected_sha256`).
- Check the licence terms of whichever copy you use before redistribution. The dataset is **not** committed to Git
  (`.gitignore`); it is versioned with DVC. Never commit private health datasets to a public repository.
- Known data issues handled by the pipeline: zeros used as missing-value codes (insulin 374 rows, skin thickness 227,
  blood pressure 35, BMI 11, glucose 5). See `reports/figures/` and `reports/data_summary.json`.

## System architecture

```
           ┌─────────────────────────┐   HTTP/JSON    ┌──────────────────────────────────┐
 user ───► │ Streamlit frontend :8501│ ─────────────► │ FastAPI backend :8000            │
           │ Home/Predict/Model/Mon. │ ◄───────────── │ /predict /health /ready /metrics │
           └─────────────────────────┘                │ loads models/production/*.joblib │
                                                      └──────┬──────────────┬────────────┘
                                         scrape /metrics     │              │ privacy-conscious logs
                                   ┌─────────────────┐       │              ▼
            Grafana :3000 ◄─────── │ Prometheus :9090│ ◄─────┘   data/monitoring/*.csv
            (dashboard)            │ + alert rules   │                      │
                                   └─────────────────┘                      ▼
                                                           src/drift_detection.py (Evidently)
                                                           → reports/drift/*.html + latest_status.json
 Offline lifecycle (DVC):                                                 (exported as data_drift_alert)
  data/raw ─► prepare ─► train (3 models, MLflow) ─► evaluate (test set) ─► models/candidate
                                                         │
          promote_model --approve --approved-by NAME ────┴─► models/production (+ archive for rollback)
          retrain --dataset-dir <approved data>  ──► new candidate (never auto-promoted)
```

### Project layout
```
diabetes-prediction-mlops/
├── app/            frontend.py · api.py · schemas.py · predictor.py · config.py · monitoring.py
├── src/            data_processing.py · features.py · train.py · evaluate.py · tracking.py
│                   drift_detection.py · retrain.py · promote_model.py · common.py
├── data/           raw/ (DVC-tracked) · processed/ (DVC outputs) · monitoring/ (runtime logs)
├── models/         candidate/ · production/ · archive/<version>/ · promotion_log.json
├── reports/        figures/ (EDA) · evaluation/ · drift/ · retraining/
├── tests/          test_preprocessing.py · test_model.py · test_api.py · test_monitoring.py · conftest.py
├── monitoring/     prometheus.yml · alert_rules.yml · grafana/{provisioning,dashboards}
├── configs/        model_config.yaml (input limits, acceptance criteria, drift thresholds)
├── Dockerfile (frontend) · Dockerfile.api · Dockerfile.mlflow · docker-compose.yml
├── params.yaml · dvc.yaml · dvc.lock · requirements*.txt · pyproject.toml · .env.example
```

## Prerequisites
- Python **3.12 or newer** (3.14 used for verification) and `pip`
- Git (DVC uses it); Docker Desktop / Docker Engine with Compose v2 for the container deployment
- ~2 GB disk space for dependencies

## Installation

**Windows PowerShell**
```powershell
cd diabetes-prediction-mlops
py -3.14 -m venv .venv            # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

**Linux / macOS**
```bash
cd diabetes-prediction-mlops
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```
All commands below are run from the project root with the virtual environment active. They are identical in
PowerShell and bash unless shown separately. (`dvc` is also available as `python -m dvc`.)

## Dataset setup
```bash
python -m src.data_processing --download    # downloads and verifies the SHA-256
```
Offline / alternative copy: save the CSV (with or without the Kaggle header row) as `data/raw/diabetes_raw.csv`.
If you use a different file, update `data.expected_sha256` in `params.yaml` (or remove the line).

## Model training and evaluation
Run the whole pipeline (recommended – only re-runs what changed):
```bash
dvc repro
```
Or step by step:
```bash
python -m src.data_processing   # profile, validate, clean, EDA figures, stratified 60/20/20 split
python -m src.train             # train + compare 3 pipelines, tune thresholds, select, log to MLflow
python -m src.evaluate          # held-out test metrics, plots, acceptance check
```
Outputs: `models/candidate/`, `reports/evaluation/test_metrics.json`, `reports/evaluation/validation_comparison.json`,
`reports/evaluation/*.png`, `reports/figures/*.png`. Use `python -m src.train --no-mlflow` to skip tracking.

## Controlled promotion and rollback
A candidate is promoted **only** if it passes the acceptance criteria (`configs/model_config.yaml`), does not regress
against the current production model on the same test set, **and** a named person approves it:
```bash
python -m src.promote_model --check                                   # dry run
python -m src.promote_model --approve --approved-by "Your Name"       # promote
python -m src.promote_model --rollback --approved-by "Your Name"      # restore the previous model
```
The previous model is archived in `models/archive/<version>/`; every action is logged to `models/promotion_log.json`
and the MLflow registry alias `production` is moved. Restart the API (or call `POST /admin/reload` with the
`X-Admin-Token` header when `ADMIN_TOKEN` is set) to serve the new model.

## FastAPI backend
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
- Interactive API docs: **http://localhost:8000/docs** (ReDoc: http://localhost:8000/redoc)

| Method | Path | Access | Purpose |
|---|---|---|---|
| GET | `/` | public | Service info and status |
| GET | `/health` | public | 200 when the model is loaded, **503** otherwise (with setup instructions) |
| GET | `/ready` | public | Readiness (503 until a model is loaded) |
| GET | `/metrics` | public | Prometheus metrics (aggregated, no health data or usernames) |
| POST | `/auth/register` | public | Create a **user** account (if `ALLOW_SELF_REGISTRATION=true`) |
| POST | `/auth/login` | public | Returns a bearer token, username and role |
| GET | `/auth/me` | user, admin | Current account |
| POST | `/predict` | user, admin | Prediction for the 8 inputs |
| GET | `/model/info` | admin | Model name, version, dataset, metrics, limitations |
| GET | `/monitoring/summary` | admin | Aggregated counters, latency, drift status |
| POST | `/admin/reload` | admin (or `X-Admin-Token`) | Reload the production artifact |
| POST | `/admin/drift-check` | admin | Upload an approved CSV batch, get an Evidently drift report |
| GET | `/admin/drift-reports/{name}` | admin | View a drift report (HTML) |
| GET/POST | `/admin/users` | admin | List / create accounts (role `user` or `admin`) |
| PATCH/DELETE | `/admin/users/{username}` | admin | Change role, enable/disable, reset password / delete |

**Request examples** (sign in first, then send the token)

PowerShell:
```powershell
$login = Invoke-RestMethod -Uri http://localhost:8000/auth/login -Method Post -ContentType "application/json" -Body (@{ username="alice"; password="your-Password1" } | ConvertTo-Json)
$headers = @{ Authorization = "Bearer $($login.access_token)" }
$body = @{ pregnancies=2; glucose=135; blood_pressure=80; skin_thickness=25; insulin=90; bmi=29.5; diabetes_pedigree_function=0.45; age=35 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Headers $headers -Body $body
```
bash:
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"your-Password1"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"pregnancies":2,"glucose":135,"blood_pressure":80,"skin_thickness":25,"insulin":90,"bmi":29.5,"diabetes_pedigree_function":0.45,"age":35}'
```
Response (values come from the loaded model; shown for format only):
```json
{"prediction": 1, "prediction_label": "Diabetes Risk Indicated", "probability": 0.56, "decision_threshold": 0.50,
 "model_version": "1.0.0", "model_name": "Random Forest", "interpretation": "...", "disclaimer": "This is a model estimate ..."}
```
Requests without a valid sign-in return **401**; users calling admin routes get **403**. Invalid input returns **422** with `[{"field": ..., "message": ...}]` (submitted values are not echoed);
bodies over `MAX_REQUEST_BYTES` (4 KB) return **413**. Logs are structured JSON with request id, path, status and
duration – never the request body. Enter **0** for skin thickness or insulin when not measured.

## Streamlit frontend
In a second terminal (API running):
```bash
streamlit run app/frontend.py
```
Open **http://localhost:8501** and sign in (or create a user account on the *Create account* tab).

| Page | User | Admin |
|---|---|---|
| Home – description, instructions, disclaimer, service status | ✓ | ✓ |
| Prediction – form → label, icon, probability bar, recommendation (never colour alone) | ✓ | ✓ |
| Model information – version, dataset, training date, validation/test metrics, comparison, limitations, plots | | ✓ |
| Monitoring – request counts, failures, latency, model version, drift status, report viewer, **Reload model** button | | ✓ |
| Drift check – upload an approved CSV batch and view the Evidently report | | ✓ |
| User management – create users/admins, change roles, disable/enable, reset passwords, delete | | ✓ |
Set `API_URL` if the API is not on `http://localhost:8000`.

## Users and admins
- **Accounts** live in `data/auth/users.db` (SQLite, git-ignored). Passwords are stored only as salted
  PBKDF2-SHA256 hashes. Accounts contain no health data, and **predictions are never linked to a username**.
- **First admin**: set `ADMIN_USERNAME` / `ADMIN_PASSWORD` in `.env`; the API creates that admin at start-up if
  no admin exists (the values are ignored afterwards). Or create one from the command line:
  ```bash
  python -m app.manage_users create admin --role admin      # prompts for the password
  python -m app.manage_users list
  python -m app.manage_users set-role alice admin
  python -m app.manage_users disable alice
  python -m app.manage_users reset-password alice
  ```
  In Docker: `docker compose exec api python -m app.manage_users list`.
- **Users** create their own accounts on the sign-in page (always role `user`; nobody can self-register as admin).
  Set `ALLOW_SELF_REGISTRATION=false` to make admins create every account.
- **Sessions**: signed bearer tokens (`AUTH_SECRET_KEY`, at least 32 characters) valid for `TOKEN_TTL_MINUTES` (60).
  Role and active status are re-checked on every request, so disabling or demoting an account takes effect immediately.
  Refreshing the browser tab signs you out.
- **Safeguards**: 5 failed sign-ins lock an account for 5 minutes; the same error is returned for unknown users and
  wrong passwords; the last active admin cannot be demoted, disabled or deleted; admins cannot delete themselves;
  admin actions are written to the API log as `audit` events.

## MLflow
Training logs to MLflow automatically (parent run + one nested run per model type: hyperparameters, dataset
hash/version, training duration, validation metrics; the evaluate stage adds test metrics and plots; the selected
pipeline is registered as `diabetes-risk-classifier` with alias `candidate`, and promotion sets alias `production`).

- Local (default `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`): start the UI with
  ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  ```
- With Docker Compose running, the server is at **http://localhost:5000**; point training at it:
  - PowerShell: `$env:MLFLOW_TRACKING_URI = "http://localhost:5000"; dvc repro`
  - bash: `MLFLOW_TRACKING_URI=http://localhost:5000 dvc repro`

## DVC
The repository is already initialised (`.dvc/`, `dvc.yaml`, `dvc.lock`, `data/raw/diabetes_raw.csv.dvc`). From scratch:
```bash
git init
dvc init
dvc add data/raw/diabetes_raw.csv                 # track the dataset (creates the .dvc pointer file)
git add data/raw/diabetes_raw.csv.dvc .gitignore dvc.yaml params.yaml
```
Storage:
```bash
dvc remote add -d localstore ../dvc-storage        # local folder / network share
# dvc remote add -d s3store s3://bucket/path       # cloud (requires dvc[s3])
dvc push                                           # upload data + model artifacts
dvc pull                                           # fetch them on another machine
```
Reproduce and compare:
```bash
dvc repro                       # run prepare → train → evaluate as needed
dvc dag                         # show the pipeline graph
dvc params diff                 # what changed in params.yaml
dvc metrics show                # current metrics
dvc metrics diff HEAD~1         # compare with a previous commit
git checkout <rev> data/raw/diabetes_raw.csv.dvc && dvc checkout   # switch dataset version
dvc diff HEAD~1                 # which data files changed between versions
```

## Monitoring (Evidently, Prometheus, Grafana)
**Prediction metadata** (`MONITORING_LOG_PREDICTIONS=true`, default): hour-bucketed time, model version, predicted
class, probability and latency → `data/monitoring/predictions.csv`. No identifiers, no inputs.

**Feature data for drift** is health data, so it is collected **only** after documented approval
(`MONITORING_LOG_FEATURES=true`) → `data/monitoring/inputs.csv` (date + 8 features + predicted class, unlinkable to a request).
Alternatively supply an approved batch file.

**Drift report** (run on demand or on a schedule, e.g. daily cron / Windows Task Scheduler):
```bash
python -m src.drift_detection                                   # uses data/monitoring/inputs.csv
python -m src.drift_detection --current path/to/approved_batch.csv
```
It compares current data with the training reference (`data/processed/train.csv`) and writes
`reports/drift/drift_report_<time>.html`, `drift_summary_<time>.json` and `latest_status.json`. It distinguishes:
- **feature drift** – per-feature statistical tests (Evidently `DataDriftPreset`), alert when the drifted share ≥ `drift_share_threshold` (0.3);
- **prediction drift** – predicted-positive rate vs. the rate on the validation split, alert when it moves > 0.15;
- **data-quality issues** – out-of-range / missing values (Evidently `DataSummaryPreset` + explicit checks);
- **model-performance degradation** – *cannot* be measured without verified outcome labels, which this system does not
  collect. Drift is a reason to investigate, **not** proof the model is inaccurate, and it **never** triggers retraining.

Alerts: a WARNING log line, `data_drift_alert=1` on `/metrics`, the Prometheus `DataDriftDetected` alert rule,
the Grafana "Data-drift alert" panel and the Streamlit Monitoring page.

**Prometheus** scrapes `api:8000/metrics` every 15 s (`monitoring/prometheus.yml`) with alert rules for API down,
model not loaded, failure rate, latency and drift (`monitoring/alert_rules.yml`). UI: **http://localhost:9090**.

**Grafana** (**http://localhost:3000**, user `admin`, password from `.env`) is provisioned automatically with the
Prometheus data source and the *Diabetes MLOps → Diabetes Prediction Service* dashboard: request rate, successful vs failed
requests, average/p95 latency, availability, model version, model-loading failures, predicted-class mix and drift status.

## Controlled retraining
Retraining is manual (or an approved scheduled job) and only uses an **approved** dataset folder:
```
data/approved/<release>/dataset.csv      # 8 features + verified outcome labels
data/approved/<release>/manifest.yaml    # provenance + approval (see src/retrain.py docstring)
```
```bash
python -m src.retrain --dataset-dir data/approved/2026-10-01
```
Workflow: verify manifest, approval and SHA-256 → validate quality → prepare/split → train (MLflow) → evaluate on the
held-out test set → compare with production → write `reports/retraining/retraining_<time>.json`. It **never** promotes;
a human runs `promote_model --approve`. Rejected: unverified labels (predictions are never labels), checksum mismatches,
user submissions without a documented consent reference. A failed candidate leaves production unchanged.

## Docker Compose
1. Train and promote a model first (see above) – the `models/` folder is mounted read-only into the API container.
2. `Copy-Item .env.example .env` (PowerShell) or `cp .env.example .env`, and set `GRAFANA_ADMIN_PASSWORD`.

```bash
docker compose up -d --build      # build and start all services
docker compose ps                 # status + health
docker compose logs -f api        # follow logs (also: frontend, mlflow, prometheus, grafana)
docker compose restart api        # pick up a newly promoted model
docker compose down               # stop (add -v to delete MLflow/Prometheus/Grafana volumes)
```
| Service | URL |
|---|---|
| Streamlit frontend | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 (or `MLFLOW_PORT`) |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Images run as a non-root user, pin dependency versions, have health checks, and contain no secrets or data.
On Linux, make `data/monitoring` writable by UID 1000 (`sudo chown -R 1000:1000 data/monitoring`).

### One-click start / stop and auto-start at login (Windows)
| What | How |
|---|---|
| Start the app and open it in the browser | double-click **`Start-App.cmd`** |
| Stop the app (data is kept) | double-click **`Stop-App.cmd`** |
| Start automatically at every Windows login | `powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1` (add `-OpenBrowser` to also open the app) |
| Remove the login trigger | `powershell -ExecutionPolicy Bypass -File scripts\uninstall-autostart.ps1` |

`scripts\start.ps1` starts Docker Desktop if it is not running, waits for the engine, runs `docker compose up -d`
and waits until the API is healthy. The login trigger is a shortcut named **Diabetes** in your Startup folder
(`shell:startup`); you can also switch it off in *Task Manager → Startup apps*. Progress and errors are written to
`logs\autostart.log`. If you move the project folder, run `install-autostart.ps1` again so the shortcut points to the
new location.

## Deploying to AWS (single EC2 server)
`deploy/aws_deploy.sh` (run from Git Bash on Windows, or Linux/macOS) uses the AWS CLI to create one Ubuntu server
and start the stack with `docker-compose.prod.yml`. Only Caddy is public: it serves the app over HTTPS on ports
80/443. The API stays on the internal Docker network; MLflow, Prometheus and Grafana listen on the server's
loopback and are reached through an SSH tunnel. SSH is limited to the IP address that ran `provision`.

```bash
aws login --profile deploy             # browser sign-in, used automatically by the script
bash deploy/aws_deploy.sh check        # creates nothing
bash deploy/aws_deploy.sh provision    # creates the server, billing starts
bash deploy/aws_deploy.sh push         # uploads and starts; repeat to deploy updates
bash deploy/aws_deploy.sh password     # the server's admin login
bash deploy/aws_deploy.sh tunnel       # private access to MLflow, Grafana, Prometheus, API docs
bash deploy/aws_deploy.sh stop         # stop paying for compute between demos (also: start)
bash deploy/aws_deploy.sh destroy      # delete everything
```

Defaults: region `ap-south-1` (`AWS_REGION`), `t3.small` with a 30 GB disk and 4 GB swap (`INSTANCE_TYPE`), and a free
`<ip>.sslip.io` address (`DOMAIN` for your own). The server generates its own secrets, and starts with its own user
accounts: local `.env`, `data/auth` and MLflow history are not uploaded. Set a billing alert in your AWS account,
and if your IP changes, update the SSH rule in the `diabetes-mlops-sg` security group.

## Testing
```bash
pytest                  # full suite (unit + integration with a really trained model)
pytest -m "not integration"   # fast unit tests only
ruff check .            # lint
```
The integration tests run the real prepare → train → evaluate pipeline on the real dataset in a temporary folder
and exercise the API against that artifact. No external service is needed (MLflow is disabled in tests);
integration tests are skipped with a clear message if the dataset has not been downloaded.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `/health` returns 503 "No trained model found" | Run `dvc repro` and `python -m src.promote_model --approve --approved-by "Name"`, then restart the API. |
| Frontend says "Cannot reach the prediction service" | Start the API, or set `API_URL`. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (or use `.venv\Scripts\activate.bat` in cmd). |
| `dvc: command not found` | Use `python -m dvc ...` or reactivate the venv. |
| Dataset download fails (offline / proxy) | Download the CSV manually to `data/raw/diabetes_raw.csv`. |
| `Checksum mismatch` | The upstream file changed: inspect it, then update `data.expected_sha256` deliberately. |
| `InconsistentVersionWarning` / unpickling errors | Training and serving must use the same scikit-learn/XGBoost versions (pinned in requirements). Retrain. |
| MLflow warnings during training | Tracking is optional; training continues. Set `MLFLOW_ENABLED=false` to silence. |
| Docker: `GRAFANA_ADMIN_PASSWORD` / `AUTH_SECRET_KEY` error | Create `.env` from `.env.example` and fill in the secrets. |
| Docker: `Bind for 0.0.0.0:5000 failed: port is already allocated` | Another program uses port 5000: set `MLFLOW_PORT=5001` in `.env`. |
| "Incorrect username or password" for the admin | `ADMIN_PASSWORD` is only used when no admin exists. Reset it: `python -m app.manage_users reset-password admin`. |
| "Too many failed sign-in attempts" | Wait 5 minutes (or restart the API). |
| Docker: `failed to connect to the docker API` | Start Docker Desktop. |
| Drift check: "No current monitoring data" | Enable approved feature logging or pass `--current file.csv` (≥ 30 rows). |

## Limitations
- Small (768 rows), single-population dataset (Pima Indian women ≥ 21 years); results may not generalise to men,
  children, other ethnicities or today's clinical populations.
- Insulin and skin-thickness are missing for a large share of records and are imputed.
- The threshold favours sensitivity (≥ 80 % validation recall), so false positives are expected; ~1 in 4 diabetic cases
  in the test set are still missed.
- Probabilities are estimates and may be imperfectly calibrated (see `reports/evaluation/calibration_curve.png`).
- No clinical validation; not a medical device. Drift monitoring cannot detect accuracy loss without verified outcomes.

## Medical disclaimer
This software is provided for educational and research purposes only. It does not provide medical advice, diagnosis
or treatment, and must not be used to make clinical decisions. Predictions may be incorrect. Anyone concerned about
diabetes should consult a qualified healthcare professional for appropriate testing and interpretation.
