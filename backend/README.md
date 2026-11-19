# EchoSafe backend and ML

## Run

Install dependencies and start the API from the project root:

```powershell
python -m pip install -r requirements.txt
python scripts/run_backend.py
```

The API is mounted under `/api`. If no valid checkpoint is configured or present,
all analyses use a **deterministic demo fallback**. The response fields `mode`,
`model_status`, and `/api/model-info` report this explicitly; the fallback is not
a trained model or a diagnostic.

Copy `.env.example` to `.env` where supported by your process manager. Set
`ECHOSAFE_MODEL_CHECKPOINT` to a `.pt` file. Checkpoints must contain
`model_state`, `emotion_labels`, and `model_config`; the matching `<checkpoint>.vocab.json`
file is required. If MongoDB configuration is absent or unreachable, history uses
`backend/data/history.json`.

## Training

The synthetic fixture requires no downloads:

```powershell
python scripts/train.py --fixture ml/fixtures/synthetic.jsonl --epochs 3
```

The training script writes a checkpoint and tokenizer vocabulary but intentionally
does not invent validation metrics. Supply real evaluated metrics to the checkpoint
only after producing them. Evaluation requires the metric dictionary to exist:

```powershell
python scripts/evaluate.py --checkpoint ml/checkpoints/echosafe.pt --fixture ml/fixtures/synthetic.jsonl
```
