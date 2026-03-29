import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = os.getenv('MODEL_DIR', './model_files')

SEV_MAP = {0:'minimal', 1:'mild', 2:'moderate', 3:'mod_severe', 4:'severe'}
DSM_COLS = [
    'c1_anhedonia', 'c2_depressed', 'c3_sleep', 'c4_fatigue',
    'c5_appetite', 'c6_worthlessness', 'c7_concentration',
    'c8_psychomotor', 'c9_ideation'
]

class RippleModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.severity_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(h, 256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 5)
        )
        self.phq_head = nn.Sequential(
            nn.Linear(h, 128), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid()
        )
        self.dsm_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(h, 256),
            nn.GELU(), nn.Linear(256, 9), nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return {
            'severity': self.severity_head(cls),
            'phq':      self.phq_head(cls).squeeze(-1),
            'dsm':      self.dsm_head(cls),
            'pooled':   cls,
        }

def ensure_model():
    if not os.path.exists(f'{MODEL_DIR}/best_model.pt'):
        print("Model not found locally, downloading from HuggingFace...")
        snapshot_download(
            repo_id=os.getenv('HF_REPO_ID', 'knguyencas/ripple-nlp-model'),
            local_dir=MODEL_DIR,
            token=os.getenv('HF_TOKEN')
        )
        print("Download complete.")

def load_model():
    ensure_model()
    tokenizer = AutoTokenizer.from_pretrained(f'{MODEL_DIR}/tokenizer')
    model = RippleModel()
    model.load_state_dict(torch.load(
        f'{MODEL_DIR}/best_model.pt',
        map_location=DEVICE
    ))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def predict(text: str, model, tokenizer, max_length=256):
    enc = tokenizer(
        text, truncation=True, max_length=max_length,
        padding='max_length', return_tensors='pt'
    )
    with torch.no_grad():
        out = model(
            enc['input_ids'].to(DEVICE),
            enc['attention_mask'].to(DEVICE)
        )
    probs      = torch.softmax(out['severity'], dim=-1)[0].cpu().tolist()
    sev_id     = int(torch.argmax(out['severity'], dim=-1).item())
    phq_score  = round(out['phq'][0].item() * 27, 1)
    dsm_scores = out['dsm'][0].cpu().tolist()
    c9         = dsm_scores[8]
    risk_flag  = c9 >= 0.7 or sev_id == 4

    return {
        'severity':    SEV_MAP[sev_id],
        'severity_id': sev_id,
        'confidence':  round(probs[sev_id], 3),
        'phq_score':   phq_score,
        'dsm':         {col: round(v, 3) for col, v in zip(DSM_COLS, dsm_scores)},
        'risk_flag':   risk_flag,
        'c9_ideation': round(c9, 3),
    }