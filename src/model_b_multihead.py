"""
Model B: Multi-Head Neural Network (다중 헤드 신경망)
- 도메인별 헤드(소형 NN)가 임베딩 벡터 생성
- Concat → 공유 레이어 → 최종 확률
- PyTorch 구현, end-to-end 학습
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from feature_pipeline import (
    build_all_features, temporal_split, get_feature_columns,
    get_domain_arrays, DATA_DIR
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAINS = ["demographics_uw", "medical", "claim", "policy", "behavior"]


# ─── 모델 정의 ───────────────────────────────────────────────────────

class DomainHead(nn.Module):
    """도메인별 헤드: input_dim → hidden → embed_dim"""
    def __init__(self, input_dim, hidden_dim=64, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadRiskModel(nn.Module):
    """다중 헤드 → 공유 레이어 → 최종 확률"""
    def __init__(self, input_dims, hidden_dim=64, embed_dim=32, shared_dim=128):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: DomainHead(dim, hidden_dim, embed_dim)
            for name, dim in input_dims.items()
        })
        total_embed = embed_dim * len(input_dims)
        self.shared = nn.Sequential(
            nn.Linear(total_embed, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, domain_inputs):
        embeddings = [self.heads[name](x) for name, x in domain_inputs.items()]
        combined = torch.cat(embeddings, dim=1)
        return self.shared(combined).squeeze(1)


# ─── 학습 유틸리티 ───────────────────────────────────────────────────

def make_dataloader(features_dict, y, batch_size=2048, shuffle=True):
    """도메인별 feature + target → DataLoader"""
    tensors = {}
    for domain in DOMAINS:
        df = features_dict[domain]
        fcols = get_feature_columns(df)
        arr = df[fcols].values.astype(np.float32)
        tensors[domain] = torch.tensor(arr)

    y_tensor = torch.tensor(y.values.astype(np.float32))

    # TensorDataset은 단일 튜플만 지원하므로 도메인별로 묶어서 전달
    all_tensors = [tensors[d] for d in DOMAINS] + [y_tensor]
    dataset = TensorDataset(*all_tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def batch_to_dict(batch):
    """DataLoader batch → {domain: tensor}, y"""
    domain_inputs = {DOMAINS[i]: batch[i].to(DEVICE) for i in range(len(DOMAINS))}
    y = batch[-1].to(DEVICE)
    return domain_inputs, y


def train_multihead(features_train, y_train, features_valid, y_valid,
                    features_test, y_test, epochs=30, lr=1e-3, batch_size=2048):
    """Multi-Head NN 학습 및 평가"""

    # 입력 차원 계산
    input_dims = {}
    for domain in DOMAINS:
        fcols = get_feature_columns(features_train[domain])
        input_dims[domain] = len(fcols)
    print(f"\n[Multi-Head NN] 입력 차원: {input_dims}")

    # DataLoader
    train_loader = make_dataloader(features_train, y_train, batch_size, shuffle=True)
    valid_loader = make_dataloader(features_valid, y_valid, batch_size, shuffle=False)
    test_loader = make_dataloader(features_test, y_test, batch_size, shuffle=False)

    # 모델
    model = MultiHeadRiskModel(input_dims).to(DEVICE)
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # 학습
    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            domain_inputs, y = batch_to_dict(batch)
            logits = model(domain_inputs)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 검증
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                domain_inputs, y = batch_to_dict(batch)
                logits = model(domain_inputs)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_labels.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(-val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1:2d} | loss={avg_loss:.4f} | val_AUC={val_auc:.4f}")

    # 최적 모델 복원
    model.load_state_dict(best_state)
    model.eval()

    # 평가 함수
    def evaluate(loader):
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                domain_inputs, y = batch_to_dict(batch)
                logits = model(domain_inputs)
                probs = torch.sigmoid(logits)
                preds.append(probs.cpu().numpy())
                labels.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        return {
            "AUC-ROC": roc_auc_score(labels, preds),
            "PR-AUC": average_precision_score(labels, preds),
            "Brier": brier_score_loss(labels, preds),
        }

    metrics = {
        "valid": evaluate(valid_loader),
        "test": evaluate(test_loader),
    }

    return metrics, model


# ─── 메인 실행 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Model B: Multi-Head Neural Network")
    print("=" * 60)

    outcome = pd.read_csv(DATA_DIR / "modeling_outcome_high_cost_event.csv")
    train_df, valid_df, test_df = temporal_split(outcome)

    print(f"\nTrain: {len(train_df):,}  Valid: {len(valid_df):,}  Test: {len(test_df):,}")

    print("\n--- Feature 추출 (Train) ---")
    feat_train, y_train = build_all_features(train_df)
    print("\n--- Feature 추출 (Valid) ---")
    feat_valid, y_valid = build_all_features(valid_df)
    print("\n--- Feature 추출 (Test) ---")
    feat_test, y_test = build_all_features(test_df)

    metrics, model = train_multihead(
        feat_train, y_train, feat_valid, y_valid, feat_test, y_test
    )

    print("\n" + "=" * 60)
    print("결과:")
    for split, m in metrics.items():
        print(f"\n  [{split}]")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")
    print("=" * 60)
