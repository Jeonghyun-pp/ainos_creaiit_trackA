"""
Model D: Mixture of Experts (전문가 혼합 모델)
- 도메인별 전문가 모델이 각각 확률 출력
- 게이팅 네트워크가 사람별 가중치 결정
- 가중 평균 → 최종 확률
- PyTorch 구현
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

class Expert(nn.Module):
    """도메인 전문가: input → hidden → 확률 (0~1)"""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    """게이팅 네트워크: 전체 feature → 전문가별 가중치 (합=1)"""
    def __init__(self, total_input_dim, n_experts, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class MixtureOfExperts(nn.Module):
    """MoE: 전문가들 + 게이팅 → 가중 평균"""
    def __init__(self, input_dims):
        super().__init__()
        self.domain_names = list(input_dims.keys())
        self.experts = nn.ModuleDict({
            name: Expert(dim) for name, dim in input_dims.items()
        })
        total_dim = sum(input_dims.values())
        self.gate = GatingNetwork(total_dim, len(input_dims))

    def forward(self, domain_inputs):
        # 전문가 예측
        expert_preds = []
        for name in self.domain_names:
            pred = self.experts[name](domain_inputs[name])
            expert_preds.append(pred)
        expert_preds = torch.cat(expert_preds, dim=1)  # (batch, n_experts)

        # 게이팅 가중치
        all_features = torch.cat([domain_inputs[n] for n in self.domain_names], dim=1)
        weights = self.gate(all_features)  # (batch, n_experts)

        # 가중 평균
        weighted = (expert_preds * weights).sum(dim=1)
        return weighted, weights, expert_preds


# ─── 학습 유틸리티 ───────────────────────────────────────────────────

def make_dataloader(features_dict, y, batch_size=2048, shuffle=True):
    tensors = []
    for domain in DOMAINS:
        df = features_dict[domain]
        fcols = get_feature_columns(df)
        arr = df[fcols].values.astype(np.float32)
        tensors.append(torch.tensor(arr))
    tensors.append(torch.tensor(y.values.astype(np.float32)))
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def batch_to_dict(batch):
    domain_inputs = {DOMAINS[i]: batch[i].to(DEVICE) for i in range(len(DOMAINS))}
    y = batch[-1].to(DEVICE)
    return domain_inputs, y


def train_moe(features_train, y_train, features_valid, y_valid,
              features_test, y_test, epochs=30, lr=1e-3, batch_size=2048):
    """MoE 학습 및 평가"""

    # 입력 차원
    input_dims = {}
    for domain in DOMAINS:
        fcols = get_feature_columns(features_train[domain])
        input_dims[domain] = len(fcols)
    print(f"\n[MoE] 입력 차원: {input_dims}")

    # DataLoader
    train_loader = make_dataloader(features_train, y_train, batch_size, shuffle=True)
    valid_loader = make_dataloader(features_valid, y_valid, batch_size, shuffle=False)
    test_loader = make_dataloader(features_test, y_test, batch_size, shuffle=False)

    # 모델
    model = MixtureOfExperts(input_dims).to(DEVICE)
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(DEVICE)
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            domain_inputs, y = batch_to_dict(batch)
            pred, weights, expert_preds = model(domain_inputs)

            # class weight 적용
            w = torch.where(y == 1, pos_weight, torch.ones_like(y))
            loss = (criterion(pred, y) * w).mean()

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
                pred, _, _ = model(domain_inputs)
                val_preds.append(pred.cpu().numpy())
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

    # 평가
    def evaluate(loader):
        preds, labels, all_weights = [], [], []
        with torch.no_grad():
            for batch in loader:
                domain_inputs, y = batch_to_dict(batch)
                pred, weights, _ = model(domain_inputs)
                preds.append(pred.cpu().numpy())
                labels.append(y.cpu().numpy())
                all_weights.append(weights.cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        all_weights = np.concatenate(all_weights)
        return {
            "AUC-ROC": roc_auc_score(labels, preds),
            "PR-AUC": average_precision_score(labels, preds),
            "Brier": brier_score_loss(labels, preds),
        }, all_weights

    valid_metrics, valid_weights = evaluate(valid_loader)
    test_metrics, test_weights = evaluate(test_loader)

    # 게이팅 가중치 분석
    print("\n[MoE] 평균 게이팅 가중치 (test set):")
    for i, domain in enumerate(DOMAINS):
        mean_w = test_weights[:, i].mean()
        std_w = test_weights[:, i].std()
        print(f"  {domain:20s}: mean={mean_w:.3f}  std={std_w:.3f}")

    metrics = {"valid": valid_metrics, "test": test_metrics}
    return metrics, model


# ─── 메인 실행 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Model D: Mixture of Experts")
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

    metrics, model = train_moe(
        feat_train, y_train, feat_valid, y_valid, feat_test, y_test
    )

    print("\n" + "=" * 60)
    print("결과:")
    for split, m in metrics.items():
        print(f"\n  [{split}]")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")
    print("=" * 60)
