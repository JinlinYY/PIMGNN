import argparse
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ._bootstrap import add_src_to_path
except ImportError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from psmi_legacy_public.data import (
    FunctionalGroupCache,
    GraphCache,
    GraphLLEDataset,
    collate_graph_batch,
)
from psmi_legacy_public.model import LLEGraphNet
from psmi_legacy_public.utils import Scaler
from psmi_legacy_public.paths import (
    BASE_TERNARY_CHECKPOINT,
    BIGSOLVDB_EXPERIMENT_ROOT,
    BIGSOLVDB_FINETUNED_CHECKPOINT,
    BIGSOLVDB_TEST_DATA,
    BIGSOLVDB_TRAIN_DATA,
)


def build_finetune_model(pretrained_path=BASE_TERNARY_CHECKPOINT, hidden_dim=3330):
    model = LLEGraphNet(use_mix_graph=True, use_fg=True, fg_vocab_size=512)
    checkpoint = torch.load(
        pretrained_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        weights_only=True,
    )
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.binary_head = nn.Sequential(
        nn.Linear(hidden_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    ).to(next(model.parameters()).device)
    return model


def prepare_dataframe(df):
    df = df.copy()
    df.rename(
        columns={
            "SMILES_1": "smiles1",
            "SMILES_2": "smiles2",
            "SMILES_3": "smiles3",
        },
        inplace=True,
    )
    df["t"] = 0.5
    df["system_id"] = 0
    df["aug_swap23"] = 0
    df["Ex1"] = df["Target"]
    df["Ex2"] = 0.0
    df["Ex3"] = 0.0
    df["Rx1"] = 0.0
    df["Rx2"] = 0.0
    df["Rx3"] = 0.0
    return df


def train_and_evaluate(
    train_path=BIGSOLVDB_TRAIN_DATA,
    test_path=BIGSOLVDB_TEST_DATA,
    pretrained_path=BASE_TERNARY_CHECKPOINT,
    output_model=BIGSOLVDB_FINETUNED_CHECKPOINT,
    output_predictions=BIGSOLVDB_EXPERIMENT_ROOT / "frozen_head_predictions.csv",
    epochs=40,
    batch_size=128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_finetune_model(pretrained_path).to(device)

    train_df = prepare_dataframe(pd.read_csv(train_path))
    test_df = prepare_dataframe(pd.read_csv(test_path))

    mean_t = train_df["T"].mean()
    std_t = train_df["T"].std()
    t_scaler = Scaler(mean=mean_t, std=std_t if std_t != 0 else 1.0)
    graph_cache = GraphCache()
    fg_cache = FunctionalGroupCache(corpus=None)

    train_dataset = GraphLLEDataset(
        train_df,
        t_scaler,
        graph_cache,
        mix_cache=None,
        fg_cache=fg_cache,
        use_fg=True,
        precompute_scalars=True,
    )
    test_dataset = GraphLLEDataset(
        test_df,
        t_scaler,
        graph_cache,
        mix_cache=None,
        fg_cache=fg_cache,
        use_fg=True,
        precompute_scalars=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graph_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
    )

    captured_features = {}

    def capture_backbone_input(module, args):
        captured_features["h"] = args[0]

    model.backbone.register_forward_pre_hook(capture_backbone_input)
    optimizer = torch.optim.Adam(model.binary_head.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch, targets in tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}"):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            model(batch)
            features = captured_features["h"]
            if features.dim() == 3:
                features = features.mean(dim=1)
            predictions = model.binary_head(features).squeeze()
            target = targets[:, 0].to(device).float()
            loss = criterion(predictions, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1:02d}: loss={running_loss / len(train_loader):.6f}")

    Path(output_model).parent.mkdir(parents=True, exist_ok=True)
    Path(output_predictions).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model)

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch, targets in tqdm(test_loader, desc="Test"):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            model(batch)
            features = captured_features["h"]
            if features.dim() == 3:
                features = features.mean(dim=1)
            predictions = model.binary_head(features).squeeze()
            y_true.extend(targets[:, 0].cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    result = test_df.iloc[: len(y_pred)].copy()
    result["True_LogS"] = y_true
    result["Pred_LogS"] = y_pred
    result.to_csv(output_predictions, index=False, encoding="utf-8-sig")


def main() -> None:
    """Parse CLI options and run frozen-head fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune the archived BigSolDB binary head.")
    parser.add_argument("--train-data", type=Path, default=BIGSOLVDB_TRAIN_DATA)
    parser.add_argument("--test-data", type=Path, default=BIGSOLVDB_TEST_DATA)
    parser.add_argument("--pretrained", type=Path, default=BASE_TERNARY_CHECKPOINT)
    parser.add_argument("--output-model", type=Path, default=BIGSOLVDB_FINETUNED_CHECKPOINT)
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=BIGSOLVDB_EXPERIMENT_ROOT / "frozen_head_predictions.csv",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train_and_evaluate(
        train_path=args.train_data,
        test_path=args.test_data,
        pretrained_path=args.pretrained,
        output_model=args.output_model,
        output_predictions=args.output_predictions,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
