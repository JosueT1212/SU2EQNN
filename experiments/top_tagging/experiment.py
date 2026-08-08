"""Top quark jet tagging experiment using GATr on Cl(3,0).

Entry point:
    python -m experiments.top_tagging.experiment

Hydra config: config/top_tagging.yaml
"""

import os
import time
import logging

import torch
import hydra
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from su2eqnn.nets.gatr import GATr
from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.layers.mlp.config import MLPConfig
from experiments.top_tagging.dataset import TopTaggingDataset
from experiments.top_tagging.embedding import (
    embed_tagging_data,
    embed_tagging_data_r8,
    apply_axis_channel_dropout,
)
from experiments.top_tagging.wrapper import TopTaggingWrapper

log = logging.getLogger(__name__)

_r8_pseudoscalar_logged = False


def build_model(cfg) -> TopTaggingWrapper:
    """Instantiate GATr + TopTaggingWrapper from config.

    Parameters
    ----------
    cfg : DictConfig or object with .model attributes:
        hidden_mv_channels : int
        num_blocks         : int
        num_heads          : int
        in_mv_channels     : int (optional, default 1)
    """
    net = GATr(
        in_mv_channels=getattr(cfg.model, "in_mv_channels", 1),
        out_mv_channels=1,
        hidden_mv_channels=cfg.model.hidden_mv_channels,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(num_heads=cfg.model.num_heads),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=cfg.model.num_blocks,
    )
    return TopTaggingWrapper(net)


def build_embedding(batch, cfg) -> dict:
    """Dispatch to the correct embedding function based on cfg.data.embedding.

    "r8" -> single-channel embed_tagging_data_r8 (jet-axis bivector,
    optional p_lead pseudoscalar). Anything else (default/unset) ->
    existing multi-channel embed_tagging_data (R1/R4/R5 path, unchanged).
    """
    global _r8_pseudoscalar_logged
    variant = getattr(cfg.data, "embedding", "default")
    if variant == "r8":
        jet_axis = batch.axes[:, 0, :]
        p_lead = getattr(batch, "p_lead", None)
        if not _r8_pseudoscalar_logged:
            log.info(f"R8 embedding: pseudoscalar={'on' if p_lead is not None else 'off'}")
            _r8_pseudoscalar_logged = True
        return embed_tagging_data_r8(
            batch.x, batch.ptr, jet_axis,
            two_beams=cfg.data.two_beams,
            rescale=cfg.data.rescale_data,
            p_lead=p_lead,
        )
    return embed_tagging_data(
        batch.x, batch.ptr,
        two_beams=cfg.data.two_beams,
        rescale=cfg.data.rescale_data,
        axes=getattr(batch, "axes", None),
    )


def evaluate(model, loader, cfg, device):
    """Run evaluation on a DataLoader, return AUC and accuracy."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embedding = build_embedding(batch, cfg)
            logits = model(embedding)  # (B, 1)
            all_logits.append(logits.cpu())
            all_labels.append(batch.label.cpu())

    logits = torch.cat(all_logits, dim=0).squeeze(1).numpy()
    labels = torch.cat(all_labels, dim=0).numpy().astype(int)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    auc = roc_auc_score(labels, probs)
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    return {"auc": auc, "accuracy": acc}


def run_experiment(cfg: DictConfig) -> None:
    """Training + evaluation loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Data
    log.info("Loading datasets...")
    ds_train = TopTaggingDataset(rescale_data=cfg.data.rescale_data)
    ds_val = TopTaggingDataset(rescale_data=cfg.data.rescale_data)
    ds_test = TopTaggingDataset(rescale_data=cfg.data.rescale_data)
    ds_train.load_data(cfg.data.path, "train")
    ds_val.load_data(cfg.data.path, "val")
    ds_test.load_data(cfg.data.path, "test")

    loader_train = GeometricDataLoader(ds_train, batch_size=cfg.training.batch_size, shuffle=True)
    loader_val = GeometricDataLoader(ds_val, batch_size=cfg.evaluation.batch_size, shuffle=False)
    loader_test = GeometricDataLoader(ds_test, batch_size=cfg.evaluation.batch_size, shuffle=False)

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    optimizer_name = str(getattr(cfg.training, "optimizer", "adam")).lower()
    if optimizer_name == "lion":
        from lion_pytorch import Lion

        optimizer = Lion(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=float(getattr(cfg.training, "weight_decay", 0.0)),
        )
        log.info(
            f"Using Lion optimizer: lr={cfg.training.lr}, "
            f"weight_decay={getattr(cfg.training, 'weight_decay', 0.0)}"
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = None
    if getattr(cfg.training, "scheduler", None) == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.training.scheduler_t0_steps,
            T_mult=1,
            eta_min=cfg.training.eta_min,
        )
        log.info(
            f"Using CosineAnnealingWarmRestarts: T_0={cfg.training.scheduler_t0_steps} "
            f"steps, eta_min={cfg.training.eta_min}"
        )
    axis_dropout_p = float(getattr(cfg.training, "axis_dropout_p", 0.0))
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = 0.0
    ckpt_path = cfg.training.checkpoint_path
    dirname = os.path.dirname(ckpt_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader_train:
            batch = batch.to(device)
            embedding = build_embedding(batch, cfg)
            if axis_dropout_p > 0.0:
                embedding = apply_axis_channel_dropout(embedding, axis_dropout_p)
            logits = model(embedding)              # (B, 1)
            targets = batch.label.float().unsqueeze(1)  # (B, 1)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % cfg.training.validate_every_n_steps == 0:
                metrics = evaluate(model, loader_val, cfg, device)
                log.info(
                    f"Step {global_step} | val AUC={metrics['auc']:.4f} "
                    f"acc={metrics['accuracy']:.4f}"
                )
                if metrics["auc"] > best_auc:
                    best_auc = metrics["auc"]
                    torch.save(model.state_dict(), ckpt_path)
                    log.info(f"Saved best model (AUC={best_auc:.4f}) to {ckpt_path}")
                model.train()

        dt = time.time() - t0
        avg_loss = epoch_loss / len(loader_train)
        log.info(f"Epoch {epoch}/{cfg.training.epochs} | loss={avg_loss:.4f} | {dt:.1f}s")

    # Final test evaluation
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, loader_test, cfg, device)
    log.info(f"Test AUC={test_metrics['auc']:.4f} | accuracy={test_metrics['accuracy']:.4f}")


@hydra.main(config_path="../../config", config_name="top_tagging", version_base=None)
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
