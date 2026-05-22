"""
Train Causal World Model on navigation data (NoMaD/NWM layout) using a frozen,
pretrained VideoSAUR for slots on-the-fly — no offline slot extraction.

Mirrors train_causalwm.py (the on-the-fly variant) but:
  - reads raw frames + SE(2) actions via NavRawDataset
  - drops proprio (navigation actions are already in initial-frame local coords)
  - keeps everything else (predictor, masking, losses) identical
"""
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F

from src.cjepa_predictor import MaskedSlotPredictor
from src.world_models.dinowm_causal import CausalWM, Embedder
from src.third_party.videosaur.videosaur import models
from src.custom_codes.custom_dataset import build_nav_datasets


# ============================================================================
# Data Setup
# ============================================================================
def get_data(cfg):
    """Raw-frame DataLoaders for navigation; VideoSAUR encodes inside forward.

    Supports a list of NWM-style datasets via `cfg.datasets`. Each dataset is
    expected on disk at:
        <cfg.data_root_base>/<name>/<traj>/{i.jpg, traj_data.pkl}
        <cfg.split_dir_base>/<name>/{train,test}/traj_names.txt
    """
    train_set, val_set = build_nav_datasets(
        data_root_base=cfg.data_root_base,
        split_dir_base=cfg.split_dir_base,
        dataset_names=list(cfg.datasets),
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        frameskip=cfg.frameskip,
        image_size=cfg.image_size,
        seed=cfg.seed,
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    logging.info(f"Train: {len(train_set)}, Val: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return spt.data.DataModule(train=train_loader, val=val_loader)


# ============================================================================
# Model Architecture
# ============================================================================
def get_world_model(cfg):
    """Pretrained VideoSAUR (frozen) + trainable MaskedSlotPredictor + action encoder."""

    def forward(self, batch, stage):
        # NaN guard at sequence boundaries (matches train_causalwm.py).
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        # Encode (pixels -> slots) + tile action across slots, on-the-fly.
        batch = self.model.encode(
            batch,
            target="embed",
            pixels_key="pixels",
            proprio_key=None,
            action_key="action",
        )

        embedding = batch["embed"][:, : cfg.dinowm.history_size, :, :]
        pred_output = self.model.predict(embedding)
        pixels_dim = batch["pixels_embed"].shape[-1]

        target_embedding = batch["embed"][
            :, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :
        ]

        if len(pred_output[1]) > 0:  # masking active
            pred_embedding, mask_indices = pred_output
            pred_history = pred_embedding[:, : cfg.dinowm.history_size, :, :]
            pred_future = pred_embedding[
                :, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :
            ]
            loss_masked_history = F.mse_loss(
                pred_history[:, :, mask_indices, :pixels_dim],
                embedding[:, :, mask_indices, :pixels_dim].detach(),
            )
            loss_future = F.mse_loss(
                pred_future[..., :pixels_dim],
                target_embedding[..., :pixels_dim].detach(),
            )
            batch["loss_masked_history"] = loss_masked_history
            batch["loss_future"] = loss_future
            batch["loss"] = loss_masked_history + loss_future
        else:
            pred_embedding = pred_output[0]
            pred_future = pred_embedding[
                :, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :
            ]
            loss_future = F.mse_loss(
                pred_future[..., :pixels_dim],
                target_embedding[..., :pixels_dim].detach(),
            )
            batch["loss_future"] = loss_future
            batch["loss"] = loss_future

        # Flatten for RankMe monitoring.
        if isinstance(pred_output, tuple) and len(pred_output) > 0:
            B, T, S, D = pred_output[0].shape
            batch["predictor_embed"] = pred_output[0].reshape(B * T, S * D)
        else:
            B, n_pred, S, D = pred_embedding.shape
            batch["predictor_embed"] = pred_embedding.reshape(B * n_pred, S * D)

        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)
        return batch

    # Build VideoSAUR; oc_ckpt weights are loaded via `cfg.model.load_weights`
    # inside models.build, and encoder/processor/initializer are wrapped EvalOnly
    # so they stay frozen during predictor training.
    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder
    slot_attention = model.processor
    initializer = model.initializer

    num_slots = cfg.videosaur.NUM_SLOTS
    slot_dim = cfg.videosaur.SLOT_DIM
    embedding_dim = slot_dim + cfg.dinowm.action_embed_dim  # no proprio for nav
    logging.info(f"Num slots: {num_slots}, Slot dim: {slot_dim}, Total embedding dim: {embedding_dim}")

    predictor = MaskedSlotPredictor(
        num_slots=num_slots,
        slot_dim=embedding_dim,
        history_frames=cfg.dinowm.history_size,
        pred_frames=cfg.dinowm.num_preds,
        num_masked_slots=cfg.get("num_masked_slots", 2),
        seed=cfg.seed,
        depth=cfg.predictor.get("depth", 6),
        heads=cfg.predictor.get("heads", 16),
        dim_head=cfg.predictor.get("dim_head", 64),
        mlp_dim=cfg.predictor.get("mlp_dim", 2048),
        dropout=cfg.predictor.get("dropout", 0.1),
    )

    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
    action_encoder = Embedder(in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim)
    logging.info(f"Action dim: {effective_act_dim} (no proprio)")

    world_model = CausalWM(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer=spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=None,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )

    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    world_model = spt.Module(
        model=world_model,
        forward=forward,
        optim={
            "predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
            "action_opt": add_opt("model.action_encoder", cfg.action_encoder_lr),
        },
    )
    return world_model


# ============================================================================
# Training Setup
# ============================================================================
def setup_pl_logger(cfg):
    if not cfg.wandb.enable:
        return None
    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb_logger = WandbLogger(
        name="cjepa_nav",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume="allow" if wandb_run_id else None,
        id=wandb_run_id,
        log_model=False,
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


class ModelObjectCallBack(Callback):
    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                output_path = self.dirpath / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
                torch.save(pl_module.state_dict(), output_path)
                logging.info(f"Saved world model object to {output_path}")
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                final_path = self.dirpath / f"{self.filename}_object.ckpt"
                torch.save(pl_module.state_dict(), final_path)
                logging.info(f"Saved final world model object to {final_path}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config_train_causal_nav")
def run(cfg):
    cache_dir = Path(swm.data.utils.get_cache_dir() if cfg.cache_dir is None else cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)
    world_model = get_world_model(cfg)

    callbacks = [ModelObjectCallBack(dirpath=cache_dir, filename=cfg.output_model_name, epoch_interval=1)]

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=wandb_logger,
        enable_checkpointing=True,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data,
        ckpt_path=str(cache_dir / f"{cfg.output_model_name}_weights.ckpt"),
        seed=cfg.seed,
    )
    manager()


if __name__ == "__main__":
    run()
