import numpy as np
try:
    # Newer stable_worldmodel layout
    from stable_worldmodel.data.formats.video import VideoDataset
except ImportError:
    # Older layout
    from stable_worldmodel.data.dataset import VideoDataset
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from torch.utils.data import Dataset, DataLoader
from loguru import logger as logging
from PIL import Image
from torchvision import transforms as T
from pathlib import Path

import pickle as pkl

from src.custom_codes.misc import angle_difference, to_local_coords


class ClevrerVideoDataset(VideoDataset):
    """
    Custom VideoDataset for CLEVRER with episode index offset support.

    This class extends VideoDataset to add an idx_offset parameters  that shifts
    all episode indices by a constant value. 
    """

    def __init__(self, name, *args, idx_offset=0, **kwargs):
        # Call parent VideoDataset.__init__
        super().__init__(name, *args, **kwargs)

        # Store the offset
        self.idx_offset =idx_offset


    def __repr__(self):
        return (
            f"ClevrerVideoDataset(name='{self.dataset}', "
            f"num_episodes={len(self.episodes)}, "
            f"idx_offset={self.idx_offset}, "
            f"frameskip={self.frameskip}, "
            f"num_steps={self.num_steps})"
        )

    def __getitem__(self, index):
        episode = self.idx_to_episode[index]
        episode_indices = self.episode_indices[episode+self.idx_offset]
        offset = index - self.episode_starts[episode]

        # determine clip bounds
        start = offset if not self.complete_traj else 0
        stop = start + self.clip_len if not self.complete_traj else len(self.episode_indices[episode+self.idx_offset])
        step_slice = episode_indices[start:stop]
        steps = self.dataset[step_slice]

        for col, data in steps.items():
            if col == "action":
                continue

            data = data[:: self.frameskip]
            steps[col] = data

            if col in self.decode_columns:
                steps[col] = self.decode(steps["data_dir"], steps[col], start=start, end=stop)

        if self.transform:
            steps = self.transform(steps)

        # stack frames
        for col in self.decode_columns:
            if col not in steps:
                continue
            steps[col] = torch.stack(steps[col])

        # reshape action
        if "action" in steps:
            act_shape = self.num_steps if not self.complete_traj else len(self.episode_indices[episode+self.idx_offset])
            steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps
    


# ============================================================================
# Dataset for Pre-extracted Slot Representations
# ============================================================================
class PushTSlotDataset(Dataset):
    """
    Dataset for pre-extracted slot representations from PushT.
    
    This class mirrors the behavior of swm.data.VideoDataset to ensure
    identical data processing. Key behaviors:
    - Window stride of 1 (not frameskip) for sample indices
    - Action is reshaped to (T, action_dim * frameskip) by VideoDataset
    - Normalization uses mean/std without clamping (same as WrapTorchTransform)
    - nan_to_num is only applied in forward pass, not in dataset
    
    Each sample contains:
    - pixels_embed: Pre-extracted slot embeddings (T, num_slots, slot_dim)
    - action: Action sequence (T, action_dim * frameskip)
    - proprio: Proprioception sequence (T, proprio_dim)
    - state: State sequence (T, state_dim) [optional, for evaluation]
    
    Args:
        slot_data: Dict mapping video_id to slot embeddings
        split: 'train' or 'val'
        history_size: Number of history frames
        num_preds: Number of future frames to predict
        action_dir: Path to action pickle file
        proprio_dir: Path to proprioception pickle file
        state_dir: Path to state pickle file (optional)
        frameskip: Frame skip factor (affects action reshaping)
        seed: Random seed for sampling
    """
    
    def __init__(
        self,
        slot_data: dict,
        split: str,
        history_size: int,
        num_preds: int,
        action_dir: str,
        proprio_dir: str,
        state_dir: str = None,
        frameskip: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.slot_data = slot_data
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip
        self.n_steps = history_size + num_preds
        self.seed = seed
        
        # Load action and proprio data
        with open(action_dir, "rb") as f:
            action_data = pkl.load(f)
        self.action_data = action_data[split]

        # Proprio is optional (some domains like navigation have none)
        self.proprio_data = None
        if proprio_dir is not None:
            with open(proprio_dir, "rb") as f:
                proprio_data = pkl.load(f)
            self.proprio_data = proprio_data[split]
        
        # State is optional (used for evaluation)
        self.state_data = None
        if state_dir is not None:
            with open(state_dir, "rb") as f:
                state_data = pkl.load(f)
            self.state_data = state_data[split]
        
        # Build index: list of (video_id, start_frame) tuples
        self.samples = self._build_sample_index()
        
        # Compute normalization statistics (matching WrapTorchTransform behavior)
        self._compute_normalization_stats()
        
        logging.info(f"[{split}] Created dataset with {len(self.samples)} samples from {len(self.slot_data)} videos")
    
    def _build_sample_index(self):
        """
        Build list of valid (video_id, start_frame) samples.
        
        Matches VideoDataset behavior: stride of 1, not frameskip.
        VideoDataset uses: episode_max_end = max(0, len(ep) - clip_len + 1)
        and iterates over all start positions with stride 1.
        """
        samples = []
        clip_len = self.n_steps * self.frameskip
        
        for video_id, slots in self.slot_data.items():
            num_frames = slots.shape[0]
            # max_start is inclusive, so we can start at positions 0 to max_start
            max_start = num_frames - clip_len
            
            if max_start < 0:
                continue
            
            # Stride 1 matching VideoDataset behavior
            for start_idx in range(0, max_start + 1):
                samples.append((video_id, start_idx))
        
        return samples
    
    def _compute_normalization_stats(self):
        """
        Compute mean and std for action and proprio normalization.
        
        Matches WrapTorchTransform(norm_col_transform(dataset, col)) behavior:
        - Computes stats over the RESHAPED action column (T, action_dim * frameskip)
        - No clamping of std (WrapTorchTransform doesn't clamp)
        - Uses tensor mean/std with unsqueeze(0)
        
        Note: VideoDataset reshapes action to (T, -1) before transform is applied.
        """
        # Collect all actions and proprios in their RESHAPED form
        # This matches how VideoDataset provides data to the transform
        all_actions = []
        all_proprios = []
        
        for video_id in self.action_data.keys():
            action_raw = self.action_data[video_id]  # (num_frames, action_dim)
            # Reshape to match VideoDataset's reshape: (T, action_dim * frameskip)
            # VideoDataset does: steps["action"].reshape(act_shape, -1)
            # where act_shape = num_steps and the raw actions are clip_len = n_steps * frameskip
            # So each T gets frameskip consecutive actions flattened
            num_frames = action_raw.shape[0]
            clip_len = self.n_steps * self.frameskip
            
            # Iterate over all possible clips (stride 1, matching _build_sample_index)
            for start_idx in range(0, num_frames - clip_len + 1):
                # Get clip_len consecutive raw actions
                action_clip = action_raw[start_idx:start_idx + clip_len]  # (clip_len, action_dim)
                # Reshape to (n_steps, action_dim * frameskip) - matching VideoDataset
                action_reshaped = action_clip.reshape(self.n_steps, -1)
                all_actions.append(action_reshaped)
        
        if self.proprio_data is not None:
            for video_id in self.proprio_data.keys():
                proprio_raw = self.proprio_data[video_id]  # (num_frames, proprio_dim)
                num_frames = proprio_raw.shape[0]
                clip_len = self.n_steps * self.frameskip

                for start_idx in range(0, num_frames - clip_len + 1):
                    # Get frames with frameskip (matching VideoDataset: data[::frameskip])
                    frame_indices = [start_idx + i * self.frameskip for i in range(self.n_steps)]
                    if frame_indices[-1] < num_frames:
                        proprio_clip = proprio_raw[frame_indices]  # (n_steps, proprio_dim)
                        all_proprios.append(proprio_clip)

        # Stack and compute stats matching norm_col_transform:
        # data.mean(0).unsqueeze(0), data.std(0).unsqueeze(0)
        all_actions = torch.from_numpy(np.concatenate(all_actions, axis=0)).float()  # (N*T, action_dim*frameskip)

        # Match norm_col_transform: mean(0).unsqueeze(0), std(0).unsqueeze(0)
        self.action_mean = all_actions.mean(0).unsqueeze(0)  # (1, action_dim * frameskip)
        self.action_std = all_actions.std(0).unsqueeze(0)    # (1, action_dim * frameskip)

        if self.proprio_data is not None:
            all_proprios = torch.from_numpy(np.concatenate(all_proprios, axis=0)).float()
            self.proprio_mean = all_proprios.mean(0).unsqueeze(0)
            self.proprio_std = all_proprios.std(0).unsqueeze(0)
        else:
            self.proprio_mean = None
            self.proprio_std = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_id, start_idx = self.samples[idx]
        
        # clip_len = n_steps * frameskip raw frames
        clip_len = self.n_steps * self.frameskip
        
        # Get frame indices with frameskip for slots (matching VideoDataset: data[::frameskip])
        frame_indices = [start_idx + i * self.frameskip for i in range(self.n_steps)]
        
        # Extract slot embeddings: (n_steps, num_slots, slot_dim)
        slots = self.slot_data[video_id]
        pixels_embed = torch.from_numpy(slots[frame_indices]).float()
        
        # Extract and reshape actions (matching VideoDataset behavior)
        # VideoDataset gets clip_len consecutive raw actions, then reshapes to (n_steps, -1)
        action_raw = self.action_data[video_id]
        action_clip = action_raw[start_idx:start_idx + clip_len]  # (clip_len, action_dim)
        # Reshape to (n_steps, action_dim * frameskip) - matching VideoDataset's reshape
        action = torch.from_numpy(action_clip.reshape(self.n_steps, -1)).float()
        
        # Normalize action (matching WrapTorchTransform behavior)
        # Note: No nan_to_num here - that's done in forward pass like train_causalwm.py
        action = (action - self.action_mean) / self.action_std

        sample = {
            "pixels_embed": pixels_embed,  # (T, S, D)
            "action": action,              # (T, action_dim * frameskip)
        }

        # Extract proprio with frameskip (matching VideoDataset: data[::frameskip])
        if self.proprio_data is not None:
            proprio_raw = self.proprio_data[video_id]
            proprio = torch.from_numpy(proprio_raw[frame_indices]).float()
            proprio = (proprio - self.proprio_mean) / self.proprio_std
            sample["proprio"] = proprio
        
        # Optionally include state
        if self.state_data is not None:
            state_raw = self.state_data[video_id]
            state = torch.from_numpy(state_raw[frame_indices]).float()
            sample["state"] = state

        return sample


# ============================================================================
# Raw-frame Navigation Dataset (NoMaD/NWM layout)
# ============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Meters of physical motion per stored frame, per dataset (from NWM/NoMaD).
# Used to put SE(2) deltas on a comparable scale across heterogeneous datasets.
METRIC_WAYPOINT_SPACING = {
    "recon": 0.25,
    "scand": 0.38,
    "tartan_drive": 0.72,
    "sacson": 0.255,
    "huron": 0.255,
    "go_stanford": 0.12,
    "go_stanford2": 0.12,
}


class NavRawDataset(Dataset):
    """Raw-frame dataset for NoMaD-preprocessed navigation data (RECON, etc.).

    On-disk layout per trajectory:
        <data_root>/<traj>/0.jpg, 1.jpg, ..., T.jpg
        <data_root>/<traj>/traj_data.pkl   {"position": (T,2), "yaw": (T,)}

    Train/val splits come from NoMaD-style split files:
        <split_dir>/{train,test}/traj_names.txt

    Each sample yields:
        "pixels": (n_steps, 3, H, W)   ImageNet-normalized frames
        "action": (n_steps, 3)         SE(2) deltas (dx, dy, dyaw) in the local
                                       frame of each frame; final-step pad = 0.

    Frames are read on-the-fly so frozen VideoSAUR runs every batch (no offline
    extraction). Action stats (mean/std) are computed at __init__ to match the
    norm_col_transform convention used by the pusht/clevrer pipelines.
    """

    def __init__(
        self,
        data_root: str,
        split_dir: str,
        split: str,                # "train" or "val"
        history_size: int,
        num_preds: int,
        frameskip: int = 1,
        image_size: int = 224,
        seed: int = 42,
        action_mean: torch.Tensor = None,   # if provided, skip recomputation (used for val)
        action_std: torch.Tensor = None,
        metric_waypoint_spacing: float = 1.0,  # divide xy deltas by this (NWM convention)
    ):
        super().__init__()
        self.data_root = Path(data_root).expanduser()
        self.split_dir = Path(split_dir).expanduser()
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.n_steps = history_size + num_preds
        self.frameskip = frameskip
        self.image_size = image_size
        self.seed = seed
        self.metric_waypoint_spacing = float(metric_waypoint_spacing) or 1.0

        # NoMaD splits use train/test; we map "val" -> "test" on disk.
        split_on_disk = "train" if split == "train" else "test"
        names_file = self.split_dir / split_on_disk / "traj_names.txt"
        traj_names = [ln.strip() for ln in names_file.read_text().splitlines() if ln.strip()]

        # Load per-trajectory poses + compute SE(2) deltas eagerly (cheap, ~MB total).
        self.actions = {}        # traj_id -> (T, 3) float32
        self.num_frames = {}     # traj_id -> int
        self.traj_ids = []
        for traj in traj_names:
            tdir = self.data_root / traj
            pkl_path = tdir / "traj_data.pkl"
            if not pkl_path.is_file():
                logging.warning(f"missing {pkl_path}; skipping")
                continue
            with open(pkl_path, "rb") as f:
                td = pkl.load(f)
            positions = np.asarray(td["position"], dtype=np.float32)
            yaws = np.asarray(td["yaw"], dtype=np.float32)
            T_frames = positions.shape[0]
            acts = np.zeros((T_frames, 3), dtype=np.float32)
            inv_spacing = 1.0 / self.metric_waypoint_spacing
            for t in range(T_frames - 1):
                local_next = to_local_coords(positions[t + 1:t + 2], positions[t], float(yaws[t]))
                acts[t, 0] = local_next[0, 0] * inv_spacing
                acts[t, 1] = local_next[0, 1] * inv_spacing
                acts[t, 2] = angle_difference(yaws[t + 1], yaws[t])
            self.actions[traj] = acts
            self.num_frames[traj] = T_frames
            self.traj_ids.append(traj)

        # Build (traj, start_idx) sample index, stride=1 (matches PushTSlotDataset).
        clip_len = self.n_steps * self.frameskip
        self.samples = []
        for traj in self.traj_ids:
            max_start = self.num_frames[traj] - clip_len
            if max_start < 0:
                continue
            self.samples.extend((traj, s) for s in range(max_start + 1))

        # Action normalization (compute on train, reuse for val).
        if action_mean is None or action_std is None:
            self._compute_action_stats()
        else:
            self.action_mean = action_mean
            self.action_std = action_std

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        logging.info(
            f"[NavRawDataset/{split}] {len(self.samples)} samples from {len(self.traj_ids)} trajectories "
            f"(clip_len={clip_len}, image_size={image_size})"
        )

    def _compute_action_stats(self):
        clip_len = self.n_steps * self.frameskip
        all_actions = []
        for traj in self.traj_ids:
            acts = self.actions[traj]
            T_frames = acts.shape[0]
            for start in range(T_frames - clip_len + 1):
                frame_indices = [start + i * self.frameskip for i in range(self.n_steps)]
                all_actions.append(acts[frame_indices])
        all_actions = torch.from_numpy(np.concatenate(all_actions, axis=0)).float()  # (N*T, 3)
        self.action_mean = all_actions.mean(0).unsqueeze(0)
        self.action_std = all_actions.std(0).unsqueeze(0).clamp(min=1e-6)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj, start = self.samples[idx]
        frame_indices = [start + i * self.frameskip for i in range(self.n_steps)]

        frames = []
        for fi in frame_indices:
            img_path = self.data_root / traj / f"{fi}.jpg"
            frames.append(self.transform(Image.open(img_path).convert("RGB")))
        pixels = torch.stack(frames, dim=0)  # (n_steps, 3, H, W)

        action = torch.from_numpy(self.actions[traj][frame_indices]).float()  # (n_steps, 3)
        action = (action - self.action_mean) / self.action_std

        return {
            "pixels": pixels,
            "action": action,
        }


def build_nav_datasets(
    data_root_base: str,
    split_dir_base: str,
    dataset_names,
    history_size: int,
    num_preds: int,
    frameskip: int = 1,
    image_size: int = 224,
    seed: int = 42,
):
    """Build train + val datasets concatenated across multiple NWM-style datasets.

    Each dataset's xy deltas are scaled by 1 / METRIC_WAYPOINT_SPACING[name] so a
    "step" represents a comparable physical distance across heterogeneous sources.
    Action mean/std is computed once over the concatenated training set and
    shared with the val set for consistent normalization.

    Returns: (train_concat, val_concat) -- both torch.utils.data.ConcatDataset.
    """
    from torch.utils.data import ConcatDataset

    data_root_base = Path(data_root_base).expanduser()
    split_dir_base = Path(split_dir_base).expanduser()

    dummy_mean = torch.zeros(1, 3)   # passed to skip per-dataset normalization
    dummy_std = torch.ones(1, 3)

    # First pass: build train sets without applying any normalization (we'll
    # overwrite mean/std with global stats after aggregation).
    train_sets = []
    for name in dataset_names:
        spacing = METRIC_WAYPOINT_SPACING.get(name, 1.0)
        ds = NavRawDataset(
            data_root=data_root_base / name,
            split_dir=split_dir_base / name,
            split="train",
            history_size=history_size,
            num_preds=num_preds,
            frameskip=frameskip,
            image_size=image_size,
            seed=seed,
            action_mean=dummy_mean,
            action_std=dummy_std,
            metric_waypoint_spacing=spacing,
        )
        train_sets.append(ds)

    # Aggregate all clip-level action samples across train datasets and compute
    # joint mean / std.
    n_steps = history_size + num_preds
    clip_len = n_steps * frameskip
    all_actions = []
    for ds in train_sets:
        for traj in ds.traj_ids:
            acts = ds.actions[traj]
            T_frames = acts.shape[0]
            for start in range(T_frames - clip_len + 1):
                fi = [start + i * frameskip for i in range(n_steps)]
                all_actions.append(acts[fi])
    all_actions = torch.from_numpy(np.concatenate(all_actions, axis=0)).float()
    global_mean = all_actions.mean(0).unsqueeze(0)
    global_std = all_actions.std(0).unsqueeze(0).clamp(min=1e-6)
    logging.info(
        f"[build_nav_datasets] global action mean={global_mean.flatten().tolist()}, "
        f"std={global_std.flatten().tolist()} (over {all_actions.shape[0]} clip-steps)"
    )

    # Inject the global stats into each train set so __getitem__ normalizes correctly.
    for ds in train_sets:
        ds.action_mean = global_mean
        ds.action_std = global_std

    # Build val sets sharing the same stats.
    val_sets = []
    for name in dataset_names:
        spacing = METRIC_WAYPOINT_SPACING.get(name, 1.0)
        ds = NavRawDataset(
            data_root=data_root_base / name,
            split_dir=split_dir_base / name,
            split="val",
            history_size=history_size,
            num_preds=num_preds,
            frameskip=frameskip,
            image_size=image_size,
            seed=seed,
            action_mean=global_mean,
            action_std=global_std,
            metric_waypoint_spacing=spacing,
        )
        val_sets.append(ds)

    return ConcatDataset(train_sets), ConcatDataset(val_sets)

