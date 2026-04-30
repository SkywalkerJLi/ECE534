"""
build_phase_dataset.py
──────────────────────
Build a phase-conditioned LeRobot v3 dataset from `so101_color_augmented`
WITHOUT re-encoding any video.

For every source episode we emit up to three sub-episodes ─ "approach",
"carry", "full" ─ each carrying its own task string so SmolVLA can
condition behaviour on the prompt.  The new sub-episodes reference the
original packed MP4s by frame range (the videos/ tree is symlinked).

    approach : frames 0 .. pickup_frame
    carry    : frames 0 .. drop_frame
    full     : frames 0 .. n_frames

Episodes without phase labels in episode_phases.csv get only "full".

Run:
    python build_phase_dataset.py
Then point lerobot training at:
    --dataset.repo_id=local/so101_phase_split
    --dataset.root=/scratch/gpfs/TSILVER/sl5183/ECE534/so101_phase_split
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/scratch/gpfs/TSILVER/sl5183/ECE534/so101_color_augmented")
DST = Path("/scratch/gpfs/TSILVER/sl5183/ECE534/so101_phase_split")
PHASES_CSV = Path("/home/sl5183/ECE534/episode_phases.csv")

VIDEO_KEY = "observation.images.front"
FPS = 30

# The dataset is np.tile(orig_80, N_COLOR_VARIANTS): episodes 0..79 are the
# originals, episodes 80..159 are color-aug copies, etc.  Phase labels in
# episode_phases.csv only exist for episodes 0..79; we propagate them to every
# copy via source_ep = aug_ep % N_ORIGINAL_EPISODES.
N_ORIGINAL_EPISODES = 80

# Edit these to taste — these are the prompts SmolVLA will see.
TASK_STRINGS = {
    "approach": "pick up the cube",
    "carry":    "pick up the cube and bring it over to the target region",
    "full":     "pick up the cube and drop it over the target region",
}


def main():
    DST.mkdir(parents=True, exist_ok=True)

    # 1. Symlink videos/ — no re-encode, sub-episodes reference the same MP4s.
    dst_videos = DST / "videos"
    if dst_videos.is_symlink() or dst_videos.exists():
        print(f"videos/ already exists at {dst_videos} — leaving as-is")
    else:
        dst_videos.symlink_to((SRC / "videos").resolve())
        print(f"symlinked {dst_videos} -> {SRC / 'videos'}")

    # 2. Load source meta + data + phase labels.
    print("loading source episodes meta ...")
    src_eps = pd.read_parquet(SRC / "meta/episodes/chunk-000/file-000.parquet")

    print("loading source data parquet ...")
    src_data = pd.read_parquet(SRC / "data/chunk-000/file-000.parquet")
    # group once for fast per-episode lookup
    data_by_ep = {ep: g.sort_values("frame_index").reset_index(drop=True)
                  for ep, g in src_data.groupby("episode_index", sort=False)}

    print("loading phase labels ...")
    phases = pd.read_csv(PHASES_CSV).set_index("episode_index")
    print(f"  {len(phases)} episodes have phase labels "
          f"(of {len(src_eps)} total)")

    task_to_idx = {s: i for i, s in enumerate(TASK_STRINGS.values())}

    new_ep_records: list[dict] = []
    new_data_chunks: list[pd.DataFrame] = []
    new_ep_idx = 0
    global_idx = 0
    seg_counts = {"approach": 0, "carry": 0, "full": 0}

    print("building sub-episodes ...")
    for _, src_ep in src_eps.iterrows():
        ep        = int(src_ep["episode_index"])
        n_frames  = int(src_ep["length"])
        ep_rows   = data_by_ep[ep]
        v_chunk   = int(src_ep[f"videos/{VIDEO_KEY}/chunk_index"])
        v_file    = int(src_ep[f"videos/{VIDEO_KEY}/file_index"])
        v_from    = float(src_ep[f"videos/{VIDEO_KEY}/from_timestamp"])

        # Phase boundaries (None if missing/unlabeled).  Color-aug copies share
        # kinematics with the original, so we look up by ep % N_ORIGINAL_EPISODES.
        src_ep_for_phase = ep % N_ORIGINAL_EPISODES
        if src_ep_for_phase in phases.index:
            row    = phases.loc[src_ep_for_phase]
            pickup = None if pd.isna(row["pickup_frame"]) else int(row["pickup_frame"])
            drop_  = None if pd.isna(row["drop_frame"])   else int(row["drop_frame"])
        else:
            pickup, drop_ = None, None

        segments = [("approach", pickup), ("carry", drop_), ("full", n_frames)]

        for seg_name, seg_end in segments:
            if seg_end is None or seg_end <= 0 or seg_end > n_frames:
                continue

            sub = ep_rows.iloc[:seg_end].copy()
            length = len(sub)

            sub["frame_index"]   = np.arange(length, dtype=np.int64)
            sub["timestamp"]     = (np.arange(length) / FPS).astype(np.float32)
            sub["episode_index"] = np.int64(new_ep_idx)
            sub["index"]         = np.arange(global_idx, global_idx + length, dtype=np.int64)
            sub["task_index"]    = np.int64(task_to_idx[TASK_STRINGS[seg_name]])
            new_data_chunks.append(sub)

            ep_record = {
                "episode_index":      np.int64(new_ep_idx),
                "tasks":              np.array([TASK_STRINGS[seg_name]], dtype=object),
                "length":             np.int64(length),
                "data/chunk_index":   np.int64(0),
                "data/file_index":    np.int64(0),
                "dataset_from_index": np.int64(global_idx),
                "dataset_to_index":   np.int64(global_idx + length),
                f"videos/{VIDEO_KEY}/chunk_index":     np.int64(v_chunk),
                f"videos/{VIDEO_KEY}/file_index":      np.int64(v_file),
                f"videos/{VIDEO_KEY}/from_timestamp":  float(v_from),
                f"videos/{VIDEO_KEY}/to_timestamp":    float(v_from + length / FPS),
                "meta/episodes/chunk_index": np.int64(0),
                "meta/episodes/file_index":  np.int64(0),
            }
            # Carry over per-feature stats from the parent episode.
            # These are approximations for sub-episodes; global normalisation
            # uses meta/stats.json which we copy verbatim below.
            for col in src_eps.columns:
                if col.startswith("stats/"):
                    ep_record[col] = src_ep[col]

            new_ep_records.append(ep_record)
            new_ep_idx += 1
            global_idx += length
            seg_counts[seg_name] += 1

    print(f"  produced {new_ep_idx} sub-episodes, {global_idx} total frames")
    print(f"  segments: {seg_counts}")

    # 3. Write data parquet.
    print("writing data/chunk-000/file-000.parquet ...")
    out_data_dir = DST / "data/chunk-000"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    new_data = pd.concat(new_data_chunks, ignore_index=True)
    new_data.to_parquet(out_data_dir / "file-000.parquet", index=False)

    # 4. Write tasks parquet (index='task', column='task_index').
    print("writing meta/tasks.parquet ...")
    (DST / "meta").mkdir(parents=True, exist_ok=True)
    tasks_df = pd.DataFrame(
        {"task_index": list(range(len(TASK_STRINGS)))},
        index=list(TASK_STRINGS.values()),
    )
    tasks_df.index.name = "task"
    tasks_df.to_parquet(DST / "meta/tasks.parquet")

    # 5. Write episodes parquet.
    print("writing meta/episodes/chunk-000/file-000.parquet ...")
    out_eps_dir = DST / "meta/episodes/chunk-000"
    out_eps_dir.mkdir(parents=True, exist_ok=True)
    new_eps_df = pd.DataFrame(new_ep_records)
    # preserve column order from source where possible
    src_cols = list(src_eps.columns)
    extra    = [c for c in new_eps_df.columns if c not in src_cols]
    new_eps_df = new_eps_df[[c for c in src_cols if c in new_eps_df.columns] + extra]
    new_eps_df.to_parquet(out_eps_dir / "file-000.parquet", index=False)

    # 6. Write info.json (updated totals).
    print("writing meta/info.json ...")
    with open(SRC / "meta/info.json") as f:
        info = json.load(f)
    info["total_episodes"] = int(new_ep_idx)
    info["total_frames"]   = int(global_idx)
    info["total_tasks"]    = len(TASK_STRINGS)
    info["splits"]         = {"train": f"0:{new_ep_idx}"}
    with open(DST / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    # 7. Copy global stats.json verbatim (frame distribution shifts only
    #    slightly under phase duplication; rerun stats if you need exact).
    print("copying meta/stats.json ...")
    shutil.copy(SRC / "meta/stats.json", DST / "meta/stats.json")

    # 8. Copy README if present.
    if (SRC / "README.md").exists():
        shutil.copy(SRC / "README.md", DST / "README.md")

    print("\ndone.")
    print(f"  output: {DST}")
    print(f"  point lerobot at --dataset.root={DST}")


def sanity_check():
    """Reload the new dataset via LeRobotDataset and verify it lines up with
    the source.  Since we do not re-encode video, decoded pixels for any
    given source frame must be byte-identical between the two datasets."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("\n── sanity check ─────────────────────────────────────────")

    new_ds = LeRobotDataset(repo_id="local/so101_phase_split",     root=DST)
    src_ds = LeRobotDataset(repo_id="local/so101_color_augmented", root=SRC)

    print(f"  new ds: {new_ds.num_episodes} episodes, {new_ds.num_frames} frames")
    print(f"  src ds: {src_ds.num_episodes} episodes, {src_ds.num_frames} frames")

    # Look up the new sub-episodes that came from source ep 0.
    new_eps = pd.read_parquet(DST / "meta/episodes/chunk-000/file-000.parquet")
    phases  = pd.read_csv(PHASES_CSV).set_index("episode_index")
    pickup  = int(phases["pickup_frame"].loc[0])
    drop_   = int(phases["drop_frame"].loc[0])
    n_full  = int(new_eps.iloc[2]["length"])

    expected_lengths = {0: pickup, 1: drop_, 2: n_full}
    print(f"  src ep 0: pickup={pickup}, drop={drop_}, n_frames={n_full}")
    for sub_ep, want_len in expected_lengths.items():
        got = int(new_eps.iloc[sub_ep]["length"])
        ok  = "OK" if got == want_len else "MISMATCH"
        seg = ["approach", "carry", "full"][sub_ep]
        print(f"  sub-ep {sub_ep} ({seg:8s}): length={got} (expected {want_len}) {ok}")

    # Pixel parity: new_ds[k] should equal src_ds[k] for k < pickup, since
    # both indices map to source ep 0 frame k via the same MP4.
    img_key = f"observation.{VIDEO_KEY.split('.', 1)[1]}" if VIDEO_KEY.startswith("observation.") else VIDEO_KEY
    img_key = VIDEO_KEY  # the feature name is exactly observation.images.front
    for k in (0, 100, pickup - 1):
        new_s = new_ds[k]
        src_s = src_ds[k]
        img_match = torch.equal(new_s[img_key], src_s[img_key])
        act_match = torch.equal(new_s["action"], src_s["action"])
        print(f"  frame {k:4d}: image match={img_match}  action match={act_match}")

    # The carry sub-episode (sub-ep 1) starts at frame 0 of source ep 0.
    carry_start_global = int(new_eps.iloc[1]["dataset_from_index"])
    carry_frame0 = new_ds[carry_start_global]
    src_frame0   = src_ds[0]
    img_match = torch.equal(carry_frame0[img_key], src_frame0[img_key])
    print(f"  carry[0] vs src ep0 frame0: image match={img_match}")

    # Check task strings flow through.
    print(f"  new_ds tasks: {list(new_ds.meta.tasks)}")

    print("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
    try:
        sanity_check()
    except Exception as e:
        print(f"\nsanity check failed: {e!r}")
        raise
