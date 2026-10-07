"""Hold out whole readers for validation and testing, and apply support floors from training data only.

Book 1 calls these in order:

    assignment = assign_user_split(interactions["user_id"].unique())      # reader -> train / validation / test
    splits     = split_interactions(interactions, assignment)              # three DataFrames
    likes      = count_likes_per_work(splits["train"])                     # support counted on train only
    train      = apply_min_likes_floor(splits["train"], likes, floor=10)   # the floor, applied to any split
    summarise_split(splits)                                                # table for the notebook

Why whole readers: every model in this project counts which readers liked which
works. If a test reader's shelvings sat inside those counts, the model would be
graded on readers it had already seen. Holding out complete readers keeps the
test set unseen; at evaluation time a test reader's history is split into seeds
the model is shown and targets it must recover.

Why hash rather than shuffle: a reader's split comes from a hash of their id and
the seed, so it never changes when rows are reordered, filtered, or re-sampled.
scripts/benchmark.py assigns its dev/test seed works the same way.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from pathlib import Path

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SHARES = (0.8, 0.1, 0.1)
RANDOM_SEED = 593
LIKE_RATINGS = (4, 5)


def assign_user_split(user_ids, seed: int = RANDOM_SEED, shares=DEFAULT_SHARES) -> pd.Series:
    """Map each reader id to "train", "validation" or "test" by hashing the id with the seed.

    The first 8 hex digits of md5("<seed>:<user_id>") give a number in [0, 2^32);
    its position within that range decides the split, so shares of (0.8, 0.1, 0.1)
    give about 80/10/10 of readers. Returns a Series indexed by user id.
    """
    if abs(sum(shares) - 1) > 1e-9 or len(shares) != 3:
        raise ValueError("shares must be three numbers summing to 1")
    user_ids = np.asarray(pd.unique(np.asarray(user_ids)))
    # md5 is used as a cheap, stable pseudo-random function, not for security: the
    # same (seed, user_id) always lands on the same fraction on every machine.
    fractions = np.fromiter(
        (int(hashlib.md5(f"{seed}:{int(u)}".encode()).hexdigest()[:8], 16) / 2**32 for u in user_ids),
        dtype=np.float64, count=len(user_ids),
    )
    cut_points = np.cumsum(shares)[:-1]                           # (0.8, 0.9)
    # fraction < 0.8 -> train, 0.8-0.9 -> validation, >= 0.9 -> test
    labels = np.array(SPLIT_NAMES)[np.searchsorted(cut_points, fractions, side="right")]
    return pd.Series(labels, index=pd.Index(user_ids, name="user_id"), name="split")


def split_interactions(interactions: pd.DataFrame, assignment: pd.Series) -> dict[str, pd.DataFrame]:
    """Partition the interactions into the three splits using the reader assignment."""
    split_of_row = assignment.reindex(interactions["user_id"].to_numpy()).to_numpy()  # label per row
    if pd.isna(split_of_row).any():
        raise ValueError("every reader in the interactions must appear in the assignment")
    return {name: interactions[split_of_row == name].reset_index(drop=True) for name in SPLIT_NAMES}


def count_likes_per_work(train_interactions: pd.DataFrame, like_ratings=LIKE_RATINGS) -> pd.Series:
    """Number of training readers who liked each work (rating 4 or 5). Index: work_id."""
    liked = train_interactions[train_interactions["rating"].isin(like_ratings)]
    return liked.groupby("work_id").size().rename("likes")


def apply_min_likes_floor(interactions: pd.DataFrame, likes_per_work: pd.Series, floor: int) -> pd.DataFrame:
    """Keep only rows for works with at least `floor` likes in `likes_per_work`.

    Pass the counts from count_likes_per_work(train) so the floor is decided by
    training data alone, then apply it to train, validation or test alike. Works
    absent from the counts have zero likes and are removed.
    """
    # Counting support on the evaluation splits would leak which works the held-out
    # readers liked, so the caller passes training counts and this just applies them.
    eligible = likes_per_work.index[likes_per_work >= floor]
    return interactions[interactions["work_id"].isin(eligible)].reset_index(drop=True)


def summarise_split(splits: dict[str, pd.DataFrame], like_ratings=LIKE_RATINGS) -> pd.DataFrame:
    """One row per split: readers, reader-work rows, likes, and works with at least one like."""
    rows = []
    for name, frame in splits.items():
        liked = frame[frame["rating"].isin(like_ratings)]
        rows.append({
            "split": name,
            "readers": frame["user_id"].nunique(),
            "reader-work rows": len(frame),
            "likes (4-5 stars)": len(liked),
            "works with a like": liked["work_id"].nunique(),
        })
    table = pd.DataFrame(rows).set_index("split")
    table["share of readers"] = (table["readers"] / table["readers"].sum()).round(3)
    return table


def load_or_build_user_splits(user_ids, path, seed: int = RANDOM_SEED, rebuild: bool = False, log=print) -> pd.Series:
    """Load the saved split assignment, or compute it and save it as a pickle.

    The assignment is deterministic (a hash of seed and reader id), so loading and rebuilding
    give the same answer; saving it means Books 2 and 3 read one file instead of re-deriving
    the rule, and a reader's split is on record.
    """
    path = Path(path)
    if path.exists() and not rebuild:
        if log:
            log(f"loading {path}")
        return pd.read_pickle(path)
    assignment = assign_user_split(user_ids, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    assignment.rename("split").rename_axis("user_id").to_pickle(path)
    return assignment


def floor_report(splits: dict[str, pd.DataFrame], likes_per_work: pd.Series, floor: int,
                 like_ratings=LIKE_RATINGS) -> pd.DataFrame:
    """What a minimum-likes floor, counted on train, keeps in every split.

    One row per split: works and likes before the floor, works and likes after it, and the share
    of likes that survive. Validation and test are filtered with the TRAIN counts, which is the
    rule Book 2 inherits: a work is recommendable because training readers liked it enough, never
    because held-out readers did.
    """
    rows = []
    for name, frame in splits.items():
        liked = frame[frame["rating"].isin(like_ratings)]
        kept = apply_min_likes_floor(liked, likes_per_work, floor)
        rows.append({
            "split": name,
            "works with a like": liked["work_id"].nunique(),
            f"works at {floor}+ train likes": kept["work_id"].nunique(),
            "likes": len(liked),
            "likes kept": len(kept),
            "share of likes kept": round(len(kept) / max(len(liked), 1), 3),
        })
    return pd.DataFrame(rows).set_index("split")


if __name__ == "__main__":
    import sys
    import time

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "data" / "processed" / "interactions_sample_50k.pkl")
    started = time.time()
    data = pd.read_pickle(path)
    assignment = assign_user_split(data["user_id"].unique())
    parts = split_interactions(data, assignment)
    print(summarise_split(parts).to_string())
    likes = count_likes_per_work(parts["train"])
    for floor in (2, 5, 10, 20):
        kept = apply_min_likes_floor(parts["train"], likes, floor)
        print(f"floor {floor:>2}: train works kept {kept['work_id'].nunique():>7,}  rows kept {len(kept):>10,}")
    print(f"done in {time.time() - started:.0f}s")
