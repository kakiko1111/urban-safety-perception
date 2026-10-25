"""Label construction, image lookup, train/val split and the PyTorch dataset."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_labels(data_path: Path, min_votes: int, threshold_std: float) -> pd.DataFrame:
    """Turn TrueSkill 'safer' scores into binary labels per place.

    Keeps the highest-vote score for each place_id, then labels places below
    mean - k*std as unsafe (0) and above mean + k*std as safe (1). Places in
    between are dropped.
    """
    studies = pd.read_csv(data_path / "studies.tsv", sep="\t")
    qscores = pd.read_csv(data_path / "qscores.tsv", sep="\t")

    safer_study_id = studies.loc[studies["study_question"] == "safer", "_id"].values[0]
    scores = qscores[qscores["study_id"] == safer_study_id].copy()
    scores = scores[scores["num_votes"] > min_votes]
    scores = scores.sort_values("num_votes", ascending=False)
    scores = scores.drop_duplicates(subset="place_id", keep="first")
    print(f"Found {len(scores)} places with safety scores")

    values = scores["trueskill.score"].values
    mean, std = np.mean(values), np.std(values)
    lower, upper = mean - threshold_std * std, mean + threshold_std * std
    print(f"Thresholds: unsafe < {lower:.2f}, safe > {upper:.2f}")

    scores["label"] = scores["trueskill.score"].apply(
        lambda x: 0 if x < lower else (1 if x > upper else -1)
    )
    labeled = scores[scores["label"] != -1][["place_id", "trueskill.score", "label"]].copy()
    print(
        f"Labeled places: {len(labeled)} "
        f"(unsafe: {(labeled['label'] == 0).sum()}, safe: {(labeled['label'] == 1).sum()})"
    )
    return labeled


def build_filename_map(images_path: Path) -> dict:
    """Map location_id -> image filename (third underscore-separated field)."""
    files = [f for f in os.listdir(images_path) if f.endswith(".JPG")]
    print(f"Found {len(files)} image files")
    mapping = {}
    for filename in tqdm(files, desc="Building filename mapping"):
        parts = filename.replace(".JPG", "").split("_")
        if len(parts) >= 3:
            mapping[parts[2]] = filename
    return mapping


def build_dataset(data_path: Path, id_to_filename: dict, min_votes: int, threshold_std: float) -> pd.DataFrame:
    """Expand labeled places to every photo at that place that exists on disk."""
    labeled = build_labels(data_path, min_votes, threshold_std)
    locations = pd.read_csv(data_path / "locations.tsv", sep="\t")

    expanded = labeled.merge(locations[["_id", "place_id"]], on="place_id", how="inner")
    expanded = expanded.rename(columns={"_id": "location_id"})
    print(f"Expanded to {len(expanded)} photos")

    expanded = expanded[expanded["location_id"].isin(id_to_filename.keys())].copy()
    df = expanded[["location_id", "place_id", "trueskill.score", "label"]].copy()
    df.columns = ["image_id", "place_id", "score", "label"]

    print(f"Dataset ready: {len(df)} images from {df['place_id'].nunique()} places")
    print(f"  Unsafe: {(df['label'] == 0).sum()}, Safe: {(df['label'] == 1).sum()}")
    return df


def place_split(df: pd.DataFrame, val_fraction: float, seed: int):
    """Split by place_id so no place appears in both train and validation."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df["place_id"]))
    train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    print(f"Train: {len(train_df)} samples from {train_df['place_id'].nunique()} places")
    print(f"Val:   {len(val_df)} samples from {val_df['place_id'].nunique()} places")
    return train_df, val_df


def get_transforms(color_mode: str, img_size: int, train: bool):
    """RGB keeps colour; grayscale is replicated to 3 channels for pretrained models."""
    steps = [transforms.Resize((img_size, img_size))]
    if color_mode == "gray":
        steps.append(transforms.Grayscale(num_output_channels=3))
    if train:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
        if color_mode == "rgb":
            steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class PlacePulseDataset(Dataset):
    """Loads Place Pulse images by image_id; missing or corrupt files become gray images."""

    def __init__(self, dataframe, images_path, id_to_filename, transform=None, return_id=False):
        self.data = dataframe.reset_index(drop=True)
        self.images_path = images_path
        self.id_to_filename = id_to_filename
        self.transform = transform
        self.return_id = return_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])

        filename = self.id_to_filename.get(image_id)
        img = None
        if filename:
            try:
                img = Image.open(os.path.join(self.images_path, filename)).convert("RGB")
            except OSError:
                img = None
        if img is None:
            img = Image.new("RGB", (224, 224), color="gray")

        if self.transform:
            img = self.transform(img)
        if self.return_id:
            return img, label, image_id
        return img, label


def make_loader(df, images_path, id_to_filename, color_mode, cfg, train, return_id=False):
    ds = PlacePulseDataset(
        df, images_path, id_to_filename,
        get_transforms(color_mode, cfg["img_size"], train=train),
        return_id=return_id,
    )
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=train, num_workers=cfg["num_workers"])
