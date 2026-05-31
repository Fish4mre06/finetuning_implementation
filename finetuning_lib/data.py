"""FinancialPhraseBank loading and splitting."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

FPB_ZIP_URL = (
    "https://huggingface.co/datasets/financial_phrasebank/"
    "resolve/main/data/FinancialPhraseBank-v1.0.zip"
)

SPLIT_FILES = {
    "all": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
    "75": "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
    "66": "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
    "50": "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
}

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
NUM_LABELS = 3


def load_financial_phrasebank(agreement: str = "all") -> Dataset:
    """Load FinancialPhraseBank from the official HF zip (no legacy dataset scripts)."""
    if agreement not in SPLIT_FILES:
        raise ValueError(f"agreement must be one of {list(SPLIT_FILES)}")

    response = requests.get(FPB_ZIP_URL, timeout=120)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    text = archive.read(SPLIT_FILES[agreement]).decode("iso-8859-1")

    rows = []
    for line in text.strip().splitlines():
        sentence, label = line.rsplit("@", 1)
        rows.append(
            {"sentence": sentence.strip(), "label": LABEL2ID[label.strip()]}
        )

    return Dataset.from_list(rows)


def load_sst2_sample(n: int = 500, seed: int = 42) -> Dataset:
    """Stratified sample of general-domain sentiment (SST-2) for a forgetting probe.

    SST-2 is binary (0=negative, 1=positive). This is out-of-domain relative to
    the financial sentiment task, so evaluating a finance-adapted model here
    probes catastrophic forgetting of general sentiment ability.

    Returns rows with keys ``sentence`` and ``label`` to match the FPB schema.
    Labels here use the SST-2 convention, NOT the FPB convention; the forgetting
    probe handles mapping (positive/negative only, no neutral).
    """
    ds = load_dataset("SetFit/sst2", split="validation")
    df = ds.to_pandas().rename(columns={"text": "sentence"})[["sentence", "label"]]

    per_class = max(1, n // 2)
    parts = []
    for label_value, group in df.groupby("label"):
        parts.append(group.sample(min(per_class, len(group)), random_state=seed))
    sampled = pd.concat(parts).reset_index(drop=True)
    return Dataset.from_pandas(sampled)


def remap_dataset_labels(dataset_dict: DatasetDict, target_label2id: dict[str, int]) -> DatasetDict:
    """Remap integer labels from FinancialPhraseBank convention to a backbone's label2id."""
    id_to_name = {v: k for k, v in LABEL2ID.items()}
    id_map = {fpb_id: target_label2id[id_to_name[fpb_id]] for fpb_id in LABEL2ID.values()}

    def _remap(batch):
        batch["label"] = [id_map[label] for label in batch["label"]]
        return batch

    return dataset_dict.map(_remap, batched=True)


def make_stratified_splits(
    df: pd.DataFrame,
    random_state: int = 42,
    train_ratio: float = 0.70,
    val_ratio_of_holdout: float = 0.50,
) -> DatasetDict:
    """70/15/15 train/val/test with stratification (matches main notebook)."""
    holdout = 1.0 - train_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=holdout,
        random_state=random_state,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_ratio_of_holdout,
        random_state=random_state,
        stratify=temp_df["label"],
    )
    return DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df[["sentence", "label"]].reset_index(drop=True)
            ),
            "validation": Dataset.from_pandas(
                val_df[["sentence", "label"]].reset_index(drop=True)
            ),
            "test": Dataset.from_pandas(
                test_df[["sentence", "label"]].reset_index(drop=True)
            ),
        }
    )
