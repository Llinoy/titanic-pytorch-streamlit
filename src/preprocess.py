import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


RARE_TITLES = [
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
    "Rev", "Sir", "Jonkheer", "Dona"
]


@dataclass
class PreprocessArtifacts:
    age_median: float
    embarked_mode: str
    embarked_categories: List[str]
    title_categories: List[str]
    deck_categories: List[str]
    feature_columns: List[str]


class TitanicPreprocessor:
    """
    Fit on TRAIN only, then transform train/val/test consistently.
    Creates features:
      - HasCabin, Deck
      - Title (from Name)
      - FamilySize
      - FareLog = log1p(Fare)
      - Sex -> 0/1
      - One-hot: Embarked, Title, Deck (NO dummy_na; we handle unknowns)
    Drops: Name, Cabin, Ticket, PassengerId, Fare
    """

    def __init__(self) -> None:
        self.artifacts: Optional[PreprocessArtifacts] = None

    @staticmethod
    def _extract_title(name: pd.Series) -> pd.Series:
        return name.astype(str).str.extract(r" ([A-Za-z]+)\.", expand=False)

    @staticmethod
    def _normalize_title(title: pd.Series) -> pd.Series:
        s = title.copy()
        s = s.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        s = s.replace(RARE_TITLES, "Rare")
        s = s.fillna("Unknown")
        return s

    @staticmethod
    def _make_deck(cabin: pd.Series) -> pd.Series:
        deck = cabin.astype(str).str[0]
        deck = deck.where(cabin.notna(), "Unknown")
        deck = deck.fillna("Unknown")
        return deck

    @staticmethod
    def _safe_mode(s: pd.Series, default: str = "S") -> str:
        s2 = s.dropna()
        if len(s2) == 0:
            return default
        return str(s2.mode().iloc[0])

    def _engineer_base(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Cabin -> HasCabin + Deck
        if "Cabin" in df.columns:
            df["HasCabin"] = df["Cabin"].notna().astype(int)
            df["Deck"] = self._make_deck(df["Cabin"])
        else:
            df["HasCabin"] = 0
            df["Deck"] = "Unknown"

        # Name -> Title
        if "Name" in df.columns:
            df["Title"] = self._extract_title(df["Name"])
            df["Title"] = self._normalize_title(df["Title"])
        else:
            df["Title"] = "Unknown"

        # Sex -> 0/1
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).fillna(0).astype(int)
        else:
            df["Sex"] = 0

        # Ensure needed cols exist
        if "Embarked" not in df.columns:
            df["Embarked"] = np.nan
        if "Age" not in df.columns:
            df["Age"] = np.nan
        if "Fare" not in df.columns:
            df["Fare"] = 0.0
        if "SibSp" not in df.columns:
            df["SibSp"] = 0
        if "Parch" not in df.columns:
            df["Parch"] = 0

        # FamilySize
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1

        # FareLog
        df["FareLog"] = np.log1p(df["Fare"].fillna(0.0).astype(float))

        # Drop unused raw columns
        drop_cols = [c for c in ["Name", "Cabin", "Ticket", "PassengerId", "Fare"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        return df

    def fit(self, df_train: pd.DataFrame) -> "TitanicPreprocessor":
        # compute imputations from TRAIN only
        age_median = float(df_train["Age"].median())
        embarked_mode = self._safe_mode(df_train["Embarked"], default="S")

        tmp = self._engineer_base(df_train)
        tmp["Age"] = tmp["Age"].fillna(age_median)
        tmp["Embarked"] = tmp["Embarked"].fillna(embarked_mode).astype(str)
        tmp["Title"] = tmp["Title"].fillna("Unknown").astype(str)
        tmp["Deck"] = tmp["Deck"].fillna("Unknown").astype(str)

        embarked_categories = sorted(tmp["Embarked"].unique().tolist())
        title_categories = sorted(tmp["Title"].unique().tolist())
        deck_categories = sorted(tmp["Deck"].unique().tolist())

        # enforce fixed categories so get_dummies always produces same columns
        tmp["Embarked"] = pd.Categorical(tmp["Embarked"], categories=embarked_categories)
        tmp["Title"] = pd.Categorical(tmp["Title"], categories=title_categories)
        tmp["Deck"] = pd.Categorical(tmp["Deck"], categories=deck_categories)

        tmp_dum = pd.get_dummies(
            tmp,
            columns=["Embarked", "Title", "Deck"],
            prefix=["Embarked", "Title", "Deck"],
            dummy_na=False,
            dtype=int
        )

        feature_columns = [c for c in tmp_dum.columns if c != "Survived"]

        self.artifacts = PreprocessArtifacts(
            age_median=age_median,
            embarked_mode=str(embarked_mode),
            embarked_categories=embarked_categories,
            title_categories=title_categories,
            deck_categories=deck_categories,
            feature_columns=feature_columns
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.artifacts is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        art = self.artifacts
        tmp = self._engineer_base(df)

        # apply train-derived imputations
        tmp["Age"] = tmp["Age"].fillna(art.age_median)
        tmp["Embarked"] = tmp["Embarked"].fillna(art.embarked_mode).astype(str)
        tmp["Title"] = tmp["Title"].fillna("Unknown").astype(str)
        tmp["Deck"] = tmp["Deck"].fillna("Unknown").astype(str)

        # handle unseen categories (map to Unknown if possible)
        def map_unseen(s: pd.Series, cats: List[str]) -> pd.Series:
            if "Unknown" in cats:
                return s.where(s.isin(cats), "Unknown")
            return s.where(s.isin(cats), cats[0])

        tmp["Embarked"] = map_unseen(tmp["Embarked"], art.embarked_categories)
        tmp["Title"] = map_unseen(tmp["Title"], art.title_categories)
        tmp["Deck"] = map_unseen(tmp["Deck"], art.deck_categories)

        tmp["Embarked"] = pd.Categorical(tmp["Embarked"], categories=art.embarked_categories)
        tmp["Title"] = pd.Categorical(tmp["Title"], categories=art.title_categories)
        tmp["Deck"] = pd.Categorical(tmp["Deck"], categories=art.deck_categories)

        tmp_dum = pd.get_dummies(
            tmp,
            columns=["Embarked", "Title", "Deck"],
            prefix=["Embarked", "Title", "Deck"],
            dummy_na=False,
            dtype=int
        )

        X = tmp_dum.reindex(columns=art.feature_columns, fill_value=0)
        return X.to_numpy(dtype=np.float32)

    def save(self, path: str) -> None:
        if self.artifacts is None:
            raise RuntimeError("Nothing to save. Fit first.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.artifacts), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "TitanicPreprocessor":
        with open(path, "r", encoding="utf-8") as f:
            d: Dict[str, Any] = json.load(f)
        obj = cls()
        obj.artifacts = PreprocessArtifacts(**d)
        return obj
