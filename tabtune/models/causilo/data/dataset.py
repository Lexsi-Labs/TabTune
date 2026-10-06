"""Fitted table transforms and ensemble metadata shared by both prediction paths."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target

from .encoding import FeatureEncoder
from .ensemble import EnsembleMember, make_ensemble_members
from .normalization import Normalizer


@dataclass
class PreparedDataset:
    """Training context in encoded feature space, before member permutations.

    ``features`` keeps the encoded training table even when transformed tables
    are not retained. ``members`` is in construction order; execution and saved
    K/V caches use ``members_by_normalization()`` order instead. Field names are
    part of the fitted-state serialization format.
    """

    encoder: FeatureEncoder
    features: np.ndarray
    targets: np.ndarray
    target_encoder: LabelEncoder | StandardScaler
    members: tuple[EnsembleMember, ...]
    normalizers: dict[str, Normalizer]
    normalized_cache: dict[str, np.ndarray]

    @classmethod
    def prepare(
        cls,
        table,
        targets,
        *,
        task: str,
        n_estimators: int,
        retain_preprocessing: bool,
        random_state: int,
        class_permutation_size: int,
    ) -> "PreparedDataset":
        """Fit transforms and build members for the model's output space.

        ``class_permutation_size`` sizes class permutations.
        """
        labels = np.asarray(targets)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 1 or len(labels) != len(table):
            raise ValueError("Targets must contain one value per training row")
        if pd.isna(labels).any():
            raise ValueError("Targets cannot contain missing values")
        classes = 0
        if task == "classification":
            if type_of_target(labels) not in {"binary", "multiclass"}:
                raise ValueError("Classification targets must be discrete labels")
            encoder = LabelEncoder().fit(labels)
            classes = len(encoder.classes_)
            if classes < 1:
                raise ValueError("Classification requires at least one class")
            encoded_targets = encoder.transform(labels)
        else:
            # Center in float64 before reducing precision for model execution.
            labels = labels.astype(np.float64)
            if not np.isfinite(labels).all():
                raise ValueError("Regression targets must be finite")
            encoder = StandardScaler().fit(labels[:, None])
            encoded_targets = encoder.transform(labels[:, None])[:, 0].astype(np.float32)
        schema, training = FeatureEncoder.fit(table)
        permutation_classes = min(classes, class_permutation_size)
        members = make_ensemble_members(training.shape[1], permutation_classes, n_estimators, random_state)
        methods = dict.fromkeys(member.normalization for member in members)
        transforms = {method: Normalizer.fit(training, method) for method in methods}
        retained = (
            {method: transform.transform(training) for method, transform in transforms.items()}
            if retain_preprocessing
            else {}
        )
        return cls(
            encoder=schema,
            features=training,
            targets=encoded_targets,
            target_encoder=encoder,
            members=members,
            normalizers=transforms,
            normalized_cache=retained,
        )

    def query_tables(self, table) -> dict[str, np.ndarray]:
        """Transform queries once for all members and output-code rows."""
        numeric = self.encoder.transform(table)
        return {name: transform.transform(numeric) for name, transform in self.normalizers.items()}

    def training_table(self, method: str) -> np.ndarray:
        """Return normalized training features, recomputing only if not retained."""
        if method in self.normalized_cache:
            return self.normalized_cache[method]
        return self.normalizers[method].transform(self.features)

    def targets_for(self, member: EnsembleMember) -> np.ndarray:
        """Map original encoded class IDs to this member's model output IDs."""
        if member.class_order is None:
            return self.targets
        return np.asarray(member.class_order)[self.targets]

    def members_by_normalization(self) -> tuple[EnsembleMember, ...]:
        """Return the stable execution order, also used when storing K/V caches.

        Grouping lets members reuse a normalized training table. For eight
        members this returns original indices 0, 4, 1, 5, 2, 6, 3, 7.
        """
        return tuple(
            member for method in self.normalizers for member in self.members if member.normalization == method
        )
