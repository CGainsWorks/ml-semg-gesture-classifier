"""Classical feature-plus-classifier baseline.

Linear discriminant analysis on Hudgins-style time-domain features is the
control scheme most commercial myoelectric prostheses actually ship. It is
not a strawman: it is cheap enough to run on an 8-bit microcontroller, it
trains in seconds, and on within-subject data it is genuinely competitive. If
a convolutional network cannot beat it by a margin that justifies the extra
milliseconds and kilobytes, the network does not belong on the device.

Three classifiers are provided so the baseline is not one lucky choice.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.config import FeatureConfig
from src.features.time_domain import channel_sigma, extract

CLASSIFIERS = ("lda", "svm", "random_forest")


def build_classifier(name: str, seed: int = 0) -> Any:
    if name == "lda":
        # Shrinkage plus the eigen solver: the feature matrix has 16 channels
        # times 17 features, and the within-class covariance is rank-deficient
        # in the smaller folds without it.
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    if name == "svm":
        return SVC(C=10.0, gamma="scale", kernel="rbf", probability=True, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    raise ValueError(f"unknown classifier {name!r}; expected one of {CLASSIFIERS}")


class ClassicalPipeline:
    """Feature extraction and a scikit-learn classifier, fit as one unit.

    The channel sigmas that set the zero-crossing, slope-sign-change and
    Willison thresholds are fit here, on the training windows only, and then
    reused verbatim at test time. Recomputing them on the test fold would let
    the held-out subject's amplitude distribution influence its own features.
    """

    def __init__(
        self,
        classifier: str = "lda",
        features: FeatureConfig | None = None,
        seed: int = 0,
    ):
        self.classifier_name = classifier
        self.feature_config = features or FeatureConfig()
        self.seed = seed
        self.classifier = build_classifier(classifier, seed)
        self.sigma: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.fit_seconds: float = 0.0

    def _features(self, x: np.ndarray) -> np.ndarray:
        if self.sigma is None:
            raise RuntimeError("fit the pipeline before extracting features")
        return extract(
            x,
            self.sigma,
            zc_threshold_sd=self.feature_config.zc_threshold_sd,
            ssc_threshold_sd=self.feature_config.ssc_threshold_sd,
            wamp_threshold_sd=self.feature_config.wamp_threshold_sd,
            ar_order=self.feature_config.ar_order,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> ClassicalPipeline:
        started = time.perf_counter()
        self.sigma = channel_sigma(x)
        self.classifier.fit(self._features(x), y)
        self.classes_ = np.asarray(self.classifier.classes_)
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            n = self.classes_.size if self.classes_ is not None else 0
            return np.empty((0, n), dtype=np.float32)
        return self.classifier.predict_proba(self._features(x)).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        return self.classifier.predict(self._features(x))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> ClassicalPipeline:
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301 - our own artefact, written by this repo
