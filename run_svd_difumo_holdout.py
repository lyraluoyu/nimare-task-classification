"""Compare TruncatedSVD-256 and DiFuMo-256 on a 10% study holdout."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_difumo
from nilearn.maskers import NiftiMapsMasker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAFeatureExtractor
from nimare.nimads import Studyset


RANDOM_SEED = 13

###############################################################################
# Load the Studyset and configure feature extraction
project_dir = Path(__file__).resolve().parent
studyset_dir = project_dir / "neurostore-studyset-nightly-ml"
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)
studyset = Studyset(studyset_dir)

extractor = MAFeatureExtractor(
    kernel_transformer=MKDAKernel(r=10),
    target_field={"source": "metadata", "field": "comparison_task"},
    missing_coordinates="drop",
)
base_bunch = extractor.transform(studyset)
target = np.asarray(base_bunch.target)
groups = np.asarray(base_bunch.groups)

print(f"Studyset: {studyset.name}")
print(f"Model analyses: {len(target)}")
print(f"Model studies: {len(np.unique(groups))}")

###############################################################################
# Select 10% of the studies within each task for testing
study_table = pd.DataFrame({"study_id": groups, "task": target}).drop_duplicates()

train_studies, test_studies = train_test_split(
    study_table["study_id"],
    test_size=0.10,
    stratify=study_table["task"],
    random_state=RANDOM_SEED,
)
train_mask = np.isin(groups, train_studies)
test_mask = np.isin(groups, test_studies)

print(
    f"Train: {train_mask.sum()} analyses, "
    f"{len(np.unique(groups[train_mask]))} studies"
)
print(
    f"Test: {test_mask.sum()} analyses, "
    f"{len(np.unique(groups[test_mask]))} studies"
)

###############################################################################
# Configure TruncatedSVD and DiFuMo reduction
difumo = fetch_atlas_difumo(
    dimension=256,
    resolution_mm=2,
    data_dir=project_dir / "atlas_cache",
)
atlas_masker = NiftiMapsMasker(
    maps_img=difumo.maps,
    standardize=False,
    resampling_target="data",
    reports=False,
)

workflows = {
    "TruncatedSVD-256": (
        "truncated_svd",
        {"n_components": 256, "random_state": RANDOM_SEED},
    ),
    "DiFuMo-256": (
        "atlas_aggregation",
        {"atlas_masker": atlas_masker, "batch_size": 25},
    ),
}

###############################################################################
# Train both workflows on the same studies and evaluate each test analysis
for name, (reducer, reducer_params) in workflows.items():
    bunch = extractor.transform(
        studyset,
        map_reducer=reducer,
        map_reducer_params=reducer_params,
    )
    pipeline = make_pipeline(
        bunch.preprocessor,
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    )
    pipeline.fit(bunch.data[train_mask], target[train_mask])
    predicted = pipeline.predict(bunch.data[test_mask])
    test_target = target[test_mask]
    classes = pipeline[-1].classes_

    results = pd.DataFrame(
        {
            "task": classes,
            "precision": precision_score(
                test_target,
                predicted,
                labels=classes,
                average=None,
                zero_division=0,
            ),
            "recall": recall_score(
                test_target,
                predicted,
                labels=classes,
                average=None,
                zero_division=0,
            ),
            "support": [np.sum(test_target == task) for task in classes],
        }
    )

    print(f"\n{name} analysis-level results:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Accuracy: {accuracy_score(test_target, predicted):.3f}")

    figure, axis = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        test_target,
        predicted,
        labels=classes,
        normalize="true",
        values_format=".2f",
        xticks_rotation=90,
        cmap="Blues",
        ax=axis,
    )
    axis.set_title(f"{name} normalized confusion matrix")
    figure.tight_layout()
    figure_path = results_dir / f"{name}_confusion_matrix.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved confusion matrix: {figure_path}")
