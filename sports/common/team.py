from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
from cuml.cluster import KMeans
from cuml.manifold import UMAP
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """

    _features_model: SiglipVisionModel | None = None
    _processor: AutoProcessor | None = None

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size

        if TeamClassifier._features_model is None or TeamClassifier._processor is None:
            TeamClassifier._features_model = SiglipVisionModel.from_pretrained(
                SIGLIP_MODEL_PATH
            )
            TeamClassifier._processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

        # share global model/processor, chỉ chuyển device nếu cần
        self.features_model = TeamClassifier._features_model.to(self.device)
        self.processor = TeamClassifier._processor
        self.reducer = UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        labels = self.cluster_model.predict(projections)
        # cuML trả về cupy array, convert sang numpy nếu cần
        return labels.get() if hasattr(labels, "get") else np.asarray(labels)

    def fit_predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Fit on crops and return cluster labels, computing embeddings only once.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        labels = self.cluster_model.labels_
        return labels.get() if hasattr(labels, "get") else np.asarray(labels)
