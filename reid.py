# reid.py — Re-Identification: extract a feature embedding from a person crop

import numpy as np

def load_reid(model_name: str = "mobilenet", device: str = "cpu"):
    """Load the ReID model. Call this once at startup."""
    from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder
    gpu = True if device.startswith("cuda") else False
    
    # We load the MobileNetv2 embedder bundled with deep_sort_realtime.
    # It has a pre-trained ReID checkpoint (mobilenetv2_bottleneck_wts.pt).
    embedder = MobileNetv2_Embedder(
        half=False,  # Use full precision
        max_batch_size=1,
        bgr=True,    # CV2 crops are BGR
        gpu=gpu
    )
    return embedder

def get_embedding(reid_model, crop: np.ndarray) -> np.ndarray:
    """
    Extract a normalized feature vector from a person crop (BGR image).
    Returns a 1-D float32 numpy array.
    """
    # MobileNetv2_Embedder predict expects a list of BGR images
    features = reid_model.predict([crop])
    vec = np.asarray(features[0], dtype=np.float32)

    # Normalize to unit length so cosine similarity = dot product
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec = vec / norm
    return vec
