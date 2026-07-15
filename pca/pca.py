import numpy as np
from numpy.typing import NDArray


def pca(
    X: NDArray[np.float32], n_components: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Principal Component Analysis

    Args:
        X: Input data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep

    Returns:
        mean: Mean of the input data
        components: Principal components matrix of shape (n_components, n_features)
        singular_values: Singular values of shape (n_components,)


    """
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    _, S, Vh = np.linalg.svd(X_centered, full_matrices=False)

    components = Vh[:n_components]
    singular_values = S[:n_components]

    return mean, components, singular_values


def pca_transform(
    X: NDArray[np.float32], mean: NDArray[np.float32], components: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Transform the input data using the principal components

    Args:
        X: Input data matrix of shape (n_samples, n_features)
        mean: Mean of the input data
        components: Principal components matrix of shape (n_components, n_features)

    Returns:
        X_transformed: Transformed data matrix of shape (n_samples, n_components)
    """
    return (X - mean) @ components.T


def pca_reconstruct(
    Z: NDArray[np.float32], mean: NDArray[np.float32], components: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Reconstruct features from PCA codes: shape (n_samples, n_features)."""
    return Z @ components + mean


def pca_residual(
    X: NDArray[np.float32], mean: NDArray[np.float32], components: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Per-sample L2 reconstruction error.
    High residual => frame is poorly explained by the training PCA basis (OOD).
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    Z = pca_transform(X, mean, components)
    X_hat = pca_reconstruct(Z, mean, components)
    return np.linalg.norm(X - X_hat, axis=1).astype(np.float32)