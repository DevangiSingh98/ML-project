"""
Preprocessing and model pipeline construction for subject-independent emotion recognition.

Guarantees:
- Feature scaling (StandardScaler) is strictly enclosed within a scikit-learn Pipeline
  so it is fitted exclusively on outer training data during each fold.
- Zero data from held-out subjects influences scaling statistics.
"""

from typing import Union
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_baseline_pipeline(
    C: float = 1.0,
    gamma: Union[str, float] = "scale",
    kernel: str = "rbf",
    cache_size: int = 1000,
) -> Pipeline:
    """
    Construct an RBF-SVM baseline pipeline with training-only StandardScaler.
    
    Parameters:
        C: Regularization parameter.
        gamma: Kernel coefficient for 'rbf'.
        kernel: Kernel type ('rbf').
        cache_size: Kernel cache size in MB (improves training speed on large sample counts).
        
    Returns:
        scikit-learn Pipeline instance with ('scaler', StandardScaler()) and ('svc', SVC(...)).
    """
    scaler = StandardScaler()
    svc = SVC(
        C=C,
        gamma=gamma,
        kernel=kernel,
        probability=False,  # Probability estimation strictly deferred to Phase D group-safe calibration
        cache_size=cache_size,
    )

    pipeline = Pipeline([
        ("scaler", scaler),
        ("svc", svc),
    ])

    return pipeline
