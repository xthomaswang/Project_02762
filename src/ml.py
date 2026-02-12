"""
Machine learning components for active learning.

Provides a Gaussian Process surrogate model and acquisition functions
for iteratively optimizing dye formulations toward target colors.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

try:
    import torch
    import gpytorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


# ======================================================================
# Color distance
# ======================================================================

def color_distance(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    """Euclidean distance between two RGB vectors (each shape (3,))."""
    return float(euclidean(np.asarray(rgb_a), np.asarray(rgb_b)))


# ======================================================================
# Surrogate model
# ======================================================================

class SurrogateModel:
    """
    Gaussian Process surrogate mapping dye volumes (R, G, B) to observed
    color distance from target. Wraps BoTorch SingleTaskGP.
    """

    def __init__(self, bounds: Optional[np.ndarray] = None):
        """
        Args:
            bounds: (2, d) array with lower/upper bounds for each input dim.
                    Defaults to [[0,0,0],[300,300,300]] (uL).
        """
        if not _BOTORCH_AVAILABLE:
            raise ImportError("botorch/gpytorch/torch required for SurrogateModel")
        self.bounds = (
            torch.tensor(bounds, dtype=torch.double)
            if bounds is not None
            else torch.tensor([[0.0, 0.0, 0.0], [300.0, 300.0, 300.0]], dtype=torch.double)
        )
        self.model: Optional[SingleTaskGP] = None
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit (or refit) the GP on training data.

        Args:
            X: (n, d) array of input dye volumes.
            Y: (n, 1) array of observed color distances.
        """
        self.train_X = torch.tensor(X, dtype=torch.double)
        self.train_Y = torch.tensor(Y, dtype=torch.double).unsqueeze(-1) if Y.ndim == 1 else torch.tensor(Y, dtype=torch.double)
        self.model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) predictions at X."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X_t = torch.tensor(X, dtype=torch.double)
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_t)
        mean = posterior.mean.numpy().squeeze()
        std = posterior.variance.sqrt().numpy().squeeze()
        return mean, std

    def update(self, X_new: np.ndarray, Y_new: np.ndarray):
        """Add new observations and refit."""
        X_new_t = torch.tensor(X_new, dtype=torch.double)
        Y_new_t = torch.tensor(Y_new, dtype=torch.double).unsqueeze(-1) if Y_new.ndim == 1 else torch.tensor(Y_new, dtype=torch.double)
        if self.train_X is not None:
            self.train_X = torch.cat([self.train_X, X_new_t])
            self.train_Y = torch.cat([self.train_Y, Y_new_t])
        else:
            self.train_X = X_new_t
            self.train_Y = Y_new_t
        self.fit(self.train_X.numpy(), self.train_Y.squeeze(-1).numpy())


# ======================================================================
# Acquisition functions
# ======================================================================

class AcquisitionFunction:
    """Wrapper around BoTorch acquisition functions."""

    def __init__(self, kind: str = "EI"):
        """
        Args:
            kind: 'EI' (Expected Improvement) or 'UCB' (Upper Confidence Bound).
        """
        if not _BOTORCH_AVAILABLE:
            raise ImportError("botorch required for AcquisitionFunction")
        if kind not in ("EI", "UCB"):
            raise ValueError(f"Unsupported acquisition kind: {kind}")
        self.kind = kind

    def suggest(self, surrogate: SurrogateModel,
                n_candidates: int = 1) -> np.ndarray:
        """
        Suggest next experiment(s) to run.

        Args:
            surrogate: Fitted SurrogateModel.
            n_candidates: Number of candidates to return.

        Returns:
            (n_candidates, d) array of suggested dye volumes.
        """
        if surrogate.model is None:
            raise RuntimeError("Surrogate model not fitted.")

        # We minimize color distance, so best_f is the minimum observed value
        best_f = surrogate.train_Y.min()

        if self.kind == "EI":
            acqf = ExpectedImprovement(model=surrogate.model, best_f=best_f, maximize=False)
        else:
            acqf = UpperConfidenceBound(model=surrogate.model, beta=2.0)

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=surrogate.bounds,
            q=n_candidates,
            num_restarts=10,
            raw_samples=256,
        )
        return candidates.detach().numpy()
