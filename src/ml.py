"""
Machine learning components for active learning.

Two surrogate architectures for mapping dye volumes to RGB:
- ``IndependentMultiOutputGP``: 3 separate GPs (GP_R, GP_G, GP_B).
- ``CorrelatedMultiOutputGP``: single multi-output GP that learns
  correlations between R, G, B channels (recommended — better with
  few observations).

Acquisition minimizes predicted RGB distance to target color,
subject to the volume constraint Vr + Vg + Vb = total_volume.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

try:
    import torch
    from botorch.models import SingleTaskGP, KroneckerMultiTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
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
# Simplex sampling (volume constraint)
# ======================================================================

def sample_simplex(
    n_samples: int,
    total_volume: float = 200.0,
    d: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample n points uniformly from the simplex
    {(v1,...,vd) : vi >= 0, sum(vi) = total_volume}.

    Uses symmetric Dirichlet(1,...,1) distribution.

    Returns:
        (n_samples, d) array of volume combinations.
    """
    if rng is None:
        rng = np.random.default_rng()
    proportions = rng.dirichlet(np.ones(d), size=n_samples)
    return proportions * total_volume


# ======================================================================
# 3-GP surrogate model
# ======================================================================

class IndependentMultiOutputGP:
    """
    3 independent GPs mapping dye volumes to RGB channel values.

    GP_R: (Vr, Vg, Vb) → predicted Red channel   (0-255)
    GP_G: (Vr, Vg, Vb) → predicted Green channel  (0-255)
    GP_B: (Vr, Vg, Vb) → predicted Blue channel   (0-255)

    Each GP is a BoTorch SingleTaskGP fitted independently.
    """

    CHANNEL_NAMES = ("R", "G", "B")

    def __init__(
        self,
        bounds: Optional[np.ndarray] = None,
        total_volume: float = 200.0,
    ):
        if not _BOTORCH_AVAILABLE:
            raise ImportError("botorch/gpytorch/torch required for IndependentMultiOutputGP")
        self.bounds = (
            torch.tensor(bounds, dtype=torch.double)
            if bounds is not None
            else torch.tensor([[0.0, 0.0, 0.0], [200.0, 200.0, 200.0]], dtype=torch.double)
        )
        self.total_volume = total_volume
        self.models: List[Optional[SingleTaskGP]] = [None, None, None]
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None  # (n, 3) RGB

    def fit(self, X: np.ndarray, Y_rgb: np.ndarray):
        """
        Fit all 3 GPs on training data.

        Args:
            X: (n, 3) array of dye volumes [Vr, Vg, Vb].
            Y_rgb: (n, 3) array of observed RGB values [R, G, B].
        """
        self.train_X = torch.tensor(np.atleast_2d(X), dtype=torch.double)
        self.train_Y = torch.tensor(np.atleast_2d(Y_rgb), dtype=torch.double)

        for i in range(3):
            y_i = self.train_Y[:, i : i + 1]  # (n, 1)
            model = SingleTaskGP(
                self.train_X, y_i,
                input_transform=Normalize(d=self.train_X.shape[1]),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            self.models[i] = model
            print(f"[ML] GP_{self.CHANNEL_NAMES[i]} fitted on {self.train_X.shape[0]} points")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict RGB for input volumes.

        Args:
            X: (n, 3) array of dye volumes.

        Returns:
            mean_rgb: (n, 3) predicted RGB means.
            std_rgb:  (n, 3) predicted RGB standard deviations.
        """
        if any(m is None for m in self.models):
            raise RuntimeError("Models not fitted. Call fit() first.")
        X_t = torch.tensor(np.atleast_2d(X), dtype=torch.double)
        means, stds = [], []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                posterior = model.posterior(X_t)
            means.append(posterior.mean.squeeze(-1).numpy())
            stds.append(posterior.variance.sqrt().squeeze(-1).numpy())
        return np.column_stack(means), np.column_stack(stds)

    def update(self, X_new: np.ndarray, Y_new_rgb: np.ndarray):
        """Add new observations and refit all 3 GPs."""
        X_all = np.vstack([self.train_X.numpy(), np.atleast_2d(X_new)])
        Y_all = np.vstack([self.train_Y.numpy(), np.atleast_2d(Y_new_rgb)])
        self.fit(X_all, Y_all)


# ======================================================================
# Correlated multi-output GP (KroneckerMultiTaskGP)
# ======================================================================

class CorrelatedMultiOutputGP:
    """
    Multi-output GP that learns correlations between R, G, B channels.

    Uses BoTorch's KroneckerMultiTaskGP (Intrinsic Coregionalization Model)
    which is efficient when all outputs are observed at every input.

    This is better than 3 independent GPs when data is scarce (< 20 points)
    because observing one channel informs the others through the learned
    inter-task covariance.

    Interface is identical to :class:`IndependentMultiOutputGP`.
    """

    def __init__(
        self,
        bounds: Optional[np.ndarray] = None,
        total_volume: float = 200.0,
    ):
        if not _BOTORCH_AVAILABLE:
            raise ImportError("botorch/gpytorch/torch required for CorrelatedMultiOutputGP")
        self.bounds = (
            torch.tensor(bounds, dtype=torch.double)
            if bounds is not None
            else torch.tensor([[0.0, 0.0, 0.0], [200.0, 200.0, 200.0]], dtype=torch.double)
        )
        self.total_volume = total_volume
        self.model: Optional[KroneckerMultiTaskGP] = None
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None

    def fit(self, X: np.ndarray, Y_rgb: np.ndarray):
        """
        Fit the correlated multi-output GP.

        Args:
            X: (n, 3) array of dye volumes [Vr, Vg, Vb].
            Y_rgb: (n, 3) array of observed RGB values [R, G, B].
        """
        self.train_X = torch.tensor(np.atleast_2d(X), dtype=torch.double)
        self.train_Y = torch.tensor(np.atleast_2d(Y_rgb), dtype=torch.double)

        self.model = KroneckerMultiTaskGP(
            self.train_X, self.train_Y,
            input_transform=Normalize(d=self.train_X.shape[1]),
            outcome_transform=Standardize(m=self.train_Y.shape[1]),
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        print(f"[ML] CorrelatedGP fitted on {self.train_X.shape[0]} points (3 correlated outputs)")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict RGB for input volumes.

        Args:
            X: (n, 3) array of dye volumes.

        Returns:
            mean_rgb: (n, 3) predicted RGB means.
            std_rgb:  (n, 3) predicted RGB standard deviations.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_t = torch.tensor(np.atleast_2d(X), dtype=torch.double)
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_t)
        mean = posterior.mean.squeeze(0).numpy()
        std = posterior.variance.sqrt().squeeze(0).numpy()
        # Ensure 2D even for single-point prediction
        if mean.ndim == 1:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return mean, std

    def update(self, X_new: np.ndarray, Y_new_rgb: np.ndarray):
        """Add new observations and refit."""
        X_all = np.vstack([self.train_X.numpy(), np.atleast_2d(X_new)])
        Y_all = np.vstack([self.train_Y.numpy(), np.atleast_2d(Y_new_rgb)])
        self.fit(X_all, Y_all)


# ======================================================================
# Surrogate factory
# ======================================================================

def create_surrogate(
    model_type: str = "correlated_gp",
    bounds: Optional[np.ndarray] = None,
    total_volume: float = 200.0,
):
    """
    Create a surrogate model by name.

    Args:
        model_type: ``"correlated_gp"`` (default) or ``"independent_gp"``.
        bounds: (2, d) array of [min, max] per dimension.
        total_volume: Sum constraint for volumes.
    """
    if model_type == "correlated_gp":
        return CorrelatedMultiOutputGP(bounds=bounds, total_volume=total_volume)
    elif model_type == "independent_gp":
        return IndependentMultiOutputGP(bounds=bounds, total_volume=total_volume)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'correlated_gp' or 'independent_gp'.")


# ======================================================================
# Acquisition function (multi-output, simplex-constrained)
# ======================================================================

class AcquisitionFunction:
    """
    Suggests next dye volumes by minimizing predicted RGB distance
    to target, subject to Vr + Vg + Vb = total_volume.

    Two modes:
    - "EI": Thompson sampling — draw from GP posteriors, pick the
      candidate whose sampled RGB is closest to target.
    - "UCB": Lower Confidence Bound on distance — predicted distance
      minus beta * uncertainty (optimistic about being close).
    """

    def __init__(
        self,
        kind: str = "EI",
        target_rgb: Optional[np.ndarray] = None,
        total_volume: float = 200.0,
    ):
        if not _BOTORCH_AVAILABLE:
            raise ImportError("botorch required for AcquisitionFunction")
        if kind not in ("EI", "UCB"):
            raise ValueError(f"Unsupported acquisition kind: {kind}")
        self.kind = kind
        self.target_rgb = np.asarray(target_rgb, dtype=float) if target_rgb is not None else None
        self.total_volume = total_volume

    def suggest(
        self,
        surrogate: IndependentMultiOutputGP,
        n_candidates: int = 1,
        n_random: int = 10000,
    ) -> np.ndarray:
        """
        Suggest next (Vr, Vg, Vb) to test.

        Generates random candidates on the simplex, evaluates each
        using the 3 GPs, and returns the best candidate(s).

        Args:
            surrogate: Fitted surrogate (IndependentMultiOutputGP or CorrelatedMultiOutputGP).
            n_candidates: Number of suggestions to return.
            n_random: Number of random simplex candidates to evaluate.

        Returns:
            (n_candidates, 3) array of suggested dye volumes.
        """
        # Check fitted — works for both Independent (has .models list) and Correlated (has .model)
        if hasattr(surrogate, "models"):
            if any(m is None for m in surrogate.models):
                raise RuntimeError("Surrogate not fitted.")
        elif getattr(surrogate, "model", None) is None:
            raise RuntimeError("Surrogate not fitted.")
        if self.target_rgb is None:
            raise RuntimeError("target_rgb not set.")

        rng = np.random.default_rng()
        candidates = sample_simplex(n_random, self.total_volume, d=3, rng=rng)

        mean_rgb, std_rgb = surrogate.predict(candidates)

        if self.kind == "EI":
            # Thompson sampling: sample from posterior, minimize distance
            sampled_rgb = mean_rgb + std_rgb * rng.standard_normal(mean_rgb.shape)
            sampled_rgb = np.clip(sampled_rgb, 0, 255)
            distances = np.sqrt(np.sum((sampled_rgb - self.target_rgb) ** 2, axis=1))
        else:
            # LCB on distance: optimistic about being close to target
            pred_distances = np.sqrt(np.sum((mean_rgb - self.target_rgb) ** 2, axis=1))
            uncertainty = np.sqrt(np.sum(std_rgb ** 2, axis=1))
            distances = pred_distances - 2.0 * uncertainty

        best_indices = np.argsort(distances)[:n_candidates]
        suggested = candidates[best_indices]

        for idx in best_indices:
            v = candidates[idx]
            m = mean_rgb[idx]
            d = distances[idx]
            print(
                f"[ML] Suggested: Vr={v[0]:.1f}, Vg={v[1]:.1f}, Vb={v[2]:.1f} "
                f"| Pred RGB=({m[0]:.0f},{m[1]:.0f},{m[2]:.0f}) | Score={d:.1f}"
            )

        return suggested
