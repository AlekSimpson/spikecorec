import numpy as np
from scipy.linalg.blas import ssyr, ssymv


class OnlineRLSProbe:
    """Multi-output RLS readout (shared-P recursive least squares with
    forgetting), tuned for fast streaming (batch-size-1) updates.

    Same math as the reference implementation, but the precision matrix P is
    kept symmetric *by construction* so the per-step cost collapses to two
    BLAS calls over a single triangle of P:

    - P is stored Fortran-ordered; only its lower triangle is semantically
      valid (the upper triangle is ignored). Every read of P goes through the
      symmetric matvec ssymv(lower=1).
    - The B=1 update is the Sherman-Morrison rank-1 downdate applied in place
      with ssyr (symmetric rank-1 update) — one triangle touched, no (d, d)
      temporary, and P stays exactly symmetric, so no re-symmetrization pass is
      needed. (The previous version spent ~80% of its time in a transposed
      full-matrix copy, `copyto(buf, P.T)`, re-symmetrizing P every step.)
    - B>1 falls back to the mini-batch matrix form, also reading P
      symmetrically and skipping the symmetrization pass (the rank-B downdate
      is itself symmetric).
    - all state stays float32; standardization uses precomputed 1/std.
    """

    def __init__(self, n_features, n_classes=10, delta=1e-2, forgetting=1.0):
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.delta = float(delta)
        self.forgetting = float(forgetting)
        self.readout_name = "online RLS probe (fast)"
        self._d = self.n_features + 1  # +1 = bias column
        self.mean = np.zeros(self.n_features, dtype=np.float32)
        self.inv_std = np.ones(self.n_features, dtype=np.float32)
        self._buf = None  # lazily-allocated (d, d) scratch, only for B>1 batches
        self.reset_weights()

    # ---------- setup ----------

    def set_normalizer(self, features):
        X = self._as_matrix(features, check_finite=True)
        self.mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        self.inv_std = (1.0 / std).astype(np.float32)

    def reset_weights(self, delta=None):
        if delta is not None:
            self.delta = float(delta)
        prior = max(self.delta, 1e-8)
        self.weights = np.zeros((self._d, self.n_classes), dtype=np.float32)
        # Fortran-ordered so the symmetric BLAS routines (ssyr/ssymv) read and
        # update P in place without copying. Only the lower triangle is kept
        # valid; every read goes through ssymv(lower=1).
        self.P = np.asfortranarray(np.eye(self._d, dtype=np.float32))
        self.P *= np.float32(1.0 / prior)
        self.updates = 0
        self.examples_seen = 0
        self.last_loss = None
        self.last_batch_acc = None

    # ---------- input prep ----------

    def _as_matrix(self, features, check_finite=False):
        X = np.asarray(features, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected (n, {self.n_features}) features, got {X.shape}")
        if check_finite and not np.isfinite(X).all():
            raise ValueError("features contain NaN/Inf; check reservoir alignment.")
        return X

    def _augment(self, features, check_finite=False):
        # standardize + append bias column in one preallocated array
        X = self._as_matrix(features, check_finite=check_finite)
        out = np.empty((X.shape[0], self._d), dtype=np.float32)
        np.subtract(X, self.mean, out=out[:, :-1])
        out[:, :-1] *= self.inv_std
        out[:, -1] = 1.0
        return out

    def _one_hot(self, labels):
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        Y = np.zeros((labels.size, self.n_classes), dtype=np.float32)
        Y[np.arange(labels.size), labels] = 1.0
        return Y, labels

    # ---------- inference ----------

    def logits(self, features):
        return self._augment(features) @ self.weights

    def probabilities(self, features):
        z = self.logits(features)
        z -= z.max(axis=1, keepdims=True)
        np.exp(z, out=z)
        z /= np.maximum(1e-8, z.sum(axis=1, keepdims=True))
        return z

    def predict(self, features):
        return self.logits(features).argmax(axis=1)

    # ---------- training ----------

    def partial_fit(self, features, labels, forgetting=None):
        def resolve_lam(forgetting):
            lam = self.forgetting if forgetting is None else float(forgetting)
            lam = float(np.clip(lam, 0.90, 1.0))
            self.forgetting = lam
            return lam

        X = self._augment(features)
        Y, labels = self._one_hot(labels)
        return self._update(X, Y, labels, resolve_lam(forgetting))

    def _update(self, X, Y, labels, lam):
        # X is already standardized+augmented float32, (B, d)
        B = X.shape[0]
        if B == 1:
            self._update_rank1(X[0], Y[0], lam)
        else:
            self._update_batch(X, Y, lam)

        after = X @ self.weights
        self.last_loss = float(np.mean((Y - after) ** 2))
        self.last_batch_acc = float(np.mean(after.argmax(axis=1) == labels))
        self.updates += 1
        self.examples_seen += B
        return self.last_loss, self.last_batch_acc

    def _update_rank1(self, x, y, lam):
        # Streaming Sherman-Morrison update. x: (d,) augmented feature,
        # y: (C,) one-hot target. P's lower triangle is read with ssymv and
        # downdated in place with ssyr, so P stays exactly symmetric — the
        # dominant cost is just these two single-triangle BLAS calls.
        P = self.P
        Px = ssymv(1.0, P, x, lower=1)                       # P @ x  (lower tri)
        denom = np.float32(lam + float(x @ Px))              # > 0, P is PD
        gain = Px / denom                                    # (d,)
        self.weights += np.outer(gain, y - x @ self.weights)  # (d, C), small
        # lower(P) -= Px Px^T / denom (rank-1 downdate), then P /= lam
        self.P = ssyr(np.float32(-1.0 / denom), Px, a=P, lower=1, overwrite_a=1)
        if lam != 1.0:
            self.P *= np.float32(1.0 / lam)

    def _update_batch(self, X, Y, lam):
        # Mini-batch (B>1) matrix form. P is read symmetrically (one ssymv per
        # row) and the symmetrization pass is dropped: the rank-B downdate
        # K_T.T @ XP is itself symmetric, so P's lower triangle stays exact.
        B, d = X.shape
        if self._buf is None or self._buf.shape[0] != d:
            self._buf = np.empty((d, d), dtype=np.float32)
        XP = np.empty((B, d), dtype=np.float32)
        for i in range(B):
            XP[i] = ssymv(1.0, self.P, X[i], lower=1)
        error = Y - X @ self.weights                  # (B, C)
        S = XP @ X.T                                  # (B, B)
        S.flat[:: B + 1] += lam
        try:
            K_T = np.linalg.solve(S, XP)              # (B, d)
        except np.linalg.LinAlgError:
            S.flat[:: B + 1] += 1e-5 * max(1.0, float(np.trace(S)) / B)
            K_T = np.linalg.solve(S, XP)
        self.weights += K_T.T @ error                 # (d, C), small
        np.matmul(K_T.T, XP, out=self._buf)           # symmetric rank-B downdate
        self.P -= self._buf
        if lam != 1.0:
            self.P *= np.float32(1.0 / lam)