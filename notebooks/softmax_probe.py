import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm is optional; train_epochs degrades to a plain loop
    tqdm = None


class OnlineSoftmaxProbe:
    """Single-layer softmax (multinomial logistic-regression) readout, tuned
    for fast streaming (batch-size-1) SGD updates.

    Same model as the reference implementation -- a linear layer
    ``logits = [x | 1] @ W`` trained with softmax cross-entropy -- but rewritten
    so the per-step path allocates almost nothing:

    - features are standardized with a *precomputed reciprocal* std (a multiply,
      not a divide) straight into one preallocated augmented buffer that already
      carries the bias column (the reference rebuilt this with ``column_stack``
      every step);
    - the B=1 gradient never materializes a one-hot or a (d, C) gradient: the
      softmax cross-entropy gradient on the logits is just ``p - e_label``, so we
      do ``p[label] -= 1`` and apply the rank-1 outer product ``x (x) (p - e)``
      to W directly;
    - L2 is folded in as decoupled weight decay on the non-bias rows, which for
      plain SGD is *algebraically identical* to adding ``l2 * W`` to the gradient
      (``W(1 - lr*l2) - lr*g``), but avoids forming the extra array;
    - the reported loss/accuracy are *prequential* -- measured on the pre-update
      logits we already compute for the gradient -- which is the honest online
      metric and costs no second matmul. (Note: OnlineRLSProbe reports its
      metric *after* the update, so per-step numbers are not directly
      comparable, though both converge to the same fit.)

    All state stays float32. Weights are zero-initialized: softmax regression is
    convex, so this is well-posed and fully reproducible (the per-class
    gradients differ from step one, so there is no symmetry to break).

    Interface mirrors OnlineRLSProbe so the two probes are drop-in
    interchangeable in the training loop.
    """

    def __init__(self, n_features, n_classes=10, lr=5e-2, l2=1e-4):
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.readout_name = "online softmax probe (fast)"
        self._d = self.n_features + 1  # +1 = bias column
        self.mean = np.zeros(self.n_features, dtype=np.float32)
        self.inv_std = np.ones(self.n_features, dtype=np.float32)
        self.reset_weights()

    # ---------- setup ----------

    def set_normalizer(self, features):
        X = self._as_matrix(features, check_finite=True)
        self.mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        self.inv_std = (1.0 / std).astype(np.float32)

    def reset_weights(self):
        self.weights = np.zeros((self._d, self.n_classes), dtype=np.float32)
        self.updates = 0
        self.examples_seen = 0
        self.last_loss = None
        self.last_batch_acc = None

    # ---------- input prep ---------- (identical layout to OnlineRLSProbe)

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

    # ---------- inference ----------

    def logits(self, features):
        return self._augment(features) @ self.weights

    def probabilities(self, features):
        z = self.logits(features)
        return self._softmax_rows(z)

    def predict(self, features):
        return self.logits(features).argmax(axis=1)

    @staticmethod
    def _softmax_rows(z):
        # numerically-stable row softmax, in place on a (B, C) float array
        z -= z.max(axis=1, keepdims=True)
        np.exp(z, out=z)
        z /= np.maximum(np.float32(1e-8), z.sum(axis=1, keepdims=True))
        return z

    # ---------- training ----------

    def partial_fit(self, features, labels, lr=None, l2=None):
        X = self._augment(features)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        step_lr = self.lr if lr is None else float(lr)
        step_l2 = self.l2 if l2 is None else float(l2)
        if X.shape[0] == 1:
            self._update_single(X[0], int(labels[0]), step_lr, step_l2)
        else:
            self._update_batch(X, labels, step_lr, step_l2)
        self.updates += 1
        self.examples_seen += labels.size
        return self.last_loss, self.last_batch_acc

    def _update_single(self, x, label, lr, l2):
        # x: (d,) standardized+augmented feature. One SGD step on a single
        # example. dL/dlogits = p - e_label, so the data gradient is the rank-1
        # outer product x (x) (p - e_label); no one-hot or (d, C) grad needed.
        probs = (x @ self.weights)[None, :]        # (1, C)
        self._softmax_rows(probs)
        p = probs[0]
        self.last_loss = float(-np.log(p[label] + 1e-8))
        self.last_batch_acc = float(p.argmax() == label)
        p[label] -= np.float32(1.0)                 # p - e_label, in place
        if l2 != 0.0:                               # decoupled L2 == L2-in-grad
            self.weights[:-1] *= np.float32(1.0 - lr * l2)
        self.weights -= np.float32(lr) * np.outer(x, p)

    def _update_batch(self, X, labels, lr, l2):
        # Mini-batch SGD. Mean cross-entropy gradient over the batch; L2 added to
        # the non-bias rows (per-step, not per-example, matching the reference).
        B = X.shape[0]
        probs = X @ self.weights                    # (B, C)
        self._softmax_rows(probs)
        rows = np.arange(B)
        self.last_loss = float(-np.log(probs[rows, labels] + 1e-8).mean())
        self.last_batch_acc = float(np.mean(probs.argmax(axis=1) == labels))
        probs[rows, labels] -= np.float32(1.0)      # p - Y, in place
        grad = (X.T @ probs) * np.float32(1.0 / B)  # (d, C)
        if l2 != 0.0:
            grad[:-1] += np.float32(l2) * self.weights[:-1]
        self.weights -= np.float32(lr) * grad

    def train_epochs(self, features, labels, epochs=1, batch_size=128, seed=0,
                     progress_desc=None):
        # Convenience epoch loop over an in-memory feature/label set (mirrors the
        # reference). The streaming use case calls partial_fit directly instead.
        rng = np.random.default_rng(seed)
        features = self._as_matrix(features)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        n = labels.size
        history = []
        epoch_iter = range(int(epochs))
        if progress_desc is not None and tqdm is not None:
            epoch_iter = tqdm(epoch_iter, total=int(epochs), desc=progress_desc, leave=False)
        for epoch in epoch_iter:
            order = rng.permutation(n)
            for start in range(0, n, int(batch_size)):
                idx = order[start:start + int(batch_size)]
                self.partial_fit(features[idx], labels[idx])
            history.append((epoch + 1, self.last_loss, self.last_batch_acc))
        return history
