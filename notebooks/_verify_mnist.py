"""Standalone replica of mnist_testing.ipynb to verify softmax-probe accuracy.

Replicates the notebook's pipeline EXACTLY, with the one engine-contract fix the
notebook is missing: step_simulation must receive override_input_neurons so the
input neurons get merged into the active set each tick (otherwise the active set
stays empty after reset_state and the reservoir never spikes -- see the engine's
gpu_step, which only evaluates the active set). This mirrors the reference
notebook's advance_static_input(values, tick, input_neurons).

Usage:
  _verify_mnist.py --n-train 2000 --n-test 2000        # quick subset
  _verify_mnist.py                                      # full 60k/10k
  _verify_mnist.py --no-override                        # reproduce the bug
"""
import argparse, sys, time
import numpy as np
import mnist
mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
import spikecorec as spc
from spikecorec import small_world_torus, SpikeEngine
sys.path.insert(0, "/Users/aleksimpson/Desktop/projects/spikecorec/notebooks")
from softmax_probe import OnlineSoftmaxProbe
from rls_probe import OnlineRLSProbe
from tqdm.auto import tqdm

# ---- hyperparameters (verbatim from the notebook) ----
RANK = 1
MNIST_DATA_SCALE = 28
SPIKE_TAU = 10.0
INPUT_GAIN = 1.15
RECURRENT_SCALE = 1.03
VOLTAGE_SCALE = 1.0
pre_steps, on_steps, off_steps = 3, 32, 8
INPUT_NEURON_COUNT = MNIST_DATA_SCALE * MNIST_DATA_SCALE
BACKGROUND_NOISE_INPUT = np.full((INPUT_NEURON_COUNT,), 0.1)
BLANK_INPUT = np.zeros((INPUT_NEURON_COUNT,))

INPUT_NEURON_IDS = None  # set in init; used as override_input_neurons


def initialize_network_for_mnist(reservoir_scale, seed, learning_rate=0.00222,
                                 decay_rate=0.22, resting_mp=0.1,
                                 randomized=False, rec_scale=RECURRENT_SCALE,
                                 rank=RANK):
    global INPUT_NEURON_IDS
    side_length = MNIST_DATA_SCALE * reservoir_scale
    network = small_world_torus(side_length, seed=seed)
    engine = SpikeEngine(network, [side_length, side_length], rank=rank,
                         resting_mp=resting_mp, decay_rate=decay_rate,
                         learning_rate=learning_rate)
    scale = side_length // MNIST_DATA_SCALE
    offset = scale // 2
    center_neuron_ids = [
        (scale * row + offset) * side_length + (scale * col + offset)
        for row in range(MNIST_DATA_SCALE) for col in range(MNIST_DATA_SCALE)
    ]
    engine.set_input_neurons(center_neuron_ids)
    if randomized:
        # reference-style: heterogeneous per-edge weights scaled to target RMS
        engine.scale_randomized_weights_near_bifurcation(scale=rec_scale, freeze_learning=True)
    else:
        # notebook as-written: single constant weight on every edge (degenerate)
        engine.scale_uniform_weights_near_bifurcation(scale=rec_scale, freeze_learning=True)
    INPUT_NEURON_IDS = center_neuron_ids
    return engine


def prepare_image_input(images, input_shape=(MNIST_DATA_SCALE, MNIST_DATA_SCALE)):
    rows, columns = input_shape
    X = images.astype(np.float32) / 255.0
    return np.ascontiguousarray(X.reshape(len(images), rows * columns), dtype=np.float32)


# readout window for the reference-style average: start once the image is on,
# accumulate every tick through to the end (matches the reference's windowed
# feature average, vs the notebook's single final-tick read).
READOUT_START = pre_steps


def run_image(engine, flat_image, override=True, windowed=False):
    engine.reset_state(0)
    ov = INPUT_NEURON_IDS if override else []
    driven = flat_image * np.float32(INPUT_GAIN)
    tick = 0
    acc = None; count = 0
    def maybe_read(t):
        nonlocal acc, count
        if windowed and t >= READOUT_START:
            v = engine.get_reservoir_features_vector(t, SPIKE_TAU, VOLTAGE_SCALE)
            acc = v.copy() if acc is None else acc + v
            count += 1
    for _ in range(pre_steps):
        engine.step_simulation(BACKGROUND_NOISE_INPUT, tick=tick, override_input_neurons=ov); maybe_read(tick); tick += 1
    for _ in range(on_steps):
        engine.step_simulation(driven, tick=tick, override_input_neurons=ov); maybe_read(tick); tick += 1
    for _ in range(off_steps):
        engine.step_simulation(BLANK_INPUT, tick=tick, override_input_neurons=ov); maybe_read(tick); tick += 1
    final = engine.get_reservoir_features_vector(tick - 1, SPIKE_TAU, VOLTAGE_SCALE)
    if windowed:
        return final, (acc / max(1, count)).astype(np.float32)
    return final


def collect_features(engine, images, n=256, seed=0, override=True):
    rng = np.random.default_rng(seed)
    n = min(int(n), len(images))
    idx = rng.choice(len(images), size=n, replace=False)
    flats = prepare_image_input(images[idx])
    feats = np.empty((n, 2 * engine.neuron_count + 1), dtype=np.float32)
    for i in range(n):
        feats[i] = run_image(engine, flats[i], override=override)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--no-override", action="store_true")
    ap.add_argument("--seed", type=int, default=1042)
    ap.add_argument("--reservoir-scale", type=int, default=3)
    ap.add_argument("--cache", type=str, default=None, help="npz path to cache/load features")
    ap.add_argument("--randomized", action="store_true", help="randomized (heterogeneous) reservoir weights")
    ap.add_argument("--rec-scale", type=float, default=RECURRENT_SCALE)
    ap.add_argument("--rank", type=int, default=RANK)
    args = ap.parse_args()
    override = not args.no_override

    train_images = mnist.train_images(); train_labels = mnist.train_labels()
    test_images = mnist.test_images(); test_labels = mnist.test_labels()
    n_train = len(train_labels) if args.n_train is None else min(args.n_train, len(train_labels))
    n_test = len(test_labels) if args.n_test is None else min(args.n_test, len(test_labels))

    engine = initialize_network_for_mnist(reservoir_scale=args.reservoir_scale, seed=args.seed,
                                          randomized=args.randomized, rec_scale=args.rec_scale,
                                          rank=args.rank)
    n_features = engine.neuron_count * 2 + 1
    print(f"neuron_count={engine.neuron_count} n_features={n_features} override={override} "
          f"randomized={args.randomized} rec_scale={args.rec_scale} rank={args.rank} "
          f"n_train={n_train} n_test={n_test}", flush=True)

    train_flats = prepare_image_input(train_images[:n_train])
    test_flats = prepare_image_input(test_images[:n_test])

    if args.cache:
        import os
        if os.path.exists(args.cache):
            d = np.load(args.cache)
            Xtr, Wtr, ytr = d["Xtr"], d["Wtr"], d["ytr"]
            Xte, Wte, yte = d["Xte"], d["Wte"], d["yte"]
            print("loaded cached features", Xtr.shape, Xte.shape, flush=True)
        else:
            Xtr, Wtr = _extract2(engine, train_flats, override, "extract-train")
            Xte, Wte = _extract2(engine, test_flats, override, "extract-test")
            ytr = train_labels[:n_train].astype(np.int64); yte = test_labels[:n_test].astype(np.int64)
            np.savez(args.cache, Xtr=Xtr, Wtr=Wtr, ytr=ytr, Xte=Xte, Wte=Wte, yte=yte)
        print("\n########## FINAL-TICK read (notebook as-written) ##########", flush=True)
        _diag(Xtr, ytr)
        _train_and_eval_cached(Xtr, ytr, Xte, yte, n_features)
        print("\n########## WINDOWED-AVERAGE read (reference-style) ##########", flush=True)
        _diag(Wtr, ytr)
        _train_and_eval_cached(Wtr, ytr, Wte, yte, n_features)
        return

    # streaming path == notebook path
    probe = OnlineSoftmaxProbe(n_features, lr=1e-2, l2=1e-3)
    calib = collect_features(engine, train_images, n=256, seed=0, override=override)
    probe.set_normalizer(calib)
    _diag(calib, None)

    t0 = time.time()
    for i in tqdm(range(n_train), desc="train"):
        f = run_image(engine, train_flats[i], override=override)
        probe.partial_fit(f, int(train_labels[i]))
    print(f"train done in {time.time()-t0:.1f}s", flush=True)

    correct = 0
    for i in tqdm(range(n_test), desc="test"):
        f = run_image(engine, test_flats[i], override=override)
        if int(probe.predict(f)[0]) == int(test_labels[i]):
            correct += 1
    acc = correct / n_test
    print(f"\nSOFTMAX PROBE TEST ACCURACY: {acc*100:.2f}%  ({correct}/{n_test})", flush=True)
    print("RESULT:", "PASS >=94%" if acc >= 0.94 else "BELOW 94%", flush=True)


def _extract2(engine, flats, override, desc):
    d = 2 * engine.neuron_count + 1
    fin = np.empty((len(flats), d), dtype=np.float32)
    win = np.empty((len(flats), d), dtype=np.float32)
    for i in tqdm(range(len(flats)), desc=desc):
        fin[i], win[i] = run_image(engine, flats[i], override=override, windowed=True)
    return fin, win


def _diag(X, y):
    nz = np.count_nonzero(X, axis=1)
    half = (X.shape[1] - 1) // 2
    spk_nz = np.count_nonzero(X[:, :half], axis=1)
    print(f"  feat diag: mean nonzero/row={nz.mean():.1f} spike-half nonzero/row={spk_nz.mean():.1f} "
          f"min={X.min():.3f} max={X.max():.3f}", flush=True)
    if y is not None and len(np.unique(y)) > 1:
        # mean within-class vs between-class L2 on a small sample
        idx = np.arange(min(200, len(y)))
        Xs, ys = X[idx], y[idx]
        D = np.sqrt(((Xs[:, None] - Xs[None]) ** 2).sum(-1))
        same = ys[:, None] == ys[None]
        np.fill_diagonal(same, False)
        print(f"  within-class L2={D[same].mean():.3f}  between-class L2={D[~same].mean():.3f}", flush=True)


def _eval(probe, Xte, yte):
    return float((probe.predict(Xte) == yte).mean())


def _train_and_eval_cached(Xtr, ytr, Xte, yte, n_features):
    # 1) single online pass, notebook config (lr=1e-2, l2=1e-3)
    p = OnlineSoftmaxProbe(n_features, lr=1e-2, l2=1e-3)
    p.set_normalizer(Xtr[:256])
    for i in range(len(ytr)):
        p.partial_fit(Xtr[i], int(ytr[i]))
    print(f"\n[single-pass lr=1e-2 l2=1e-3] TEST ACC: {_eval(p,Xte,yte)*100:.2f}%", flush=True)

    # 2) multi-epoch batch training at a few lr settings -> achievable ceiling
    for lr, l2, ep in [(5e-2, 1e-4, 30), (1e-1, 1e-4, 50), (5e-2, 1e-5, 50)]:
        p = OnlineSoftmaxProbe(n_features, lr=lr, l2=l2)
        p.set_normalizer(Xtr[:256])
        p.train_epochs(Xtr, ytr, epochs=ep, batch_size=128, seed=0)
        acc = _eval(p, Xte, yte)
        print(f"[{ep}-epoch lr={lr} l2={l2}] TEST ACC: {acc*100:.2f}%  "
              f"{'PASS>=94%' if acc>=0.94 else ''}", flush=True)

    # 3) closed-form ridge readout (RLS-equivalent least squares) = linear ceiling
    _ridge_ceiling(Xtr, ytr, Xte, yte)


def _ridge_ceiling(Xtr, ytr, Xte, yte, lam=10.0):
    # one-hot ridge regression = linear readout ceiling. d=14114 >> n, so use the
    # DUAL form: W = A^T (A A^T + lam I)^-1 Y, an (n x n) solve instead of (d x d).
    mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-6] = 1.0
    A = ((Xtr - mean) / std).astype(np.float64)
    B = ((Xte - mean) / std).astype(np.float64)
    Y = np.eye(10, dtype=np.float64)[ytr]
    n = A.shape[0]
    G = A @ A.T                                  # (n, n) Gram
    alpha = np.linalg.solve(G + lam * np.eye(n), Y)   # (n, 10)
    # test predictions: B @ A^T @ alpha  == (B @ A.T) @ alpha
    pred = (B @ A.T) @ alpha
    acc = float((np.argmax(pred, 1) == yte).mean())
    print(f"[ridge dual lam={lam}] LINEAR CEILING TEST ACC: {acc*100:.2f}%  "
          f"{'PASS>=94%' if acc>=0.94 else ''}", flush=True)


if __name__ == "__main__":
    main()
