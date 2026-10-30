"""Extracellular spike-sorting pipeline benchmarked on simulated recordings with ground truth.

Pipeline: band-pass (300-3000 Hz) -> robust MAD threshold -> peak alignment ->
waveform snippets -> PCA -> Gaussian-mixture clustering (BIC model selection) ->
evaluation against known spike times (Hungarian matching, precision/recall/F1).
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

FS = 30000
PRE, POST = 12, 28           # samples around the trough

def template(width, amp, rebound):
    t = np.arange(-PRE, POST)
    return -amp * np.exp(-0.5 * (t / width) ** 2) + rebound * np.exp(-0.5 * ((t - 3 * width) / (2 * width)) ** 2)

TEMPLATES = [template(2.0, 140, 25), template(3.4, 100, 12), template(1.5, 90, 35)]

def simulate(duration_s=60, noise_uv=10, rates=(8, 6, 10), seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    # coloured background noise (sum of many distant units ~ band-limited gaussian)
    x = rng.standard_normal(n); sos = butter(2, [200, 4000], "bandpass", fs=FS, output="sos")
    x = sosfiltfilt(sos, x); x *= noise_uv / x.std()
    truth = []
    for k, (tpl, r) in enumerate(zip(TEMPLATES, rates)):
        times = np.cumsum(rng.exponential(1 / r, int(r * duration_s * 1.5)))
        times = times[times < duration_s - 0.01]
        idx = (times * FS).astype(int)
        keep = np.r_[True, np.diff(idx) > int(0.003 * FS)]      # 3 ms refractory
        idx = idx[keep]; truth.append(idx)
        for i in idx:
            a = 1 + 0.1 * rng.standard_normal()                  # amplitude jitter
            if i - PRE >= 0 and i + POST < n: x[i - PRE:i + POST] += a * tpl
    return x, truth

def detect(x, k=4.5):
    sos = butter(3, [300, 3000], "bandpass", fs=FS, output="sos")
    y = sosfiltfilt(sos, x)
    sigma = np.median(np.abs(y)) / 0.6745
    below = np.flatnonzero(y < -k * sigma)
    if len(below) == 0: return y, np.array([], int)
    groups = np.split(below, np.flatnonzero(np.diff(below) > 20) + 1)
    peaks = np.array([g[np.argmin(y[g])] for g in groups])
    peaks = peaks[(peaks > PRE) & (peaks < len(y) - POST)]
    return y, peaks

def sort_spikes(y, peaks, max_k=6, seed=0):
    wf = np.stack([y[p - PRE:p + POST] for p in peaks])
    feats = PCA(n_components=3, random_state=seed).fit_transform(wf)
    best, best_bic = None, np.inf
    for k in range(1, max_k + 1):
        g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed).fit(feats)
        b = g.bic(feats)
        if b < best_bic: best, best_bic = g, b
    return best.predict(feats), feats, wf

def evaluate(truth, peaks, labels, tol=int(0.0005 * FS)):
    clusters = np.unique(labels)
    nT = len(truth); f1 = np.zeros((nT, len(clusters))); stats = {}
    for i, tt in enumerate(truth):
        for j, c in enumerate(clusters):
            pk = peaks[labels == c]
            if len(pk) == 0: continue
            d = np.abs(tt[:, None] - pk[None, :]).min(1) <= tol * 2
            tp = d.sum(); prec = tp / len(pk); rec = tp / len(tt)
            f1[i, j] = 0 if tp == 0 else 2 * prec * rec / (prec + rec)
            stats[(i, j)] = (prec, rec)
    r, c = linear_sum_assignment(-f1)
    return {f"unit{i}": dict(f1=float(f1[i, j]), precision=float(stats.get((i, j), (0, 0))[0]),
                             recall=float(stats.get((i, j), (0, 0))[1])) for i, j in zip(r, c)}, len(clusters)

if __name__ == "__main__":
    out = {}
    for noise in (6, 10, 15):
        f1s, ks = [], []
        for seed in range(3):
            x, truth = simulate(noise_uv=noise, seed=seed)
            y, peaks = detect(x)
            labels, feats, wf = sort_spikes(y, peaks)
            ev, k = evaluate(truth, peaks, labels)
            f1s.append(np.mean([v["f1"] for v in ev.values()])); ks.append(k)
            if noise == 10 and seed == 0:
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                for c in np.unique(labels):
                    ax[0].scatter(feats[labels == c, 0], feats[labels == c, 1], s=3, label=f"cluster {c}")
                    ax[1].plot(wf[labels == c][:100].T, lw=0.3, color=f"C{c}")
                ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend(markerscale=4)
                ax[1].set_xlabel("sample"); ax[1].set_ylabel("uV")
                plt.tight_layout(); plt.savefig("sorting_result.png", dpi=150)
        out[f"noise_{noise}uV"] = dict(mean_f1=float(np.mean(f1s)), clusters_found=ks)
        print(noise, out[f"noise_{noise}uV"])
    json.dump(out, open("results.json", "w"), indent=2)
