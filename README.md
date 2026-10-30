# Extracellular Spike-Sorting Pipeline (simulated ground truth)

End-to-end spike sorting: band-pass 300–3000 Hz → robust MAD threshold (4.5σ) → peak alignment → waveform
snippets → PCA (3 comps) → Gaussian-mixture clustering with BIC model selection → evaluation against known
spike times (Hungarian unit matching, precision/recall/F1).

Data are **simulated** (30 kHz, 60 s, three units with distinct waveforms, Poisson firing with refractory
period, amplitude jitter, band-limited coloured noise), because ground truth is needed for scoring.

```
pip install numpy scipy scikit-learn matplotlib
python spike_sorting.py     # writes results.json and sorting_result.png
```

## Results (3 seeds per noise level, `results.json`)
| Noise (µV rms) | Mean F1 over units |
|---|---|
| 6 | 0.96 |
| 10 | 0.92 |
| 15 | 0.75 |

At low noise BIC over-splits (4–5 clusters found for 3 units; matching scores the best cluster per unit).
At 15 µV the weakest units fall under the detection threshold and are missed. Not validated on real recordings.
