import sys
import numpy as np
import wfdb
import matplotlib.pyplot as plt

record = sys.argv[1] if len(sys.argv) > 1 else "100"

# CHANGE 1: use the record variable (note the f before the quote)
rec = wfdb.rdrecord(f"data/raw/mitdb/{record}")
ann = wfdb.rdann(f"data/raw/mitdb/{record}", "atr")

# CHANGE 2: self-check, so a wrong record can't slip through silently
print("loaded:", rec.record_name, ann.record_name)
assert rec.record_name == record, "loaded the wrong record"

fs = rec.fs
n = 10 * fs                      # first 10 seconds
sig = rec.p_signal[:n, 0]        # channel 0
t = [i / fs for i in range(n)]

# keep only real beats (drop rhythm marker "+"), inside the window
beat_idx = [i for i, sym in enumerate(ann.symbol)
            if sym != "+" and ann.sample[i] < n]
beat_samples = np.array([ann.sample[i] for i in beat_idx])
syms = [ann.symbol[i] for i in beat_idx]
beat_t = beat_samples / fs

rr = np.diff(beat_t)
diffs = np.diff(rr)
rmssd_all = np.sqrt(np.mean(diffs ** 2)) * 1000      # ms

nn_ok = [a == "N" and b == "N" for a, b in zip(syms[:-1], syms[1:])]
valid = np.array([nn_ok[i] and nn_ok[i + 1] for i in range(len(diffs))])

# CHANGE 3: handle "no clean differences" instead of printing nan
if valid.sum() == 0:
    rmssd_nn = None
    print("RMSSD (NN only): NOT EVALUATED, no clean NN differences in window")
else:
    rmssd_nn = np.sqrt(np.mean(diffs[valid] ** 2)) * 1000
    print(f"RMSSD (NN only): {rmssd_nn:.2f} ms from {valid.sum()} differences")

print(f"RMSSD (all beats): {rmssd_all:.2f} ms")
print("symbols in window:", set(syms))
print("nn_ok:", nn_ok)
print("valid:", valid)
print("rr (s):", rr)
print("bpm per interval:", 60 / rr)

# plot last, so the numbers print before the window blocks the script
plt.plot(t, sig)
plt.plot(beat_t, [sig[s] for s in beat_samples], "ro")
plt.xlabel("seconds")
plt.ylabel(f"{rec.sig_name[0]} ({rec.units[0]})")
plt.title(f"MIT-BIH record {record}, first 10 s")
plt.show()
