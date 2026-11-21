"""MIT-BIH beat extraction with AAMI classes."""
import numpy as np, wfdb
from scipy.signal import butter, filtfilt
import os
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'mitdb') + os.sep
FS = 360
PACED = {102, 104, 107, 217}
DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
AAMI = {**{s:'N' for s in 'NLRej'}, **{s:'S' for s in 'AaJS'}, **{s:'V' for s in 'VE'}, 'F':'F', **{s:'Q' for s in '/fQ'}}
CLASSES = ['N','S','V','F','Q']
PRE, POST = 90, 166   # 0.25 s before R, 0.46 s after -> 256 samples

def bandpass(x, lo=0.5, hi=40):
    b, a = butter(3, [lo/(FS/2), hi/(FS/2)], 'band'); return filtfilt(b, a, x)

def load_record(r):
    rec = wfdb.rdrecord(DB + str(r)); ann = wfdb.rdann(DB + str(r), 'atr')
    ch = rec.sig_name.index('MLII') if 'MLII' in rec.sig_name else 0
    return bandpass(rec.p_signal[:, ch]), np.array(ann.sample), np.array(ann.symbol)

def beats(r):
    """Return windows (n,256), labels (n,), RR features (n,4), R positions."""
    x, samp, sym = load_record(r)
    keep = np.array([s in AAMI for s in sym]); samp, sym = samp[keep], sym[keep]
    rr = np.diff(samp) / FS
    W, Y, F, P = [], [], [], []
    for i in range(1, len(samp) - 1):
        s = samp[i]
        if s - PRE < 0 or s + POST > len(x): continue
        pre, post = rr[i-1], rr[i]
        local = rr[max(0, i-10):i].mean()          # mean of the previous 10 RR intervals (causal)
        W.append(x[s-PRE:s+POST]); Y.append(CLASSES.index(AAMI[sym[i]]))
        F.append([pre, post, pre/local, post/pre]); P.append(s)
    W = np.array(W, np.float32)
    W = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-6)
    return W, np.array(Y), np.array(F, np.float32), np.array(P)

def beats2(r, n_tpl=20, n_long=200, pre_s=PRE, post_s=POST, step=1):
    """2-channel input: the beat, and the beat minus the patient's running median beat (previous n_tpl beats).
    RR features: pre, post, pre/local, post/pre, pre/long, local/long (all from the patient's own past)."""
    x, samp, sym = load_record(r)
    keep = np.array([s in AAMI for s in sym]); samp, sym = samp[keep], sym[keep]
    rr = np.diff(samp) / FS
    raw, Y, F, P = [], [], [], []
    for i in range(1, len(samp) - 1):
        s = samp[i]
        if s - pre_s < 0 or s + post_s > len(x): continue
        pre, post = rr[i-1], rr[i]
        local = rr[max(0, i-10):i].mean(); long_ = rr[max(0, i-n_long):i].mean()
        raw.append(x[s-pre_s:s+post_s:step]); Y.append(CLASSES.index(AAMI[sym[i]]))
        F.append([pre, post, pre/local, post/pre, pre/long_, local/long_]); P.append(s)
    raw = np.array(raw, np.float32); n = len(raw)
    tpl = np.empty_like(raw)
    for i in range(n):
        lo = max(0, i - n_tpl); tpl[i] = np.median(raw[lo:i], 0) if i > 0 else raw[0]
    sc = tpl.std(1, keepdims=True) + 1e-3
    a = (raw - tpl.mean(1, keepdims=True)) / sc; d = (raw - tpl) / sc
    W = np.stack([a, d], 1).astype(np.float32)
    return W, np.array(Y), np.array(F, np.float32), np.array(P)
