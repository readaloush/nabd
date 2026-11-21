import numpy as np, torch, torch.nn as nn, json, time
from sklearn.metrics import confusion_matrix
from model import BeatNet
torch.set_num_threads(2)
import os
D = np.load(os.environ.get('NABD_DATA', '../data/beats.npz')); W, Y, F, R = D['W'], D['Y'], D['F'], D['R']
CL = ['N','S','V','F','Q']

def feats(Fx):  # log-scale RR features, clipped
    return np.clip(np.log(np.maximum(Fx, 1e-3)), -3, 3).astype(np.float32)

def metrics(y, p):
    cm = confusion_matrix(y, p, labels=range(5)); out = {'acc': float((y == p).mean()), 'cm': cm.tolist()}
    for k in range(4):
        tp = cm[k, k]; se = tp / max(cm[k].sum(), 1); pp = tp / max(cm[:, k].sum(), 1)
        out[CL[k]] = {'se': float(se), 'ppv': float(pp), 'f1': float(2*se*pp/max(se+pp, 1e-9)), 'n': int(cm[k].sum())}
    out['macro_f1'] = float(np.mean([out[c]['f1'] for c in 'NSVF'])); return out

def predict(m, idx, bs=2048):
    m.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i+bs]; P.append(m(torch.from_numpy(W[j]), torch.from_numpy(feats(F[j]))).softmax(1).numpy())
    return np.concatenate(P)

def train(tr, va, seed=0, epochs=12, use_rr=True, bs=256, lr=2e-3, log=None, wpow=0.5, chans=None, nf=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = BeatNet(use_rr=use_rr, in_ch=(W.shape[1] if W.ndim == 3 else 1), n_f=F.shape[1]); cnt = np.bincount(Y[tr], minlength=5) + 1
    wts = torch.tensor((cnt.sum() / cnt) ** wpow, dtype=torch.float32); wts /= wts.mean()
    lossf = nn.CrossEntropyLoss(weight=wts); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-3)
    steps = epochs * (len(tr) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    best, best_state = -1, None
    for ep in range(epochs):
        m.train(); perm = rng.permutation(tr); t0 = time.time()
        for i in range(0, len(perm) - bs + 1, bs):
            j = perm[i:i+bs]; w = torch.from_numpy(W[j])
            w = w * (1 + 0.1*torch.randn(len(j), *([1]*(w.dim()-1)))) + 0.05*torch.randn_like(w)   # amplitude + noise augmentation
            loss = lossf(m(w, torch.from_numpy(feats(F[j]))), torch.from_numpy(Y[j]))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if va is None: continue
        mv = metrics(Y[va], predict(m, va).argmax(1))
        if log: print(f'  ep{ep} {time.time()-t0:.0f}s val acc {mv["acc"]:.4f} macroF1 {mv["macro_f1"]:.3f}', flush=True)
        if mv['macro_f1'] > best: best, best_state = mv['macro_f1'], {k: v.clone() for k, v in m.state_dict().items()}
    if best_state is not None: m.load_state_dict(best_state)
    return m
