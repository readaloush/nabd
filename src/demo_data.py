"""Pick a 60-second strip from several DS2 (test) patients and save what the browser needs to recompute features itself."""
import numpy as np, json, base64
from data import *
D = np.load('../data/beats2.npz'); Rr, Pp = D['R'], D['P']
probs = np.load('../results/ds2_probs.npy'); te = np.where(np.isin(Rr, DS2))[0]
pmap = {(int(Rr[i]), int(Pp[i])): probs[k] for k, i in enumerate(te)}
PICK = {100: 'Mostly normal rhythm with a few early atrial beats', 200: 'Frequent ventricular beats', 213: 'Fusion beats',
        222: 'Runs of supraventricular beats', 232: 'The hard case: most beats are atrial premature beats'}
out = []
for rec, note in PICK.items():
    x, samp, sym = load_record(rec)
    keep = np.array([s in AAMI for s in sym]); samp, sym = samp[keep], sym[keep]
    lab = np.array([CLASSES.index(AAMI[s]) for s in sym])
    L = 60*FS; best, bs = -1, None
    for st in range(300*FS, len(x) - L - FS, 5*FS):          # leave >= 5 minutes of history before the strip
        m = (samp >= st) & (samp < st + L); score = (lab[m] != 0).sum() + 3*len(set(lab[m]))
        if 0 < score and score > best: best, bs = score, st
    st = bs; ctx = st - 30*FS                                  # 30 s of context for the running median beat
    sig = x[ctx: st + L + FS]
    m = (samp >= ctx + PRE) & (samp < st + L + FS - POST)
    r = (samp[m] - ctx).tolist(); y = lab[m].tolist()
    first = np.where(samp == samp[m][0])[0][0]
    rr_prior = (np.diff(samp[max(0, first-201):first+1]) / FS).tolist()  # RR intervals before the first beat
    ref = [pmap.get((rec, int(s)), np.full(5, np.nan)).round(4).tolist() for s in samp[m]]
    q = np.clip(np.round(sig*1000), -32768, 32767).astype('<i2').tobytes()
    out.append({'rec': rec, 'note': note, 'fs': FS, 'show': 30*FS, 'sig': base64.b64encode(q).decode(), 'r': r, 'y': y,
                'rrPrior': [round(v, 5) for v in rr_prior], 'ref': ref, 'nBefore': int(first)})
    print(rec, 'strip start s', st/FS, 'beats', len(r), 'labels', np.bincount(y, minlength=5))
json.dump(out, open('../docs/data/strips.json', 'w'))
import os; print('KB', os.path.getsize('../docs/data/strips.json')//1024)
