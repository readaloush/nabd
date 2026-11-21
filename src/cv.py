"""Patient-grouped 4-fold CV on DS1 only. Usage: NABD_DATA=... python cv.py NAME [wpow] [epochs]"""
import numpy as np, json, sys
from train import *
from data import DS1
name = sys.argv[1]; wpow = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5; ep = int(sys.argv[3]) if len(sys.argv) > 3 else 8
folds = [[101,109,116,203,209,223],[106,112,118,205,215,230],[108,114,119,207,220],[115,122,124,201,208]]
assert sorted(sum(folds, [])) == sorted(DS1)
oof_p = np.zeros(len(Y), int); oof_prob = np.zeros((len(Y), 5), np.float32); mask = np.zeros(len(Y), bool)
for k, f in enumerate(folds):
    va = np.where(np.isin(R, f))[0]; tr = np.where(np.isin(R, DS1) & ~np.isin(R, f))[0]
    m = train(tr, None, seed=k + int(__import__("os").environ.get("SEED_OFF", 0)), epochs=ep, wpow=wpow)
    pr = predict(m, va); oof_prob[va] = pr; oof_p[va] = pr.argmax(1); mask[va] = True
    print(name, 'fold', k, 'done', flush=True)
o = metrics(Y[mask], oof_p[mask])
print(name, 'OOF acc', round(o['acc'], 4), 'macroF1', round(o['macro_f1'], 3), {c: (round(o[c]['se'], 3), round(o[c]['ppv'], 3)) for c in 'NSVF'}, flush=True)
np.save(f'../results/oof_{name}.npy', oof_prob)
json.dump(o, open(f'../results/cv_{name}.json', 'w'), indent=1)
