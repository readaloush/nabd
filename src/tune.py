import numpy as np, json, itertools
from train import metrics
D = np.load('../data/beats2.npz'); Y, R = D['Y'], D['R']
from data import DS1
mask = np.isin(R, DS1); y = Y[mask]
pc = np.load('../results/oof_v2_probs.npy')[mask]; pg = np.load('../results/oof_gbm.npy')[mask]
def score(p): return metrics(y, p.argmax(1))
res = {}
for a in [0, 0.25, 0.5, 0.75, 1.0]:
    p = a*pc + (1-a)*pg; o = score(p); res[a] = o['macro_f1']; print('cnn weight', a, 'macroF1', round(o['macro_f1'], 3), 'S', round(o['S']['se'],3), round(o['S']['ppv'],3))
a = max(res, key=res.get); p = a*pc + (1-a)*pg
grid = [0.5, 0.7, 1, 1.4, 2, 3]; best = (-1, None)
for s, v, f in itertools.product(grid, grid, grid):
    q = p * np.array([1, s, v, f, 1]); mf = score(q)['macro_f1']
    if mf > best[0]: best = (mf, (s, v, f))
q = p * np.array([1, *best[1], 1]); o = score(q)
print('chosen cnn weight', a, 'scales S,V,F', best[1], 'macroF1', round(o['macro_f1'], 3), {c: (round(o[c]['se'], 3), round(o[c]['ppv'], 3)) for c in 'NSVF'})
json.dump({'cnn_weight': a, 'scales': [1, *best[1], 1], 'oof': o}, open('../results/tuned.json', 'w'), indent=1)
