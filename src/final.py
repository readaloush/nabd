"""Final protocol. inter: train on all of DS1, evaluate DS2 once. intra: random beat split over all 44 records. 3 seeds each, 8 epochs."""
import numpy as np, json, torch, sys
from train import *
from data import DS1, DS2
mode = sys.argv[1]; out = {}; probs = []
te = np.where(np.isin(R, DS2))[0] if mode == 'inter' else None
for seed in [0, 1, 2]:
    if mode == 'inter':
        tr = np.where(np.isin(R, DS1))[0]
    else:
        rng = np.random.default_rng(seed); perm = rng.permutation(len(Y)); te = perm[:len(perm)//5]; tr = perm[len(perm)//5:]
    m = train(tr, None, seed=seed, epochs=8)
    p = predict(m, te); out[f'seed{seed}'] = metrics(Y[te], p.argmax(1))
    o = out[f'seed{seed}']; print(mode, seed, 'acc', round(o['acc'], 4), 'macroF1', round(o['macro_f1'], 3), {c: (round(o[c]['se'], 3), round(o[c]['ppv'], 3)) for c in 'NSVF'}, flush=True)
    if mode == 'inter':
        probs.append(p); torch.save(m.state_dict(), f'../results/nabd_seed{seed}.pt')
if mode == 'inter':
    pe = np.mean(probs, 0); out['ensemble'] = metrics(Y[te], pe.argmax(1)); np.save('../results/ds2_probs.npy', pe)
    o = out['ensemble']; print('ensemble acc', round(o['acc'], 4), 'macroF1', round(o['macro_f1'], 3), {c: (round(o[c]['se'], 3), round(o[c]['ppv'], 3)) for c in 'NSVF'}, flush=True)
json.dump(out, open(f'../results/final_{mode}.json', 'w'), indent=1)
