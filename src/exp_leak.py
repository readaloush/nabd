import numpy as np, json, sys
from train import *
from data import DS1, DS2
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(seed); allidx = np.arange(len(Y))
# A) intra-patient: beats from every patient shuffled into train / test
perm = rng.permutation(allidx); nte = len(perm)//5; te_a = perm[:nte]; rest = perm[nte:]
va_a = rest[:len(rest)//10]; tr_a = rest[len(rest)//10:]
# B) inter-patient: DS1 patients train, DS2 patients test, 4 DS1 patients for validation
VAL = [209, 119, 220, 115]
tr_b = np.where(np.isin(R, DS1) & ~np.isin(R, VAL))[0]; va_b = np.where(np.isin(R, VAL))[0]; te_b = np.where(np.isin(R, DS2))[0]
out = {}
for name, (tr, va, te) in {'intra_patient': (tr_a, va_a, te_a), 'inter_patient': (tr_b, va_b, te_b)}.items():
    print(name, len(tr), len(va), len(te), flush=True)
    m = train(tr, va, seed=seed, log=True)
    out[name] = metrics(Y[te], predict(m, te).argmax(1))
    o = out[name]; print(name, 'acc', round(o['acc'], 4), {c: (round(o[c]['se'], 3), round(o[c]['ppv'], 3)) for c in 'NSVF'}, flush=True)
json.dump(out, open(f'../results/leak_seed{seed}.json', 'w'), indent=1)
