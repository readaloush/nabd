"""Classic baseline: LightGBM on RR features + downsampled beat and template-difference (DS1 patient-grouped CV)."""
import numpy as np, json, lightgbm as lgb
from data import DS1
import os
D = np.load('../data/beats2.npz'); W, Y, F, R = D['W'], D['Y'], D['F'], D['R']
from train import metrics
X = np.concatenate([np.log(np.maximum(F, 1e-3)), W[:, 0, ::8], W[:, 1, ::8],
                    np.abs(W[:, 1]).mean(1, keepdims=True), W[:, 0].max(1, keepdims=True) - W[:, 0].min(1, keepdims=True)], 1)
folds = [[101,109,116,203,209,223],[106,112,118,205,215,230],[108,114,119,207,220],[115,122,124,201,208]]
oof = np.zeros(len(Y), int); oofp = np.zeros((len(Y), 5), np.float32); mask = np.zeros(len(Y), bool)
for k, f in enumerate(folds):
    va = np.where(np.isin(R, f))[0]; tr = np.where(np.isin(R, DS1) & ~np.isin(R, f))[0]
    cnt = np.bincount(Y[tr], minlength=5) + 1; w = ((cnt.sum() / cnt) ** 0.5)[Y[tr]]
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=1)
    m.fit(X[tr], Y[tr], sample_weight=w); pp = np.zeros((len(va), 5)); pp[:, m.classes_] = m.predict_proba(X[va]); oofp[va] = pp; oof[va] = pp.argmax(1); mask[va] = True
o = metrics(Y[mask], oof[mask])
print('gbm OOF acc', round(o['acc'], 4), 'macroF1', round(o['macro_f1'], 3), {c: (round(o[c]['se'], 3), round(o[c]['ppv'], 3)) for c in 'NSVF'})
np.save('../results/oof_gbm.npy', oofp)
json.dump(o, open('../results/cv_gbm.json', 'w'), indent=1)
