import numpy as np
from data import *
W,Y,F,R,P=[],[],[],[],[]
for r in DS1+DS2:
    w,y,f,p=beats(r); W.append(w); Y.append(y); F.append(f); R.append(np.full(len(y),r)); P.append(p)
np.savez_compressed('../data/beats.npz', W=np.concatenate(W), Y=np.concatenate(Y), F=np.concatenate(F), R=np.concatenate(R), P=np.concatenate(P))
print('saved', sum(len(y) for y in Y))
