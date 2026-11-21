import torch, torch.nn as nn, numpy as np, onnx, onnxruntime as ort
from model import BeatNet
class Ens(nn.Module):
    def __init__(s, ms): super().__init__(); s.ms = nn.ModuleList(ms)
    def forward(s, w, f):
        f = torch.clamp(torch.log(torch.clamp(f, min=1e-3)), -3, 3)
        return torch.stack([m(w, f).softmax(1) for m in s.ms]).mean(0)
ms = []
for k in range(3):
    m = BeatNet(in_ch=2, n_f=6); m.load_state_dict(torch.load(f'../results/nabd_seed{k}.pt')); m.eval(); ms.append(m)
e = Ens(ms).eval()
w = torch.randn(4, 2, 256); f = torch.rand(4, 6) + 0.5
torch.onnx.export(e, (w, f), '../docs/nabd.onnx', input_names=['beat', 'rr'], output_names=['probs'],
                  dynamic_axes={'beat': {0: 'n'}, 'rr': {0: 'n'}, 'probs': {0: 'n'}}, opset_version=17, dynamo=False)
# check against torch on real DS2 beats
D = np.load('../data/beats2.npz'); from data import DS2
te = np.where(np.isin(D['R'], DS2))[0]
s = ort.InferenceSession('../docs/nabd.onnx')
po = np.concatenate([s.run(None, {'beat': D['W'][te[i:i+4096]], 'rr': D['F'][te[i:i+4096]]})[0] for i in range(0, len(te), 4096)])
pt = np.load('../results/ds2_probs.npy')
import os; print('onnx KB', os.path.getsize('../docs/nabd.onnx')//1024, 'max abs diff', float(np.abs(po-pt).max()), 'same argmax', float((po.argmax(1)==pt.argmax(1)).mean()))
