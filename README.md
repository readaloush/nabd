# Nabd نبض

Heartbeat classification on the MIT-BIH Arrhythmia Database, evaluated on patients the model has never seen, and running in the browser.

**Demo:** [readaloush.github.io/nabd](https://readaloush.github.io/nabd/): pick a test patient and every beat in a 60-second ECG strip is labelled in your browser, next to the cardiologists' label.

Research and education only. Not a medical device.

## The finding

Heartbeat classifiers on MIT-BIH routinely report 98–99% accuracy. Most of them split the beats at random, so the test set is full of beats from patients the model was trained on. The same model, trained the same way, was evaluated both ways (3 seeds, mean ± std):

| | Beats shuffled (intra-patient) | New patients (inter-patient) |
|---|---|---|
| Accuracy | 98.8 ± 0.1 | 90.8 ± 1.1 |
| N sensitivity / PPV | 99.1 / 99.8 | 94.4 / 96.3 |
| S sensitivity / PPV | 94.7 / 86.8 | **20.1 / 19.1** |
| V sensitivity / PPV | 98.1 / 96.7 | 93.0 / 81.0 |
| F sensitivity / PPV | 91.3 / 69.3 | **1.9 / 1.8** |

Intra-patient: all 44 non-paced records, 80/20 random beat split. Inter-patient: trained on DS1, tested on DS2 (the 22/22 patient split of de Chazal et al., 2004). Percent, AAMI classes N, S (supraventricular ectopic), V (ventricular ectopic), F (fusion).

Ventricular beats transfer to new patients: 93.0% sensitivity at 81.0% PPV, against 77.7% / 81.9% for de Chazal et al. (2004) on the same split. Supraventricular and fusion beats do not.

## Why S fails: one patient

Three quarters of the S beats in DS2 (1,381 of 1,836) come from record 232, in which most beats are atrial premature beats. The model compares each beat with the median of the same patient's previous 20 beats and gives RR intervals relative to the patient's own average. Both assume that a patient's typical beat is normal. In record 232 the typical beat is the abnormal one, and the model finds 3.8% of its S beats. On the other 21 test patients it finds 52.7% (at 14.3% PPV).

This was found after the single evaluation on DS2, so the model was **not** changed to fit record 232. Adjusting it now would be tuning on the test set, which is what this project measures. Record 232 is one of the demo strips.

F beats are nearly hopeless across patients: 372 of the 414 F beats in DS1 come from record 208, so the model has seen almost no other patient's fusion beats.

## Method

**Beats.** Lead MLII, 0.5–40 Hz band-pass (3rd-order Butterworth, zero-phase). Beats at the annotated R peaks, 0.25 s before to 0.46 s after (256 samples at 360 Hz), mapped to AAMI classes. Paced records 102, 104, 107 and 217 excluded. DS1: 50,977 beats, DS2: 49,668 (`src/data.py`, `src/cache2.py`).

**Input (no labels used).** Two channels: the beat, and the beat minus the running median of the patient's previous 20 beats, both scaled by that median beat's spread. Six RR features: previous and next RR, previous / mean of the last 10, next / previous, previous / mean of the last 200, last-10 mean / last-200 mean.

**Model.** 1-D CNN (three blocks of two convolutions, 16–32–64 channels), global average pooling, RR features joined before a 64-unit layer; 50,565 parameters. Class-weighted cross-entropy (inverse square-root frequency), AdamW, one-cycle schedule, amplitude and noise augmentation, 8 epochs on CPU (`src/model.py`, `src/train.py`).

**Selection on DS1 only.** Every design choice was compared by 4-fold cross-validation grouped by patient within DS1. Macro F1 over N, S, V, F, out-of-fold:

| Variant (DS1 cross-validation) | Macro F1 | S sens. / PPV |
|---|---|---|
| Beat only + 4 RR features | 0.420 | 40.8 / 16.5 |
| **+ running-median channel, 6 RR features (chosen)** | **0.524** (0.514 with other seeds) | 47.4 / 32.3 |
| same, linear class weights | 0.493 | 60.7 / 14.9 |
| same, 2-second context window | 0.484 | 41.5 / 18.0 |
| LightGBM on the same features | 0.510 | 33.5 / 38.7 |

Averaging the CNN with LightGBM did not help, and per-class decision weights tuned on the out-of-fold predictions gained only 0.01, so neither was used (`src/cv.py`, `src/gbm_cv.py`, `src/tune.py`).

**Final.** Three seeds trained on all of DS1, DS2 scored once (`src/final.py`). The three are averaged into one ONNX model (`docs/nabd.onnx`, 600 KB). The ensemble gives 91.6% accuracy on DS2, S 16.0 / 17.0, V 92.9 / 82.2.

**First baseline** (beat only, 4 RR features, `src/exp_leak.py`): 98.7% intra-patient against 83.1% inter-patient.

## Demo

`docs/index.html` runs the ONNX model with ONNX Runtime Web. For five DS2 patients it ships a band-pass filtered 90-second signal (30 s of history plus the 60 s shown), the R-peak positions from the database annotations and the RR intervals before the strip. The two-channel input and the RR features are computed in the browser. On all 418 demo beats the browser's labels match the Python pipeline, with probabilities within 0.5 percentage points (from storing the signal as 16-bit integers).

A real device would also need its own R-peak detector; here the annotated positions are used so that the classifier is judged on its own.

## Run it

```
pip install -r requirements.txt
python src/download.py        # MIT-BIH from PhysioNet into data/mitdb
cd src
python cache.py && python cache2.py
python exp_leak.py 0                                  # first baseline, both splits
NABD_DATA=../data/beats2.npz python cv.py v2 0.5     # DS1 cross-validation
NABD_DATA=../data/beats2.npz python final.py inter   # DS1 -> DS2, 3 seeds
NABD_DATA=../data/beats2.npz python final.py intra
python export.py && python demo_data.py
```

Results of the runs above are in `results/`.

## Limitations

- One lead (MLII). de Chazal et al. used two.
- R peaks from the annotations, not detected.
- 22 test patients: per-class numbers for S and F depend on a handful of records.
- Single database. MIT-BIH recordings are from the late 1970s.

## Data and references

- G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Eng. Med. Biol. Mag.*, 20(3), 2001. Data from PhysioNet, Open Data Commons Attribution License v1.0. The demo includes short excerpts of five records; the full database is not redistributed.
- P. de Chazal, M. O'Dwyer and R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," *IEEE Trans. Biomed. Eng.*, 51(7), 1196–1206, 2004.
- ANSI/AAMI EC57, testing and reporting performance results of cardiac rhythm and ST-segment measurement algorithms.

MIT License. Read Alalloş, [readaloush.com](https://readaloush.com).
