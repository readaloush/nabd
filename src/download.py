"""Download the MIT-BIH Arrhythmia Database from PhysioNet into data/mitdb."""
import os, wfdb
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data', 'mitdb'), exist_ok=True)
wfdb.dl_database('mitdb', os.path.join(os.path.dirname(__file__), '..', 'data', 'mitdb'))
