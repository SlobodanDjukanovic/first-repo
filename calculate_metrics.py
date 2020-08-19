import h5py
import os, numpy as np, plot as plot
import feature_extraction as fe
from feature_extraction import time_dist_threshold as tdt
from importlib import reload
reload(fe)

folder = "Probs/"
all_files = os.listdir(folder)
prob_files = [file for file in all_files if file.endswith(".h5") and not file.endswith("unseen.h5")]
prob_files = [file for file in all_files if file.endswith("unseen.h5")]
# prob_files = ['Probs_feat#hfp#lms.h5']

for file in prob_files:
    hf = h5py.File(folder + file, 'r')
    correct = np.array(hf['correct'], dtype=np.float32).flatten()
    false_pos = np.array(hf['false_pos'], dtype=np.float32).flatten()
    false_neg = np.array(hf['false_neg'], dtype=np.float32).flatten()

    NAUC = np.mean(correct[:-1])/100
    ind_min = np.argmin(np.abs(false_pos - false_neg)).flatten()[0]

    ind1, ind2 = file.find('#'), file.find('.h5')
    feat_str = file[ind1+1: ind2]
    feat_str = feat_str.replace("#", "+").upper()
    print("& {} & ${:.3f}$ & ${:.2f}$ (${:.2f}$) \\\\"\
          .format(feat_str, NAUC, false_pos[ind_min], abs(false_pos[ind_min]-false_neg[ind_min])))

    hf.close()
