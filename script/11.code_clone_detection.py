"""
This script checks whether clone exists in the datasets
"""
import glob
import os
from difflib import SequenceMatcher

from tqdm import tqdm


def check_clones():
    pass


if __name__ == '__main__':
    os.chdir("..")

    ss_files = glob.glob('clone-detection-check-data/GroundTruthModels-SS/addressbook/*.content_tags')
    ds_files = glob.glob('clone-detection-check-data/01booster.com/*.content_tags')

    ds_files = ds_files[:1]

    num_comparisons = len(ss_files) * len(ds_files)
    num_identical = 0
    num_clones = 0

    print("Pages in SS: %d" % len(ss_files))
    print("Pages in DS + CC: %d" % len(ds_files))
    print("Comparisons: %d" % num_comparisons)
    all_ratios = 0.0

    for file_in_ss in tqdm(ss_files):
        for file_in_ds in ds_files:
            ratio = SequenceMatcher(None, open(file_in_ss, "r").read(), open(file_in_ds, "r").read()).ratio()
            all_ratios = all_ratios + ratio
            # print(ratio)
            if ratio == 1.0:
                print("identical")
                num_identical = num_identical + 1
            elif ratio >= 0.2:
                print("potential clone")
                num_clones = num_clones + 1

    print("identical (sim = 1): %d (%.2f %%)" % (num_identical, num_identical / num_comparisons))
    print("clones (sim > 0.2): %d (%.2f %%)" % (num_clones, num_identical / num_comparisons))
    print("avg similarity (0=dissimilar, 1=identical): %.2f" % (all_ratios / num_comparisons))
