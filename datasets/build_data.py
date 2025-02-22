import os

import numpy as np

from tqdm import tqdm
import pickle

pj = os.path.join

# TODO SELECT THE DATASET
# ------------------------ FOR KIT 2022 (21 JOINTS) -----------------------------
dataset_name = "kit"

# ---------------------- FOR HumanML3D (22 JOINTS) -------------------------------
# dataset_name =  "h3D"

if dataset_name=="kit":
    folder_name="KIT-ML"
    save_to_path = r"/media/hdd/usr/tao/momask-codes/dataset/KIT-ML/kit_with_splits_2023.npz"
else:
    folder_name="HumanML3D"
    save_to_path = r"/media/hdd/usr/tao/momask-codes/dataset/HumanML3D/all_humanML3D.npz"


abs_path = r"/media/hdd/usr/tao/momask-codes/dataset"
splits = abs_path + f"/{folder_name}/"
poses_dir = splits+ "new_joints"
texts = splits + "texts"


ids = []
list_desc = []
kit_poses = []
nameid = []
with open(splits+"train.txt") as f:
    train_ids = f.read().split("\n")
with open(splits+"test.txt") as f:
    test_ids = f.read().split("\n")
with open(splits+"val.txt") as f:
    val_ids = f.read().split("\n")

split_ids = []
for ndesc in tqdm(os.listdir(texts)):
    npose = ndesc.replace("txt","npy")
    try:
        kit_poses.append( np.load(pj(poses_dir,npose)))
    except FileNotFoundError:
        print("Pose without description Id: ",npose)
        continue
    with open(pj(texts,ndesc),encoding='utf-8') as f:
        list_desc +=[[ph.split("#")[0] for ph in f.readlines()]]
        ids += [npose.split(".")[0]]
        nameid.append(npose.split(".")[0])
        #print(nameid[-1])
        if nameid[-1] in train_ids:
            split_ids.append("train")
        elif nameid[-1] in test_ids:
            split_ids.append("test")
        elif nameid[-1]  in val_ids:
            split_ids.append("val")
        else:
            print("Unclassified sample ID: ",nameid)
            break
        

    assert npose.split(".")[0]== ndesc.split(".")[0]

if dataset_name=='kit':
    N_samples = 6016
    assert len(kit_poses)==N_samples
    assert len(list_desc)==len(kit_poses)
    
with open(save_to_path, 'wb') as f:
    pickle.dump({
        'kitmld_array': kit_poses,
        'old_desc': list_desc,
        'sample_ids': nameid,
        'splits_ids': split_ids
    }, f)


# ids_train = np.where(np.asarray(split_ids)=="train")[0]
# ids_train_val = np.where(np.asarray(split_ids)=="val")[0]
# ids_test= np.where(np.asarray(split_ids)=="test")[0]






