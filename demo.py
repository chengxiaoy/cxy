from scipy import io
import numpy as np
import pandas as pd

fd_path = "KinFaceW-II/meta_data/fd_pairs.mat"
fs_path = 'KinFaceW-II/meta_data/fs_pairs.mat'
md_path = 'KinFaceW-II/meta_data/md_pairs.mat'
ms_path = 'KinFaceW-II/meta_data/ms_pairs.mat'

paths = [fd_path, fs_path, md_path, ms_path]

res = []
for path in paths:
    pairs = io.loadmat(path)['pairs']
    index = pairs[:, 1] == np.array([[1]], dtype=np.uint8)
    true_pairs = pairs[index[0]][:, 2:]

    for p1, p2 in true_pairs:
        res.append([p1[0], p2[0]])

res = np.array(res)

df = pd.DataFrame({'p1': res[:, 0], 'p2': res[:, 1]})
df.to_csv("KinFaceW-II/kfacew_2.csv", index=False)
