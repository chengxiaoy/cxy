import pandas as pd
from fiw_dataset import *
from tets import *
import torch

test_path = "../input/test/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


from tqdm import tqdm

submission = pd.read_csv('Faces_in_the_Wild/sample_submission.csv')

predictions = []
model = SiameseNetwork(False).to(device).eval()

model.load_state_dict(torch.load('SiameseNetwork.pth'))

for batch in tqdm(chunker(submission.img_pair.values)):
    X1 = [x.split("-")[0] for x in batch]
    X1 = [loader(test_path + x, 'extract') for x in X1]

    X2 = [x.split("-")[1] for x in batch]
    X2 = [loader(test_path + x, 'extract') for x in X2]

    res = model(torch.Tensor(X1).to(device), torch.Tensor(X2).to(device), [False, 0]).data.cpu().numpy()
    res = np.squeeze(res)

    predictions += res>0.5

submission['is_related'] = predictions

submission.to_csv("vgg_face.csv", index=False)
