import pandas as pd
from fiw_dataset import *
from tets import *
import torch
from torch.utils.data.dataloader import default_collate

test_path = "Faces_in_the_Wild/test/"

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

    res = model(default_collate(X1).to(device), default_collate(X2).to(device), [False, 0])
    res = res.data.cpu().numpy()
    res = np.squeeze(res)
    # pred = res > 0.5
    predictions += res.tolist()

predictions = np.array(predictions).astype(float)
submission['is_related'] = predictions

submission.to_csv("vgg_face.csv", index=False)
