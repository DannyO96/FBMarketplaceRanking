import torch
import pandas as pd

class TextDataset(torch.utils.data.Dataset):
    """"
    """
    def __init__(self):
        self.prods_imgs = pd.read_csv('/home/danny/git/FBMarketplaceRanking/my.secrets.data/prods_imgs.csv',  lineterminator='\n')
        self.labels = self.prods_imgs['category'].to_list()
        self.descriptions = self.prods_imgs['product_description'].to_list()
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

    def __getitem__(self, index):
        label = self.labels[index]
        label = self.encoder[label]
        #label = torch.as_tensor(label)
        description = self.descriptions[index]
        return description, label

    def __len__(self):
        return len(self.labels)

dataset = TextDataset()
print(len(dataset))
print(dataset.num_classes)
print(dataset[1000])