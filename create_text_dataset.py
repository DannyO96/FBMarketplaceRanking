import torch
import pandas as pd

class TextDataset(torch.utils.data.Dataset):
    """
    This class prepares the text dataset it inherits from torch.utils.dataset 
    """
    def __init__(self):
        self.prods_imgs = pd.read_csv('/home/danny/git/FBMarketplaceRanking/my.secrets.data/prods_imgs.csv',  lineterminator='\n')
        self.labels = self.prods_imgs['category'].to_list()
        self.descriptions = self.prods_imgs['product_description'].to_list()
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

    def __getitem__(self, index):
        """
        Overwrites the built in python __getitem__ function and defines what is done to the dataset when an index is referenced.
        """
        label = self.labels[index]
        label = self.encoder[label]
        #label = torch.as_tensor(label)
        description = self.descriptions[index]
        return description, label

    def __len__(self):
        """
        Overwrites the built in python __len__ function and defines what is done to the dataset when the len funtion is called.
        """
        return len(self.labels)

dataset = TextDataset()
print(len(dataset))
print(dataset.num_classes)
print(dataset[1000])