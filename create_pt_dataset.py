import pandas as pd
import PIL
from torch.utils.data import Dataset, DataLoader


class CreateImageDataset(Dataset):
    """
    This class inherits from torch.utils.data.Dataset
    
    """
    
    def __init__(self):
        super().__init__()
        self.image_data = pd.read_csv('/home/danny/git/FBMarketplaceRanking/my.secrets.data/Images.csv')
        self.product_data = pd.read_csv('/home/danny/git/FBMarketplaceRanking/my.secrets.data/CleanProducts.csv', lineterminator='\n')
        self.link_data = pd.read_csv('/home/danny/git/FBMarketplaceRanking/my.secrets.data/Links.csv')
        self.prods_imgs = pd.read_csv('/home/danny/git/FBMarketplaceRanking/my.secrets.data/prods_imgs.csv',  lineterminator='\n')

    def __getitem__(self, idx):
        """
        Overwrites the built in python __getitem__ function and defines what is done to the dataset when an index is referenced.
        """
        item = self.prods_imgs.iloc[idx]
        label = self.get_category(idx)
        features = item[4]
        return (features, label)

    def __len__(self):
        """
        Overwrites the built in python __len__ function and defines what is done to the dataset when the len funtion is called.
        """
        return len(self.prods_imgs)

    def merge(self):
        """
        Funtion to merge the Images.csv and the Products.csv to a single dataframe which is then saved as prods_imgs.csv
        """
        prds_imgs = self.product_data.merge(self.image_data, left_on='id', right_on='product_id').rename(columns={'id_y': 'image_id'}).drop('id_x', axis=1)
        prds_imgs.to_csv('/home/danny/git/FBMarketplaceRanking/my.secrets.data/prods_imgs.csv')
        print(prds_imgs)

    def get_category(self, idx):
        """
        Function to obtain the category of an index position in the pros_imgs.csv
        Args: self
            : index
        Returns: Category -> str with / and , removed
        """
        item = self.prods_imgs.iloc[idx]
        cat = item[3]
        cat1 = cat.replace('/','')
        category = cat1.replace(',','')
        return category
        

dataset = CreateImageDataset()
print(len(dataset))
print(dataset[1061])
print(dataset.get_category(1061))






"""
code to use cuda later....
def get_default_device():
    
    It returns the device object representing the default device type
    :return: The device object
    
    Picking GPU if available or else CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
"""