import pandas as pd
import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageFile

#ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        self.labels = self.prods_imgs['category'].to_list()
        self.num_classes = len(set(self.labels))
        self.image_id = self.prods_imgs['image_id'].values[0]
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        """
        Overwrites the built in python __getitem__ function and defines what is done to the dataset when an index is referenced.
        """
        label = self.labels[idx]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        image = Image.open('/home/danny/git/FBMarketplaceRanking/my.secrets.data/resized_images/' + self.image_id + '_resized.jpg')
        image_tensor = self.transform(image)
        return (image_tensor, label)

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

    def get_category_clean(self, idx):
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

    def split_train_test(dataset, train_percentage):
        """
        Funtioon to split the dataset into a training dataset for training the model and a validation dataset for validating the models ability to predict the
        categories of images
        Args: dataset
            : train_percentage
        Returns: train_dataset
               : validation_dataset
        """
        train_split = int(len(dataset) * train_percentage)
        train_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [train_split, len(dataset) - train_split]
        ) 
        return train_dataset, validation_dataset


dataset = CreateImageDataset()
print(dataset[0][0])
print(dataset.decoder[int(dataset[0][1])])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
for i, (data, labels) in enumerate(dataloader):
    print(data)
    print(labels)
    print(data.size())
    if i == 0:
        break
"""
with open('/home/danny/git/FBMarketplaceRanking/my.secrets.data/image_decoder.pkl','rb') as f:
    data = pickle.load(f)
print(data)

print(len(dataset))
print(dataset.num_classes)
print(dataset[1061])
print(dataset.get_category_clean(10))

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