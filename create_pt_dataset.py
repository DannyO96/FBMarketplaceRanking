import pandas as pd
import PIL
from torch.utils.data import Dataset, DataLoader



class CreatePTDataset(Dataset):
    """
    This class inherits from torch.utils.data.Dataset
    
    """
    def __innit__(self):
        super().__init__()
        self.data = pd.read_csv('/my.secrets.data/Products.csv')

    def __getitem__(self, index):
        item = self.data.iloc[index]
        id = item[2]
        product_name = item[3]
        category = item[4]
        product_description = item[5]
        price = item[6]
        location = item[7]
        url = item[8]
        page_id = item[9]
        create_time = item[10]
        label = item[-1]
        features = [id, product_name, product_description, category, price]
        other_info = [location, url, page_id, create_time]
        return {label, features, other_info}

    def __len__(self):
        return len(self.data)


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