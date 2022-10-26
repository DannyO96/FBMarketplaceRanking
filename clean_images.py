import pandas as pd
from PIL import Image

class CleanImageData:
    """
    """
    def __innit__(self, image_df:pd.DataFrame):
        self.image_df = image_df

    def clean_image_data(self, image_mode, image_size):
        if image_mode != 'RGB':
            image_mode = 'RGB'

    def image_info(self, image_id):
        path:str ="/home/danny/git/FBMarketplaceRanking/data/images/"
        image = Image.open(f"{path}{image_id}.jpg")
        image_size = image.size
        image_mode = image.mode
        return(image_mode, image_size)

    def create_img_id_list(self):
        images_df = pd.read_csv("/home/danny/git/FBMarketplaceRanking/data/Images.csv")
        image_id_list = images_df["id"].to_list()
        return(image_id_list)

image_cleaner = CleanImageData()
img_id_list = image_cleaner.create_img_id_list()
for img_id in img_id_list:
    img_mode, img_size = image_cleaner.image_info(img_id)



       




'''

from PIL import Image
import os

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':
    path = "images/"
    dirs = os.listdir(path)
    final_size = 512
    for n, item in enumerate(dirs[:5], 1):
        im = Image.open('images/' + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'{n}_resized.jpg')

'''