import pandas as pd
from PIL import Image

class CleanImageData:
    """
    Class to define the methods of the image cleaning pipeline.
    """
    def __innit__(self, image_df:pd.DataFrame, path:str = "/home/danny/git/FBMarketplaceRanking/data/images/", resized_path:str ="/home/danny/git/FBMarketplaceRanking/data/resized_images/"):
        self.image_df = image_df
        self.path = path
        self.resized_path = resized_path

    def create_img_id_list(self):
        """
        Function to create a list of image id's.
        Args: self
        Returns: image_id_list
        """
        images_df = pd.read_csv("/home/danny/git/FBMarketplaceRanking/data/Images.csv")
        image_id_list = images_df["id"].to_list()
        return(image_id_list)

    def open_image(self, image_id):
        """
        Funtion to open the image with the python image library.
        Args:self
            :image_id
        Returns: image
        """
        path:str = "/home/danny/git/FBMarketplaceRanking/data/images/"
        image = Image.open(f"{path}{image_id}.jpg")
        return(image)

    def resize_image(self, final_size, image):
        """
        Function to resize the image.
        Args:final_size
            :image
        Returns:resized image
        """
        size = image.size
        ratio = float(final_size) / max(size)
        new_image_size = tuple([int(x*ratio) for x in size])
        image = image.resize(new_image_size, Image.ANTIALIAS)
        new_image = Image.new("RGB", (final_size, final_size))
        new_image.paste(image, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        return(new_image)

image_cleaner = CleanImageData()
resized_path:str ="/home/danny/git/FBMarketplaceRanking/data/resized_images/"
img_id_list = image_cleaner.create_img_id_list()
for img_id in img_id_list:
    image = image_cleaner.open_image(img_id)
    resized_image:Image = image_cleaner.resize_image(512,image)
    resized_image.Image.save(f'{img_id}_resized.jpg', resized_path)
    break

'''
if __name__ == '__main__':
    path = "images/"
    dirs = os.listdir(path)
    final_size = 512
    for n, item in enumerate(dirs[:5], 1):
        im = Image.open('images/' + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'{n}_resized.jpg')
'''