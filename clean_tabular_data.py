import pandas as pd
import unicodedata


class CleanTabularData:
    '''
    '''
    def __innit__(self):
        self.product_df : pd.DataFrame

    def clean_prices(price: str):
        """
        """
        products_df = pd.read_csv("/home/danny/git/FBMarketplaceRanking/data/Products.csv", lineterminator="\n")
        df1 = products_df["price"]
        df1 = df1.str.replace('£','')
        df1 = df1.str.replace(',','')
        prods_df = df1.astype(float)



        print(prods_df)


tabular = CleanTabularData()
tabular.clean_prices()