import pandas as pd


class CleanTabularData:
    '''
    '''
    def __innit__(self, products_df:pd.DataFrame):
        self.products_df = products_df

    def clean_prices(self):
        """
        """
        df1 = pd.read_csv("/home/danny/git/FBMarketplaceRanking/data/Products.csv", lineterminator="\n")
        df2 = df1["price"]
        df3 = df2.str.replace('£','')
        df4 = df3.str.replace(',','')
        products_df = df4.astype(float)

        print(products_df)


tabular = CleanTabularData()
tabular.clean_prices()