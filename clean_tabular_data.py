import pandas as pd


class CleanTabularData:
    '''
    '''
    def __innit__(self, products_df:pd.DataFrame):
        self.products_df = products_df

    def clean_prices(self):
        """
        """
        unclean_df = pd.read_csv("/home/danny/git/FBMarketplaceRanking/data/Products.csv", lineterminator="\n")
        no_price_df = unclean_df.drop(columns=["price"])
        price_only_df = unclean_df["price"]
        df2 = price_only_df
        df2 = df2.str.replace('£','')
        df2 = df2.str.replace(',','')
        df2 = df2.astype(float)
        products_df = no_price_df.join(df2)
        products_df = products_df.iloc[:,1:]

        print(products_df)

tabular = CleanTabularData()
tabular.clean_prices()