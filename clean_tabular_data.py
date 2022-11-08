import pandas as pd

class CleanTabularData:
    '''
    Class to define the methods of the tabular data cleaning pipeline.
    '''
    def __innit__(self, products_df:pd.DataFrame):
        self.products_df = products_df

    def clean_prices(self):
        """
        This is a function to remove the comma's and pound signs from the price collumn of the dataframe obtained from
        the products.csv data.

        Args: None

        Returns: Cleaned dataframe of product prices
        """
        unclean_df = pd.read_csv("/home/danny/git/FBMarketplaceRanking/my.secrets.data/Products.csv", lineterminator="\n")
        no_price_df = unclean_df.drop(columns=["price"])
        price_only_df = unclean_df["price"]
        df2 = price_only_df
        df2 = df2.str.replace('£','')
        df2 = df2.str.replace(',','')
        df2 = df2.astype(float)
        products_df = no_price_df.join(df2)
        products_df = products_df.iloc[:,1:]

        return(products_df)

tabular = CleanTabularData()
clean_df = tabular.clean_prices()
print(clean_df)
clean_df.to_csv('/home/danny/git/FBMarketplaceRanking/my.secrets.data/CleanProducts.csv')
