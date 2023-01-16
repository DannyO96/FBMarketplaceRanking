Facebook Marketplace's Recommendation Ranking System
This is my machine learning specialisation project 

Milestone 1:
I have beggun cleaning the raw data in order for it to be processed by the machine learning model namey normalising the prices collummn in the products csv by turning the csv into a DataFrame and using the pd.Dataframe.replace method to remove the unwanted pouonds and commas from the collumns. The images dataset has been cleaned by resizing the images and ensuring all the images are RGB before saving them in a new directory named resized_images, the image cleaning was conducted using methods from the python image library. Once the data has been cleaned the images, labels and features have been turned into tensors so they can be fed into the convolutional neural network.

Milestone 2:
I have begune creating the vision model to predict the categories of the training dataset. I have built a training loop and am tracking the losses of each batch as scalars using tensor board. The model is then saved and the weights are saved in a seperate folder in order to aid hyperparameter tuning.