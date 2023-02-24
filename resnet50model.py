import pickle
import torch
from create_image_dataset import CreateImageDataset
from itertools import product
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ImageClassifier(torch.nn.Module):
    """
    This class inherits from torch.nn.module and is the image classification neural network.
    It uses the ResNet50 pre-trained neural network from PyTorch and adds linear layers 
    to make it specific for the image classification of the classes of my images from the Facebook Marketplace. 
    An activation function is also added to prevent linearity.
    """
    def __init__(self, device, num_classes):
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = torch.nn.Linear(out_features, num_classes).to(device)
        self.regularisation = torch.nn.Dropout(p=0.5, inplace=False)
        self.main = torch.nn.Sequential(
            self.resnet50,
            torch.nn.ReLU(),
            self.regularisation, 
            self.linear
            ).to(device)

    def forward(self, X):
        """
        Function to compute output tensors from input tensors
        Args: 
            X: input tensor
        Returns: 
            X: output tensor
        """
        X = self.main(X)#return prediction
        return X

def train(model, epochs = 5):
    """
    This function trains a given model for a certain number of epochs using different combinations of hyperparameters. 
    It also logs the accuracy and loss metrics for each epoch using Tensorboard's SummaryWriter.
    Args: 
        model: model used to predict the category of images specified in self.main of the init
        epochs: total number of iterations of all the training data in one cycle for training
    Returns: 
        trained model
        
    """
    #Hyperparameters
    print(model.resnet50)
    parameters = dict(
        lr = [0.01, 0.001],
        batch_size = [8,16,32],
        shuffle = [True, False]
        )
    param_values = [v for v in parameters.values()]
    

    for run_id, (lr,batch_size,shuffle) in enumerate(product(*param_values)):
        print("run id:", run_id+1)
        dataset = CreateImageDataset() #create dataset object
        dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=shuffle)

        criterion = torch.nn.CrossEntropyLoss() #define loss function
        optimizer = torch.optim.Adam(model.parameters(), lr==lr)
        comment = f' batch_size = {batch_size} lr = {lr} shuffle = {shuffle}'
        writer = SummaryWriter(comment=comment) #create summary writer object to log metrics on tensorboard
        for epoch in range(epochs):
            #losses = []
            #histogram_accuracy = []
            accuracy = 0
            total_loss = 0
            total_correct = 0

            for i, (data, labels) in tqdm(enumerate(dataloader), total = len(dataloader)):
                data = data.to(device)
                labels = labels.to(device)   
                
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss+= loss.item()
                total_correct+= outputs.argmax(dim=1).eq(labels).sum().item()
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accuracy = torch.sum(torch.argmax(outputs, dim = 1)==labels).item()/len(labels)

                #histogram_accuracy.append(accuracy)
                #losses.append(loss.item())
                tqdm(enumerate(dataloader), total = len(dataloader)).set_description(f'epoch = {epoch + 1}/{epochs}. Acc = {round(accuracy, 2)}, Losses = {round(loss.item(), 2)}')
                
                print(f"Epoch: {epoch} Batch: {i} Loss: {loss.item()}")
                print('-'*20)
                print(f"Accuracy: {accuracy}")
                print('-'*20)
                print(f"Total number of Correct Guesses: {total_correct}")
                print('-'*20)
                
            #accuracy = total_correct/ len(train_set)
            #writer.add_scalar('Loss', loss.item(), batch_idx)
            writer.add_scalar("Accuracy",accuracy, epoch)
            writer.add_scalar('Total Loss', total_loss, epoch)
            writer.add_scalar("Correct", total_correct, epoch)
            #writer.add_graph('Loss', loss.item(), batch_idx)
            #writer.add_graph('Accuracy', accuracy, batch_idx)
            #writer.add_scalar('Accuracy', accuracy.item(), batch_idx)
            
    writer.add_hparams(
            {"lr": lr, "bsize": batch_size, "shuffle":shuffle},
            {
                "accuracy": accuracy,
                "loss": total_loss,
            },
        )
    torch.save(model.state_dict(), 'model_evaluation/weights.pth')
    with open('my.secrets.data/image_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)


if __name__ == '__main__':
    dataset = CreateImageDataset()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #dataloader = DataLoader(dataset)
    model = ImageClassifier(device, num_classes = dataset.num_classes)
    train(model)