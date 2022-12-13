import torch
from create_image_dataset import CreateImageDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ImageClassifier(torch.nn.Module):
    """
    This class inherits from torch.nn.module and is the image classification neural network it uses the resnet 50 pretrained nueral network from pytorch
    with some linear layers added to make it specific for the image classification of the classes of my images from the facebook marketplace. And of course an
    activation function to prevent linearity.
    """
    def __init__(self, device, num_classes):
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = torch.nn.Linear(out_features, num_classes).to(device)
        self.main = torch.nn.Sequential(self.resnet50,torch.nn.ReLU(), self.linear).to(device)


    def forward(self, X):
        X = self.main(X)#return prediction
        return X

def train(model, epochs = 10):

    print(model.resnet50)
    batch_idx = 0
    lr = 0.001
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr==lr)
    losses = []
    writer = SummaryWriter()
    
    for epoch in range(epochs):
        histogram_accuracy = []
        accuracy = 0
        for i, (data, labels) in tqdm(enumerate(dataloader), total = len(dataloader)):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            accuracy = torch.sum(torch.argmax(outputs, dim = 1)==labels).item()/len(labels)
            histogram_accuracy.append(accuracy)
            losses.append(l.item())
            tqdm(enumerate(dataloader), total = len(dataloader)).set_description(f'epoch = {epoch + 1}/{epochs}. Acc = {round(accuracy, 2)}, Losses = {round(l.item(), 2)}')
            print(f"Epoch: {epoch} Batch: {i} Loss: {l.item()}")
            print('-'*20)
            print(f"Accuracy: {accuracy}")
            print('-'*20)
            writer.add_scalar('Loss', l.item(), batch_idx)
            
if __name__ == '__main__':
    dataset = CreateImageDataset()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = ImageClassifier(device, num_classes = dataset.num_classes)
    train(model)