import torch
from create_image_dataset import CreateImageDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


dataset = CreateImageDataset()
ngpu = 1
epochs = 1
batch_size = 8
lr = 0.001
num_classes = dataset.num_classes
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=1 )

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ImageClassifier(torch.nn.Module):
    """
    This class inherits from torch.nn.module and is the image classification neural network it uses the resnet 50 pretrained nueral network from pytorch
    with some linear layers added to make it specific for the image classification of the classes of my images from the facebook marketplace.
    
    """
    def __init__(self, ngpu, num_classes):
        super(ImageClassifier, self).__init__()
        self.ngpu = ngpu
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = torch.nn.Linear(out_features, num_classes).to(device)
        self.main = torch.nn.Sequential(self.resnet50, self.linear).to(device)


    def forward(self, X):
        X = self.main(X)#return prediction
        return X

model_cnn = ImageClassifier(ngpu=ngpu, num_classes=num_classes)
print(model_cnn.resnet50)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_cnn.parameters(), lr=lr)
losses = []

for epoch in range(epochs):
    histogram_accuracy = []
    accuracy = 0
    for i, (data, labels) in tqdm(enumerate(dataloader), total = len(dataloader)):
        data = data.to(device)
        lables = labels.to(device)
        optimizer.zero_grad()
        outputs = model_cnn(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        accuracy = torch.sum(torch.argmax(outputs, dim = 1)==labels).item()/len(labels)
        histogram_accuracy.append(accuracy)
        losses.append(loss.item())
        tqdm(enumerate(dataloader), total = len(dataloader)).set_description(f'epoch = {epoch + 1}/{epochs}. Acc = {round(torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels), 2)}, Losses = {round(loss.item(), 2)}')
        print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")
        print('-'*20)
        print(f"Accuracy: {torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels)}")
        print('-'*20)
        
