import torch
from create_image_dataset import CreateImageDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



dataset = CreateImageDataset()
ngpu = 2
epochs = 1
batch_size = 32
lr = 0.001
num_classes = dataset.num_classes
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ImageClassifier(torch.nn.Module):
    def __init__(self, ngpu, num_classes):
        super(ImageClassifier, self).__init__()
        self.ngpu = ngpu
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        output_features = self.resnet50.fc.output_features
        self.linear = torch.nn.Linear(output_features, num_classes).to(device)
        self.main = torch.nn.Sequential(self.resnet50, self.linear).to(device)


    def forward(self, X):
        X = self.main(X)#return prediction
        return X

model_cnn = ImageClassifier(ngpu=ngpu, num_classes=num_classes)
print(model_cnn.resnet50)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_cnn.parameters(), lr=lr)