import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
import torch.optim as optim
#The first step: for constructing dataset
############################
def dataset():
    train_tfm = transforms.Compose([transforms.Resize((224, 224)),transforms.RandomRotation(45),transforms.ToTensor(),])
    test_tfm = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])
    batch_size = 128
    train_set = DatasetFolder("OxFlower17/train", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    test_set = DatasetFolder("OxFlower17/test", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


#The scecond step: design model
##############################
class SEResNet(nn.Module):
    def __init__(self, model):
        super(SEResNet, self).__init__()
        self.layer1 = nn.Sequential(model)
        #SE_Resnet 输出为(1,1000)的张量,而本例中共有17种花的类别
        # 故线性层输入为1000,输出为17 
        self.layer2 = nn.Linear(1000, 17)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
model = timm.create_model('seresnet50', pretrained=True)
model.eval()
model = SEResNet(model)
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model.to(device)


#The third step: contructing optimizer and loss function
##############################
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)


#The fourth step: constructing train function and test function
##############################
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader,0):
        inputs , target = data
        inputs, target = inputs.to(device), target.to(device)
        #print('inputs',inputs)
        #print('target',target)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d]loss:%.3f'%(epoch+1,batch_idx+1,running_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0 
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # print('labels:',labels)
            # print('labels.size(0)',labels.size(0))
            outputs = model(images)
            _,predicted = torch.max(outputs.data,dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' %(100*correct/total))
    
#main function
##########################
if __name__ == '__main__':
    train_loader, test_loader = dataset()
    for epoch in range(10):
        train(epoch)
        test()