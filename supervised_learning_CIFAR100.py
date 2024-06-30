from models.resnet18_CIFAR100 import resnet18_CIFAR100
import torch
from torchvision import transforms, datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils import save_checkpoint, accuracy
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def cifar100_data_loaders(path, shuffle=False, batch_size=256):  #数据加载函数（划分训练集和测试集）

    data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = datasets.CIFAR100(root=path, train=True, download=False,
                                    transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=0, drop_last=False, shuffle=shuffle)
    
    test_dataset = datasets.CIFAR100(root=path, train=False, download=False,
                                    transform=test_transforms)


    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                                num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def supervised_train(model, data_path, save_path, epochs):  #监督学习训练函数
    os.makedirs(save_path, exist_ok=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'tensorboard_logs'))
    #Linear Classification Protocol
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=0.001, weight_decay=1e-4)

    cifar100_train_loader, cifar100_test_loader = cifar100_data_loaders(path=data_path, shuffle=True)

    epochs = epochs
    top1_train_accuracy_list = [0]
    top1_accuracy_list = [0]
    top5_accuracy_list = [0]
    epoch_list = [0]

    for epoch in tqdm(range(epochs)):
        top1_train_accuracy = 0
        losses = 0
        for counter, (x_batch, y_batch) in enumerate(cifar100_train_loader):  
            x_batch = x_batch.to(device)  #数据放到gpu上跑
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]
            losses +=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses /= (counter+1)
        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(cifar100_test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
        
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

        writer.add_scalar('test-acc/top1', top1_accuracy, epoch)
        writer.add_scalar('test-acc/top5', top5_accuracy, epoch)
        writer.add_scalar('train-acc/top1',top1_train_accuracy, epoch)
        writer.add_scalar('train/loss', losses, epoch)

        top1_train_accuracy_list.append(top1_train_accuracy.item())
        top1_accuracy_list.append(top1_accuracy.item())
        top5_accuracy_list.append(top5_accuracy.item())
        epoch_list.append(epoch+1)
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        if epoch%20 == 0 and epoch>1:
            save_checkpoint({
            'epoch': epochs,
            'state_dict': model.state_dict(),
        }, is_best=False, filename=os.path.join(save_path, "checkpoint.pth.tar"))
            
    save_checkpoint({
            'epoch': epochs,
            'state_dict': model.state_dict(),
        }, is_best=False, filename=os.path.join(save_path, "model.pth.tar"))
       

    # accuracy曲线
    top1_train_accuracy_list.pop(0)
    top1_accuracy_list.pop(0)
    top5_accuracy_list.pop(0)
    epoch_list.pop(0)

    plt.figure(figsize = (16, 9))
    plt.rcParams.update({'font.size': 14})
    plt.title('CIFAR100 Accuracy曲线图(监督学习)')
    plt.plot(epoch_list,top1_train_accuracy_list, c='b')
    plt.plot(epoch_list,top1_accuracy_list, c='g')
    plt.plot(epoch_list,top5_accuracy_list, c='r')
    plt.legend(['Training Accuracy', 'Test Accuracy(top1)', 'Test Accuracy(top5)'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show() 
    plt.savefig(os.path.join(save_path, "supervised_on_CIFAR100-accuracy曲线图.png"))

if __name__ == "__main__":
    data_path = "datasets"
    save_path = "/classify_CIFAR100"
    model = resnet18_CIFAR100().to(device)  #gpu加速训练
    supervised_train(model, data_path, save_path, epochs=100)
