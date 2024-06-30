import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter
import os
from utils import save_checkpoint, accuracy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_cifar100_data_loaders(download, path, shuffle=False, batch_size=256, choice='stl10'):
    if choice=='imagenet' or choice=="init":
        data_transforms = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif choice=="stl10":
        data_transforms = transforms.Compose([transforms.Resize(96),
                                            transforms.ToTensor(),
                                            ])
    elif choice=="cifar10":
        data_transforms = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            ])

    train_dataset = datasets.CIFAR100(root=path, train=True, download=download,
                                    transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=4, drop_last=False, shuffle=shuffle)
    
    test_dataset = datasets.CIFAR100(root=path, train=False, download=download,
                                    transform=data_transforms)

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                                num_workers=4, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def load_model(path=None, pretrain=False):
    # Create model
    model = torchvision.models.resnet18(pretrained=pretrain)
    
    # freeze all layers except the last
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    num_classes = 100
    model.fc = torch.nn.Linear(512,num_classes)
    init.kaiming_normal_(model.fc.weight.data)
    # model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    if path is not None:
        # loading the trained check point data
        checkpoint = torch.load(path, map_location=device)

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
        log = model.load_state_dict(state_dict, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']

    model = model.to(device)
    return model

def train(model, data_path, save_path, choice):
    os.makedirs(save_path, exist_ok=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'tensorboard_logs'))
    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2
    optimizer = torch.optim.Adam(parameters, lr=0.01, weight_decay=0.0008)


    cifar100_train_loader, cifar100_test_loader = get_cifar100_data_loaders(path=data_path, download=False, choice=choice)

    # supervise learning on CIFAR100 dataset

    epochs = 10
    top1_train_accuracy_list = [0]
    top1_accuracy_list = [0]
    top5_accuracy_list = [0]
    epoch_list = [0]

    for epoch in range(epochs):
        top1_train_accuracy = 0
        model.train()
        for counter, (x_batch, y_batch) in enumerate(cifar100_train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        model.eval()
        with torch.no_grad():
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
        writer.add_scalar('train-acc/top5', top1_train_accuracy, epoch)

        top1_train_accuracy_list.append(top1_train_accuracy.item())
        top1_accuracy_list.append(top1_accuracy.item())
        top5_accuracy_list.append(top5_accuracy.item())
        epoch_list.append(epoch+1)
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    
    save_checkpoint({
            'epoch': epochs,
            'state_dict': model.state_dict(),
        }, is_best=False, filename=os.path.join(save_path, "model.pth.tar"))
       

    #accuracy曲线
    top1_train_accuracy_list.pop(0)
    top1_accuracy_list.pop(0)
    top5_accuracy_list.pop(0)
    epoch_list.pop(0)

    plt.figure(figsize = (16, 9))
    plt.rcParams.update({'font.size': 14})
    plt.title('CIFAR100 Accuracy Plot')
    plt.plot(epoch_list,top1_train_accuracy_list, c='b')
    plt.plot(epoch_list,top1_accuracy_list, c='g')
    plt.plot(epoch_list,top5_accuracy_list, c='r')
    plt.legend(['Training Accuracy', 'Top 1 Test Accuracy', 'Top 5 Test Accuracy'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show() 
    plt.savefig(os.path.join(save_path, 'plot.png'))

if __name__ == "__main__":
    model_path = "runs/stl10-20step-w5e-5/checkpoint_0020.pth.tar"
    data_path = "datasets"
    save_path = "classify_CIFAR100/weight_decay/stl10-20step-w5e-5"
    choice = 'stl10'
    model = load_model(path=model_path,pretrain=False)
    train(model,"datasets",save_path, choice=choice)



