import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch
import matplotlib.pyplot as plt
import copy
from bert_dataset import MyDataSet,PadCollate
from torch.utils.data import DataLoader
from transformers import BertModel


cfg={
    "batch_size":4,
    "num_epochs":10,  
    'lr':1e-4,
    'optim':'Adam', 
    'momentum':0.9,  
    'weight_decay':1e-5, 
    'loss':'CEloss',
}



class BTextClassificationModel(nn.Module):
    def __init__(self,embed_dim=768, hidden_size=8,num_class=2):
        super(BTextClassificationModel, self).__init__()
        #self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        #self.lstm = nn.LSTM(embed_dim,hidden_size,num_layers, bidirectional=True,batch_first=True,dropout=0)
        self.backbone=BertModel.from_pretrained("bert-base-chinese")
        connect_num1=hidden_size * 2
        connect_num=connect_num1*2
        self.pool=nn.AdaptiveAvgPool1d(connect_num*connect_num1)
        self.fc1 = nn.Linear(connect_num*connect_num1, connect_num)
        self.act1=nn.ReLU(connect_num)
        self.out=nn.Linear(connect_num, num_class)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,embed):
        embed=embed.squeeze()
        out= self.backbone(embed)[0]
        out=out[:,0:1,:]
        out = F.relu(out)
        out = self.pool(out) 
        out =out.reshape(out.size()[0],-1)
        out = self.act1(self.fc1(out))
        out = self.out(out)
        out =  self.softmax(out)
        return out

if __name__ == '__main__':
    tr_dataset = MyDataSet(dataset_path='./train.csv')
    val_dataset = MyDataSet(dataset_path='./val.csv')
    # print(tr_dataset)
    tr_dataset_sizes=len(tr_dataset)
    val_dataset_sizes=len(val_dataset)
    
    train_dataloaders = DataLoader(tr_dataset, batch_size=cfg["batch_size"],collate_fn=PadCollate(dim=1),shuffle=True, num_workers=0)
    val_dataloaders = DataLoader(val_dataset, batch_size=cfg["batch_size"],collate_fn=PadCollate(dim=1),shuffle=True, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

#define model
    bert_name="bert-base-chinese"
    model=BTextClassificationModel(num_class=2)
    model=model.to(device)
#optimizer
    optimizer=torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[4,6], gamma=0.2)
#loss
    criterion = nn.CrossEntropyLoss()
#Training model
    best_acc=0.0
    best_epoch=0
    best_model_wt={}
    loss_list=[]
    tra_acc_list=[]
    val_acc_list=[]
    for epoch in range(cfg["num_epochs"]):
        print('Epoch {}/{}'.format(epoch, cfg["num_epochs"] - 1))
        print('-' * 10)
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_batch=tr_dataset_sizes//cfg["batch_size"]

        for iterate, (inputs, labels) in enumerate(train_dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 
            running_corrects += torch.sum(preds == labels.data)
            if iterate%50 ==49:
                print('Epoch {}/{} batch {}/{} loss {:.4f}'.format(epoch, cfg["num_epochs"] - 1,iterate,total_batch,loss.item()))

        scheduler.step()

        epoch_loss = running_loss / tr_dataset_sizes
        epoch_acc = running_corrects.double() / tr_dataset_sizes
        loss_list.append(epoch_loss)
        tra_acc_list.append(epoch_acc.cpu())
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))
        
        ####################validation###################
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for iterate, (inputs, labels) in enumerate(val_dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                #loss = criterion(outputs, labels)
                #running_loss += loss.item() 
                running_corrects += torch.sum(preds == labels.data) 

        epoch_acc = running_corrects.double() / val_dataset_sizes
        val_acc_list.append(epoch_acc.cpu())
        print('{} Acc: {:.4f}'.format("val",  epoch_acc))

        ###########save best#####################
        if epoch_acc >best_acc:
            best_acc=epoch_acc
            best_epoch=epoch
            best_model_wt=copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),'./best_model.pth')
            
    print("best epoch:{}  val_Acc:{:.4f}".format(best_epoch,best_acc))
    print("best model has been saved.")

    ###############result show###############
    x=range(cfg["num_epochs"])
    #print(x)
    figname='./loss_curve'
    plt.figure(1)
    plt.plot(x,loss_list,label=figname[2:])
    plt.title('training  loss')
    plt.xlabel('epoch')
    plt.ylabel('epoch loss')
    plt.legend()
    plt.savefig(figname)
    #plt.show()

    plt.figure(2)
    figname='./Acc_curve'
    plt.plot(x,tra_acc_list,label="tra_acc",color='b')
    plt.plot(x,val_acc_list,label="val_acc",color='r')
    plt.title('Acc Curve')
    plt.xlabel('epoch')
    plt.ylabel('epoch Acc')
    plt.legend()
    plt.savefig(figname)
    #plt.show()


       






