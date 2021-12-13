import pandas as pd
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch
import matplotlib.pyplot as plt
# import copy
from bert_dataset import MyDataSet,PadCollate
from torch.utils.data import DataLoader
from transformers import BertModel
# from matplotlib.pyplot import MultipleLocator

class BTextClassificationModel(nn.Module):
    def __init__(self,embed_dim=768, hidden_size=8,num_class=2):
        super(BTextClassificationModel, self).__init__()
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
    
    
def segmentation_data(path):
    pd_all = pd.read_csv(path)
    #划分数据6：2：2
    train_divide=math.floor(pd_all.shape[0]*0.6)
    val_divide=math.floor(pd_all.shape[0]*0.8)
    test_divide=math.floor(pd_all.shape[0]*1.0)
    file_data = pd.read_csv('./weibo_senti_100k.csv', encoding='utf-8')
    file_data.drop_duplicates(keep='first', inplace=True)  # 去重，只保留第一次出现的样本
    file_data = file_data.sample(frac=1.0)  # 全部打乱
    df_train, df_val,df_test = file_data.iloc[:train_divide], file_data.iloc[train_divide:val_divide], file_data.iloc[val_divide:test_divide]
    df_train.to_csv('train.csv',sep=',', header=True, index=False)
    df_val.to_csv('val.csv',sep=',', header=True, index=False)
    df_test.to_csv('test.csv',sep=',', header=True, index=False)
    

def train(batch_size, epochs, lr, tr_dataset, val_dataset, model, device):    
    tr_dataset_sizes=len(tr_dataset)
    val_dataset_sizes=len(val_dataset)
    
    train_dataloaders = DataLoader(tr_dataset, batch_size=batch_size,collate_fn=PadCollate(dim=1),shuffle=True, num_workers=0)
    val_dataloaders = DataLoader(val_dataset, batch_size=batch_size,collate_fn=PadCollate(dim=1),shuffle=True, num_workers=0)
    
    optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[4,6], gamma=0.2)
    criterion = nn.CrossEntropyLoss()
    total_batch=tr_dataset_sizes//batch_size
    
    best_acc=0.0
    best_epoch=0
    # best_model_wt={}
    loss_list=[]
    tra_acc_list=[]
    val_acc_list=[]
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        
        for i, (inputs, labels) in enumerate(train_dataloaders):
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
            if i%50 ==49:
                print('Epoch {}/{} batch {}/{} loss {:.4f}'.format(epoch, epochs - 1,i,total_batch,loss.item()))
        scheduler.step()

        epoch_loss = running_loss / tr_dataset_sizes
        epoch_acc = running_corrects.double() / tr_dataset_sizes
        loss_list.append(epoch_loss)
        tra_acc_list.append(epoch_acc.cpu())
        # print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))
        print("[ Epoch{}/{} ] Train | Loss:{:.4f} Acc: {:.4f}% ".format(epoch, epochs - 1, epoch_loss, epoch_acc))
            
        
        ####################validation###################
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_dataloaders):
                inputs = inputs.to(device)
                labels = labels.to(device)    
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                #loss = criterion(outputs, labels)
                running_loss += loss.item() 
                running_corrects += torch.sum(preds == labels.data) 

        epoch_loss = running_loss / val_dataset_sizes
        epoch_acc = running_corrects.double() / val_dataset_sizes
        val_acc_list.append(epoch_acc.cpu())
        print("[ Epoch{}/{} ] Val | Loss:{:.4f} Acc: {:.4f}% ".format(epoch, epochs - 1, epoch_loss, epoch_acc))
        
        ###########save best#####################
        if epoch_acc >best_acc:
            best_acc=epoch_acc
            best_epoch=epoch
            # best_model_wt=copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),'./best_model.pth')
            
    print("best epoch:{}  val_Acc:{:.4f}".format(best_epoch,best_acc))
    print("best model has been saved.")

    ###############result show###############
    x=range(epochs)
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


def test(batch_size, test_dataset, model, device):
    test_dataset_sizes=len(test_dataset)
    test_dataloaders = DataLoader(test_dataset, batch_size=batch_size,collate_fn=PadCollate(dim=1),shuffle=True, num_workers=0)

    #define model
    # model = BTextClassificationModel(num_class=2)
    # model_wt=torch.load("./best_model.pth")
    # model.load_state_dict(model_wt)
    # model=model.to(device)
    #test model
    val_acc_list=[]
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data) 

    epoch_acc = running_corrects.double() / test_dataset_sizes
    val_acc_list.append(epoch_acc.cpu())
    print('{} Acc: {:.4f}'.format("Test dataset",  epoch_acc))

       
if __name__ == '__main__':
    
    path="weibo_senti_100k.csv"
    #划分数据
    # segmentation_data(path)
    
    batch_size = 4
    epochs = 10  
    lr = 1e-4
    momentum = 0.9
    weight_decay = 1e-5
    tr_dataset = MyDataSet(dataset_path='./train.csv')
    val_dataset = MyDataSet(dataset_path='./val.csv')
    test_dataset = MyDataSet(dataset_path='./test.csv')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    bert_name="bert-base-chinese"
    #训练
    model=BTextClassificationModel(num_class=2)
    model=model.to(device)
    train(batch_size, epochs, lr, tr_dataset, val_dataset, model, device)

    #测试
    test_model = BTextClassificationModel(num_class=2)
    model_wt=torch.load("./best_model.pth")
    test_model.load_state_dict(model_wt)
    test_model=test_model.to(device)
    test(batch_size, test_dataset, test_model, device)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
