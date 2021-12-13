import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import argparse
import os
from PIL import Image
from torch.autograd import Variable
import glob

class BottleBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, down_sample=False, expansion=4, isNormalize=True, isResidual=True):
        super(BottleBlock, self).__init__()
        self.down_sample = down_sample
        self.expansion = expansion
        self.isNormalize = isNormalize
        self.isResidual = isResidual
        if self.isNormalize:
            # resnet中bottleBlock有输入通道和输出通道相同和不同两种 相同需要进行转换
            self.bottleBlock = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channel, output_channel * self.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(output_channel * self.expansion))
        else:
            self.bottleBlock = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channel, output_channel * self.expansion, kernel_size=1, stride=1, bias=False))

        self.relu = nn.ReLU(inplace=True)
        if self.down_sample:
            self.downSample = nn.Sequential(nn.Conv2d(input_channel, output_channel*self.expansion, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(output_channel*self.expansion))

    def forward(self, x):
        if self.down_sample:
            input = self.downSample(x)
        else:
            input = x
        if self.isResidual:
            result = self.bottleBlock(x) + input
        else:
            result = self.bottleBlock(x)
        result = self.relu(result)
        return result

class ResNet(nn.Module):
    def __init__(self, blocks, drop_rate, classes=100, expansion=4, isNormalize=True, isResidual=True):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.drop_rate = drop_rate
        self.isNormalize = isNormalize
        self.isResidual = isResidual
        self.classes = classes
        # 图片3x224x224经过卷积层1 64x112x112
        self.init = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # 最大池化64x56x56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # block1 256x56x56
        # resnet50有4个blockLayer, 每个layer依次有3,4,6,3个bottleBlock
        self.block1 = self.blockLayer(64, 64, 1, blocks[0], self.isNormalize, self.isResidual)
        # block2 512x28x28
        self.block2 = self.blockLayer(256, 128, 2, blocks[1], self.isNormalize, self.isResidual)
        # block 31024x14x14
        self.block3 = self.blockLayer(512, 256, 2, blocks[2], self.isNormalize, self.isResidual)
        # block4 2048x7x7
        self.block4 = self.blockLayer(1024, 512, 2, blocks[3], self.isNormalize, self.isResidual)
        # 平均池化 2048x1x1
        self.avgpool = nn.AvgPool2d(kernel_size=7, padding=1)
        # 全连接 输出100类
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(self.drop_rate),
                                nn.Linear(512, self.classes), nn.LogSoftmax(dim=1))

    def blockLayer(self, input_channel, output_channel, stride, block_num, isNormalize, isResidual):
        blockModel = []
        blockModel.append(BottleBlock(input_channel, output_channel, stride=stride, down_sample=True, isNormalize=isNormalize, isResidual=isResidual))
        for i in range(1, block_num):
            blockModel.append(BottleBlock(output_channel * self.expansion, output_channel, isNormalize=isNormalize, isResidual=isResidual))
        return nn.Sequential(*blockModel)

    def forward(self, x):
        conv_result = self.init(x)
        maxpool_result = self.maxpool(conv_result)
        block_result = self.block1(maxpool_result)
        block_result = self.block2(block_result)
        block_result = self.block3(block_result)
        block_result = self.block4(block_result)
        avgpool_result = self.avgpool(block_result)
        avgpool_result = torch.flatten(avgpool_result, 1)
        class_result = self.fc(avgpool_result)
        return class_result

class ImageNetClassify:
    def __init__(self, train_dir, val_dir, batch_size, epoches, layer_num, isNormalize, isResidual, isDecay):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.epoches = epoches
        self.layer_num = layer_num
        self.isNormalize = isNormalize
        self.isRedisual = isResidual
        self.isDecay = isDecay
        self.train_transform = transforms.Compose([transforms.RandomRotation(10), transforms.Resize(256),
                                            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                            transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        self.val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
        self.val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_transform)

    def loadTrainData(self):
        train_data = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_data

    def loadValData(self):
        val_data = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        return val_data
    def loadOptimizer(self, model, lr):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if self.isDecay:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            return scheduler
        else:
            return optimizer
    def trainProcess(self, lr, dr):
        if self.layer_num == 50:
            model = ResNet([3, 4, 6, 3], dr, isNormalize=self.isNormalize, isResidual=self.isRedisual).cuda()
        elif self.layer_num == 101:
            model = ResNet([3, 4, 23, 3], dr, isNormalize=self.isNormalize, isResidual=self.isRedisual).cuda()
        else:
            model = ResNet([2, 2, 2, 2], dr, isNormalize=self.isNormalize, isResidual=self.isRedisual).cuda()
        # if self.how_normal == "bn":
        #     model = ResNet(self.blocks, dr).cuda()
        # else:
        #     model = ResNetLN(self.blocks, dr).cuda()
        train_data = self.loadTrainData()
        val_data = self.loadValData()
        train_data_size = len(self.train_dataset)
        val_data_size = len(self.val_dataset)

        if self.isNormalize and self.isRedisual:
            fs = open("./result/train/train_val_lr__{}__dropRate_{}_resnet{}_normalized_residualed.txt".format(lr, dr, self.layer_num), "a")
        elif self.isNormalize and not self.isRedisual:
            fs = open("./result/train/train_val_lr_{}_dropRate_{}_resnet{}_normalized_no_residual.txt".format(lr, dr, self.layer_num),"a")
        elif not self.isNormalize and self.isRedisual:
            fs = open("./result/train/train_val_lr_{}_dropRate_{}_resnet{}_no_normalized_residual.txt".format(lr, dr, self.layer_num),"a")
        else:
            fs = open("./result/train/train_val_lr_{}_dropRate_{}_resnet{}_no_normalized_no_residual.txt".format(lr, dr, self.layer_num),"a")
        # for param in model.parameters():
        #     param.requires_grad = True
        loss_function = nn.NLLLoss()
        optimizer = self.loadOptimizer(model, lr=lr)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        best_acc = 0
        best_epoch = 0

        for epoch in range(self.epoches):
            # epoch_start_time = time.time()
            print("Epoch:{}/{}".format(epoch+1, self.epoches))
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0

            for i, (inputs, labels) in enumerate(train_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                # loss.item()返回浮点数更精确
                train_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                train_acc += acc.item() * inputs.size(0)

            #验证集上不计算梯度 节省时间
            with torch.no_grad():
                model.eval()
                for j, (inputs, labels) in enumerate(val_data):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))
                    val_acc += acc.item() * inputs.size(0)

            # optimizer.step()
            avg_train_loss = train_loss / train_data_size
            avg_train_acc = train_acc / train_data_size

            avg_valid_loss = val_loss / val_data_size
            avg_valid_acc = val_acc / val_data_size

            # record.append([avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc])
            print(avg_train_acc)
            if avg_train_acc > best_acc:
                best_acc = avg_train_acc
                best_epoch = epoch + 1

            # epoch_end_time = time.time()
            fs.write("Resnet" + str(self.layer_num) + " " + "epoch " + str(epoch+1) + " train_loss: " + str(avg_train_loss) + " train_acc: " + str(avg_train_acc)
                         + " val_loss: " + str(avg_valid_loss) + " val_acc: " + str(avg_valid_acc) + "\n")
            # print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            #         epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
            #         epoch_end_time - epoch_start_time))
            # print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        fs.write("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        fs.close()
        if self.isNormalize and self.isRedisual:
            torch.save(model.state_dict(), "./model/lr_{}_dr_{}_resnet{}_normalized_residual.pt".format(lr, dr, self.layer_num))
        elif self.isNormalize and not self.isRedisual:
            torch.save(model.state_dict(), "./model/lr_{}_dr_{}_resnet{}_normalized_no_residual.pt".format(lr, dr, self.layer_num))
        elif not self.isNormalize and self.isRedisual:
            torch.save(model.state_dict(), "./model/lr_{}_dr_{}_resnet{}_no_normalize_residual.pt".format(lr, dr, self.layer_num))
        else:
            torch.save(model.state_dict(), "./model/lr_{}_dr_{}_resnet{}_no_normalize_no_residual.pt".format(lr, dr, self.layer_num))
        # torch.save(model.state_dict(), "./model/lr_{}_dr_{}_resnet{}.pt".format(lr, dr, self.layer_num))

def test(test_dir, model_path):
    fs = open("./result/test/test_lr_0.0001_dr_0.3_resnet18.txt", 'w')
    class_name = [str(i) for i in range(0, 100)]
    model = ResNet([2, 2, 2, 2], 0.3).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    picList = os.listdir(test_dir)
    picList.sort(key=lambda x: int(x[:-4]))
    pic_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    for picName in picList:
        # print(picName)
        current_pic = Image.open(test_dir + picName)
        current_pic = pic_transform(current_pic)
        current_pic = torch.unsqueeze(current_pic, dim=0).cuda()
        outputs = model(current_pic)
        _, predict = torch.max(outputs, 1)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        # _, pre_class = predict.topk(1, dim=1)
        perc = percentage[int(predict)].item()
        result = class_name[predict]
        fs.write("test/"+picName + " " + result + "\n")
        # print(result)
    fs.close()
        # pics = np.array([np.array(Image.open(frame).convert('L').resize((224, 224))) for frame in picList])
        # pics = torch.from_numpy(pics).type(torch.FloatTensor).cuda()
        # pics = Variable(torch.unsqueeze(pics, dim=1).float(), requires_grad=False)
        # outputs = model(pics)
        # _, predicts = torch.max(outputs, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="set some parameters")
    parser.add_argument('--dropRate', type=float, default=0.5, help='set to change dropout rate')
    parser.add_argument('--isNormal', type=bool, default=True, help='add or not add normalize layer')
    parser.add_argument('--learningRate', type=float, default='0.01', help='set to change learning rate')
    parser.add_argument('--isResidual', type=bool, default=True, help='add or not add residual layer')
    parser.add_argument('--layerNum', type=int, help='set layer num 50, 101, 152')
    parser.add_argument('--isLrDecay', type=bool, default=True, help='set or not set lr decay')
    args = parser.parse_args()
    # for lr in [1e-2, 1e-3, 1e-4]:
    #     for dr in [0.3, 0.5, 0.7]:
    # for i in [True, False]:
    #     classifier_resnet18 = ImageNetClassify("./data/new_train", "./data/new_val", 30, 10, 18, i, args.isResidual)
    #     classifier_resnet18.trainProcess(1e-4, 0.3)
    # for j in [True, False]:
    #     classifier_resnet18 = ImageNetClassify("./data/new_train", "./data/new_val", 30, 10, 18, args.isNormal, j)
    #     classifier_resnet18.trainProcess(1e-4, 0.3)
    classifier_resnet18 = ImageNetClassify("./data/new_train", "./data/new_val", 30, 10, 18, args.isNormal, args.isResidual,
                                           args.isLrDecay)
    classifier_resnet18.trainProcess(1e-4, 0.3)
    # test("./data/test/", "./model/lr_0.0001_dr_0.3_resnet18.pt")