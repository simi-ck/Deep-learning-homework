import os
import shutil
train_path = "./data/train/"
new_train_path = "./data/new_train/"
val_path = "./data/val/"
new_val_path = "./data/new_val/"
# for i in range(100):
#     curren_new_path = new_train_path + str(i) + "/"
#     os.mkdir(curren_new_path)
# for i in range(100):
#     current_path = train_path + str(i) + "/"
#     curren_new_path = new_train_path + str(i) + "/"
#     k = 0
#     if not os.path.exists(current_path):
#         print("error")
#     else:
#         fileDir = os.listdir(current_path)
#         fileDir.sort(key=lambda x:int(x[:-4]))
#         for fileName in fileDir:
#             if k < 300:
#                 shutil.copyfile(current_path + fileName, curren_new_path + fileName)
#                 k += 1


# for i in range(100):
#     curren_new_path = new_val_path + str(i) + "/"
#     os.mkdir(curren_new_path)
for i in range(100):
    current_path = val_path + str(i) + "/"
    curren_new_path = new_val_path + str(i) + "/"
    k = 0
    if not os.path.exists(current_path):
        print("error")
    else:
        fileDir = os.listdir(current_path)
        fileDir.sort(key=lambda x:int(x[:-4]))
        for fileName in fileDir:
            if k < 30:
                shutil.copyfile(current_path + fileName, curren_new_path + fileName)
                k += 1