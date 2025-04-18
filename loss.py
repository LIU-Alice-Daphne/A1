import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)

# x_val_loss=0
# y_val_loss=0
# x_train_loss=0
# y_train_loss=0

def loss(name):


    train_loss_path = os.path.join('models', name, 'train_loss.txt')
    val_loss_path = os.path.join('models', name, 'val_loss.txt')



    try:
        y_train_loss = data_read(train_loss_path)        # loss值，即y轴
        x_train_loss = range(len(y_train_loss))			 # loss的数量，即x轴
    except:
        y_train_loss = 0
        x_train_loss = 0
    try:
        y_val_loss = data_read(val_loss_path)  # loss值，即y轴
        x_val_loss = range(len(y_val_loss))  # loss的数量，即x轴
    except:
        x_val_loss = 0
        y_val_loss = 0
    fig = plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



    plt.xlabel('epochs')    # x轴标签
    plt.ylabel('loss')     # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.plot(x_val_loss, y_val_loss, linewidth=1, linestyle="solid", label="val loss",color='red')

    plt.legend()
    plt.title('Loss curve')

    # plt.savefig('models/dsb2018_96_UNet_woDS/loss.png')
    # plt.savefig('models/dsb2018_96_NestedUNet_woDS/loss.png')
    # plt.savefig('models/dsb2018_96_FPN_woDS/loss.png')
    # plt.savefig('models/dsb2018_96_SegNet_woDS/loss.png')
    # plt.savefig('models/dsb2018_96_TransUnet_woDS/loss.png')
    # plt.savefig('models/dsb2018_96_AttU_Net_woDS/loss.png')
    # plt.savefig('models/dsb2018_96_GRUUNet_woDS/loss.png')
    # plt.savefig('models/dsb2018_96_RUNet_woDS/loss.png')
    #D_UNet
    # plt.savefig('models/dsb2018_96_RUNet2_woDS/loss.png')
    # plt.savefig('models/dsb2018_96_D_UNet_woDS/loss.png')
    # plt.savefig('models/dsb2018_96_ResNeXtUnet18_34_woDS/loss.png')
    plt.savefig(os.path.join('models', name, 'loss.png'))

    # fig.show()
    plt.close()


