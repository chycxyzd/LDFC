import torch.nn as nn
from torchvision import transforms, datasets
import json
import argparse
import os
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from Adan import Adan

from resnet50 import resnet50
from train_utils import get_params_groups, create_lr_scheduler, train_one_epoch, evaluate
from Kappa import confusion_matrix

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    # device = torch.cuda.set_device(1)
    print(device)
    tb_writer = SummaryWriter()

    data_transform = {

        "train": transforms.Compose([transforms.Resize((448, 448)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation((0, 360)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]),
        "val": transforms.Compose([transforms.Resize((448, 448)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])}

    image_path = args.data_path
    train_dataset = datasets.ImageFolder(root=image_path + "train",
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=image_path + "test",            # test_less
                                       transform=data_transform["val"])

    train_num = len(train_dataset)
    batch_size = args.batch_size
    nw = 8
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model_name = "ConvNext"
    model = resnet50(num_classes=3)
    #model = torch.nn.DataParallel(model)
    model.to(device)

    pg = get_params_groups(model, weight_decay=args.wd)
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd, momentum=0.9)
    optimizer = Adan(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=10)

    best_acc = 0.
    best_train_acc = 0.
    best_kappa = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc, train_kappa1, train_truee, train_predd = train_one_epoch(model=model,
                                                                                        optimizer=optimizer,
                                                                                        data_loader=train_loader,
                                                                                        device=device,
                                                                                        epoch=epoch,
                                                                                        lr_scheduler=lr_scheduler)
        print("train_kappa= %.4f" % train_kappa1)

        if train_acc > best_train_acc:
            conf_matrix = confusion_matrix(train_truee, train_predd)
            # print(f"Confusion matrix: {conf_matrix}")
            data = {'True label': train_truee,
                    'Predict label': train_predd}
            df = pd.DataFrame(data, columns=['True label', 'Predict label'])
            confmtpd = pd.crosstab(df['True label'], df['Predict label'], dropna=False)
            print(f"Confusion matrix with pandas: {confmtpd}")
            cfmfig = plt.figure()
            sn.heatmap(confmtpd, annot=True, cmap='Greens', fmt='d')


            best_train_acc = train_acc

        # validate
        val_loss, val_acc, val_kappa1, val_truee, val_predd = evaluate(model=model,
                                                                       data_loader=val_loader,
                                                                       device=device,
                                                                       epoch=epoch)
        print("val_kappa= %.4f" % val_kappa1)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "train_kappa", "val_kappa"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], train_kappa1, epoch)
        tb_writer.add_scalar(tags[6], val_kappa1, epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "/home/Chen_hongyu/cancer_classification_dataset/zhr_classfication_result/cancer_resnet50.pth")  # /home/Chen_hongyu/ThreeStage_Model/ThreeStage_Model/RepLKNet_test/cancer_classification/cancer_resnet50.pth

            # Constructing a confusion matrix
            val_conf_matrix = confusion_matrix(val_truee, val_predd)
            # print(f"Confusion matrix: {val_conf_matrix}")
            val_data = {'True label': val_truee,
                        'Predict label': val_predd}
            val_df = pd.DataFrame(val_data, columns=['True label', 'Predict label'])
            val_confmtpd = pd.crosstab(val_df['True label'], val_df['Predict label'], dropna=False)
            print(f"Confusion matrix with pandas: {val_confmtpd}")

            val_cfmfig = plt.figure()
            sn.heatmap(val_confmtpd, annot=True, cmap='Greens', fmt='d')
            val_cfmfig.savefig(f'/home/Chen_hongyu/cancer_classification_dataset/zhr_classfication_result/cancer_resnet50.png')  # /home/Chen_hongyu/ThreeStage_Model/ThreeStage_Model/RepLKNet_test/cancer_classification/cancer_resnet50.png

            best_acc = val_acc

        if best_kappa < val_kappa1:
            best_kappa = val_kappa1

        print("best_train_acc = %.4f" % best_train_acc)
        print("best_val_acc = %.4f" % best_acc)
        print("best_val_kappa = %.4f" % best_kappa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=0.05)
    parser.add_argument('--data-path', type=str,
                        default=r"/home/Chen_hongyu/cancer_classification_dataset/augment_BMP/")
    parser.add_argument('--weights', type=str,
                        default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)