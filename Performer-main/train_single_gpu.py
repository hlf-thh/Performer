import os
import math
import argparse
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataSet
from multi_train_utils import train_one_epoch, evaluate
from utils import read_split_data, plot_data_loader_image
from Performer import Performer_small


def TC_calc_lr(epoch, args):
    max_epoch = args.epochs
    lr = args.lrf + 0.5 * (args.lr - args.lrf) * (1 + math.cos(math.pi * (epoch) / max_epoch))
    return lr


def calc_lr(epoch, args):
    max_epoch = args.epochs
    a = max_epoch/4
    b = max_epoch/2
    c = max_epoch*3/4
    d = max_epoch
    ret1 = args.lrf + 0.5 * (args.lr - args.lrf) * (1 + math.cos(math.pi * a / d))
    ret2 = args.lrf + 0.5 * (ret1 - args.lrf) * (1 + math.cos(math.pi * a / c))
    ret3 = args.lrf + 0.5 * (ret2 - args.lrf) * (1 + math.cos(math.pi * a / b))
    if epoch < a:
        lr = args.lrf + 0.5 * (args.lr - args.lrf) * (1 + math.cos(math.pi * (epoch) / d))
    elif epoch < b:
        lr = args.lrf + 0.5 * (ret1 - args.lrf) * (1 + math.cos(math.pi * (epoch-a) / c))
    elif epoch < c:
        lr = args.lrf + 0.5 * (ret2 - args.lrf) * (1 + math.cos(math.pi * (epoch-b) / b))
    else:
        lr = args.lrf + 0.5 * (ret3 - args.lrf) * (1 + math.cos(math.pi * (epoch-c) / a))
    return lr


def N_calc_lr(epoch, args):
    lr = args.lr
    lrf = args.lrf
    max_epoch = args.epochs
    max = max_epoch / 4
    a = max_epoch/4 - 1
    b = max_epoch/2 - 1
    c = max_epoch * 3 / 4 - 1
    d = max_epoch - 1
    ret1 = lrf + 0.5 * (lr - lrf) * (1 + math.cos(math.pi * a / max_epoch))
    ret2 = lrf + 0.5 * (lr - lrf) * (1 + math.cos(math.pi * b / max_epoch))
    ret3 = lrf + 0.5 * (lr - lrf) * (1 + math.cos(math.pi * c / max_epoch))
    if epoch <= a:
        LR = ret1 + 0.5 * (lr - ret1) * (1 + math.cos(math.pi * epoch / max))
    elif epoch <= b:
        LR = ret2 + 0.5 * (ret1 - ret2) * (1 + math.cos(math.pi * (epoch - a) / max))
    elif epoch <= c:
        LR = ret3 + 0.5 * (ret2 - ret3) * (1 + math.cos(math.pi * (epoch - b) / max))
    else:
        LR = lrf + 0.5 * (ret3 - lrf) * (1 + math.cos(math.pi * (epoch - c) / max))
    return LR


def P_calc_lr(epoch, args):
    lr = args.lr
    lrf = args.lrf
    max_epoch = args.epochs
    max = max_epoch / 4
    a = max_epoch/4 - 1
    b = max_epoch/2 - 1
    c = max_epoch * 3 / 4 - 1
    d = max_epoch - 1
    ret1 = lr * 0.75
    ret2 = lr * 0.5
    ret3 = lr * 0.25
    if epoch <= a:
        LR = ret1 + 0.5 * (lr - ret1) * (1 + math.cos(math.pi * epoch / max))
    elif epoch <= b:
        LR = ret2 + 0.5 * (ret1 - ret2) * (1 + math.cos(math.pi * (epoch - a) / max))
    elif epoch <= c:
        LR = ret3 + 0.5 * (ret2 - ret3) * (1 + math.cos(math.pi * (epoch - b) / max))
    else:
        LR = lrf + 0.5 * (ret3 - lrf) * (1 + math.cos(math.pi * (epoch - c) / max))
    return LR

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform['val'])

    batch_size = args.batch_size

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)


    # # ladnet50
    # model = lad38(num_classes=200, p=args.p).to(device)


    # mixnet3
    model = Performer_small(num_classes=args.num_classes, p=args.p).to(device)

    # ### acmix
    # from swin_transformer_acmix import SwinTransformer_acmix
    # model = SwinTransformer_acmix(depths=[2,2,6,2], drop_path_rate=0.3, num_classes=200)
    # model.to(device)
    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params=pg, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.lrf)



    # ### conformer
    # from conformer import Conformer_tiny_patch16
    # model = Conformer_tiny_patch16(num_classes=args.num_classes,drop_rate=args.p,
    #                                attn_drop_rate=args.p, drop_path_rate=args.p)
    # model.to(device)
    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(params=pg, lr=args.lr, eps=1e-8, weight_decay=0.05)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.lrf)


    # ### VIT
    # model = torchvision.models.vit_b_16(pretrained=False, dropout=0.1)
    # model.heads = nn.Linear(768, args.num_classes)
    # model.to(device)
    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(params=pg, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.lrf)


    # ### ResNet50
    # model = torchvision.models.resnet50(pretrained=False)
    # model.heads = nn.Linear(2048, args.num_classes)
    # model.to(device)

    # ### efficientnet_b2
    # model = torchvision.models.efficientnet_b2(Weight=None)
    # model.classifier = nn.Sequential(
    #     nn.Dropout(p=0.2, inplace=True),
    #     nn.Linear(1408, args.num_classes),
    # )
    # model.to(device)


    # # 如果存在预训练权重则载入
    # if args.weights != "":
    #     if os.path.exists(args.weights):
    #         weights_dict = torch.load(args.weights, map_location=device)
    #         load_weights_dict = {k: v for k, v in weights_dict.items()
    #                              if model.state_dict()[k].numel() == v.numel()}
    #         print(model.load_state_dict(load_weights_dict, strict=False))
    #     else:
    #         raise FileNotFoundError("not found weights file: {}".format(args.weights))
    #
    # # 是否冻结权重
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除最后的全连接层外，其他权重全部冻结
    #         if "fc" not in name:
    #             para.requires_grad_(False)
    #
    # 选用优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    if args.op == "SGD":
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.op == "RAdam":
        optimizer = optim.RAdam(params=pg, lr=args.lr, weight_decay=args.weight_decay)
    if args.op == "AdamW":
        optimizer = optim.AdamW(params=pg, lr=args.lr, weight_decay=args.weight_decay)
    if args.op == "AdAm":
        optimizer = optim.Adam(params=pg, lr=args.lr, weight_decay=args.weight_decay)



    # 选用学习率
    if args.lr_scheduler[0] == 'Cosine':
        Calc = TC_calc_lr

    if args.lr_scheduler[0] == 'Step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    if args.lr_scheduler[0] == 'Exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_scheduler[1])

    if args.lr_scheduler[0] == 'ours':
        Calc = calc_lr

    if args.lr_scheduler[0] == 'news':
        Calc = N_calc_lr

    if args.lr_scheduler[0] == 'pro':
        Calc = P_calc_lr
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf



    file = open(r"./performer/performer_small_op={}_lr_scheduler={}_gamma={}_p={}_lr={}_epoch{}_best_model.txt".
                format(args.op, args.lr_scheduler[0],args.lr_scheduler[1], args.p, args.lr, args.epochs), "w", encoding='utf-8')
    file.write('{}\n'.format(str(args)))


    best_acc = 0.0
    best_epoch = 0
    ACC=[]

    for epoch in range(args.epochs):
        lr = Calc(epoch, args)
        optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=args.weight_decay)
        train_loss, train_acc = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,
                                    warmup=True)

        # scheduler.step()

        # validate
        test_loss, acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)
        print("[epoch {}] test_loss {} accuracy: {}".format(epoch, round(test_loss, 3), round(acc, 3)))
        file.write("[epoch {}], train_loss: {}, train_acc: {}, test_loss: {},test_acc: {}\n".
                   format(epoch, round(train_loss, 4), round(train_acc, 4), round(test_loss, 4), round(acc, 4)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        ACC.append(acc)
        if best_acc < acc:
            torch.save(model.state_dict(), "./weights/performer/performer_epoch400.pth")
            best_acc = acc
            best_epoch = epoch
    file.write("[best_epoch {}], best_acc: {}".format(best_epoch, round(best_acc, 4)))
    file.close()
    print(best_acc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--op', type=str, default='SGD')
    parser.add_argument('--lr_scheduler', type=list, default=['Cosine', 4])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=4e-5)
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="/media/ubuntu/D/xxxx/data/imagenet-200")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
