from config import opt
import torch as t
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm, trange
from torch.nn import functional
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from PIL import Image


@t.no_grad()
def test(**kwargs):
    opt._parse(kwargs)

    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)

        probability = functional.softmax(score, dim=1)[:, 0].detach().tolist()

        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, opt.result_file)


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env, port=opt.vis_port)

    model: models.BasicModule = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    train_data = DogCat(opt.train_data_root, train=True)

    val_data = DogCat(opt.train_data_root, train=False)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_data_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    ceiterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    previous_loss = 1e10

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = ceiterion(score, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(score.detach(), target.detach())

            if (ii + 1) % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
        model.save()

        val_cm, val_accuracy = val(model, val_data_loader)

        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def mytrain(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env, port=opt.vis_port)

    model: models.BasicModule = getattr(models, opt.model)(num_classes=19)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    to_tensor = T.Compose([  # 做一点数据增强
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),  # 随机翻转
        T.ColorJitter(brightness=0.4, hue=0.3),  # 变化图片的色彩等
        T.ToTensor()
    ])
    train_data = ImageFolder(opt.train_data_root, transform=to_tensor)

    val_data = ImageFolder("./traindata/totest", transform=to_tensor)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_data_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    ceiterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(19)

    previous_loss = 1e10
    # train_model = t.nn.DataParallel(model, device_ids=[0, 1])
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = ceiterion(score, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(score.detach(), target.detach())

            if (ii + 1) % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
        model.save()

        val_cm, val_accuracy = val(model, val_data_loader)

        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


@t.no_grad()
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()  # 建模型置为测试模式  dropout层不会生效
    confusion_matrix = meter.ConfusionMeter(19)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.long())

    model.train()  # 将模型置为训练模式，所有的层都会生效
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value.trace()) / (cm_value.sum())
    return confusion_matrix, accuracy


def mytest(flie_name, **kwargs):
    vis = Visualizer(opt.env, port=opt.vis_port)
    opt._parse(kwargs)

    model = getattr(models, opt.model)(num_classes=19).eval()
    if opt.load_model_path:
        print("loading models.....")
        model.load(opt.load_model_path)
    model.to(opt.device)

    com = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )

    train_data = ImageFolder("./traindata/totrain", transform=com)
    # train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False, num_workers=opt.num_workers)
    class_dict = {v: k for k, v in train_data.class_to_idx.items()}

    for f in flie_name:
        img = Image.open(f)
        tt = com(img).unsqueeze(0)
        tt = tt.to(opt.device)

        score = model(tt)
        pre_class = class_dict[score.argmax().item()]
        vis.img("the predict classes is {}".format(pre_class), tt)
        print(pre_class)
    # acc_count = 0
    # total = 0
    # for d, l in train_dataloader:
    #     total += d.size(0)
    #     d = d.to(opt.device)
    #     l = l.to(opt.device)
    #     score: t.Tensor = model(d)
    #     pre_label = score.argmax(dim=1)
    #     result: t.Tensor = pre_label == l
    #     acc_count += result.sum().item()
    # print("total acc is {}".format((acc_count / total) * 100))


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()
