import torch as t
import torchvision as tv
import torchnet as tnt

from torch.utils import data
from transformer_net import TransformerNet
import util
from PackedVGG import Vgg16
from torch.nn import functional as F
import tqdm
import os
import ipdb

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Config(object):
    image_size = 256  # 图片大小
    batch_size = 8
    data_root = 'data/'  # 数据集存放路径：data/coco/a.jpg
    num_workers = 4  # 多线程加载数据
    use_gpu = True  # 使用GPU

    style_path = 'style.jpg'  # 风格图片存放路径
    lr = 1e-3  # 学习率

    env = 'neural-style'  # visdom env
    plot_every = 10  # 每10个batch可视化一次

    epoches = 2  # 训练epoch

    content_weight = 1e5  # content_loss 的权重
    style_weight = 1e10  # style_loss的权重

    model_path = None  # 预训练模型的路径
    debug_file = '/tmp/debugnn'  # touch $debug_fie 进入调试模式

    content_path = 'input.png'  # 需要进行分割迁移的图片
    result_path = 'output.png'  # 风格迁移结果的保存路径


def train(**kwargs):
    opt = Config()
    for k_,v_ in kwargs.items():
        setattr(opt,k_,v_)

    device = t.device('cuda') if opt.use_gpu else t.device("cpu")
    vis = util.Visualizer(opt.env)

    transforms = tv.transforms.Compose({
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x:x*255)
    })

    dataset = tv.datasets.ImageFolder(opt.data_root,transforms)
    dataLoader = data.DataLoader(dataset,opt.batch_size)

    transform = TransformerNet()

    if opt.model_path:
        transform.load_state_dict(t.load(opt.model_path,map_location=lambda _s,_:_s))
    transform.to(device)

    vgg = Vgg16().eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    optimizer = t.optim.Adam(transform.parameters(),opt.lr)

