import warnings
import torch as t


class DefaultConfig(object):
    env = 'tt'
    vis_port = 8097
    model = 'SqueezeNet'

    train_data_root = "./traindata/totrain"
    test_data_root = './data/test1'
    load_model_path = "./checkpoints/squeezenet_0120_10:42:44.pth"  # "./checkpoints/squeezenet_0119_15:20:54.pth"

    batch_size = 32
    use_gpu = True
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
