import torch
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
import math

class Manager_Scheduler_Optimizer:
    '''
    Return the scheduler and optimizer
    '''
    def __init__(self, args, model, *vars, **kwargs):
        '''
        :param args: the args from the model_configs file
        '''
        self.args = args
        self.model = model
        self.optimizer = self.get_optimizer(args)
        self.scheduler = self.get_scheduler(args, self.optimizer)


    def get_initial_lr(self, args, *vars, **kwargs):
        if args.scheduler == 'linear_warmup_linear_decay' or args.scheduler == 'linear_warmup_cosine_decay':
            return args.warmup_lr_initial
        else:
            return args.learning_rate

    def get_optimizer(self, args,*vars, **kwargs):
        self.optimizer_dict = {
        'adam':torch.optim.Adam,
        'adamw':torch.optim.AdamW,
        }

        optimizer = self.optimizer_dict[args.optimizer]
        optimizer_args = self.get_optimizer_args(args)
        optimizer = optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), **optimizer_args)
        return optimizer
        # pass
    def get_optimizer_args(self, args, *vars, **kwargs):
        lr = self.get_initial_lr(args)
        optimizer_args_dict = {
        'adam':{'lr':lr, 'betas':(args.adam_beta[0],args.adam_beta[1]), 'weight_decay':args.adam_weight_decay},
        'adamw':{'lr':lr, 'betas':(args.adam_beta[0], args.adam_beta[1]), 'weight_decay':args.adamw_weight_decay},
        }
        return optimizer_args_dict[args.optimizer]
    def get_scheduler(self, args, optimizer, *vars, **kwargs):
        self.scheduler_dict = {
        'None':None,
        'cosine':torch.optim.lr_scheduler.CosineAnnealingLR,
        'linear_warmup_linear_decay': LambdaLR,
        'linear_warmup_cosine_decay': LambdaLR,
        'MultiStepLR':torch.optim.lr_scheduler.MultiStepLR,
        }
        scheduler = self.scheduler_dict[args.scheduler]
        scheduler_args = self.get_scheduler_args(args)
        if scheduler is not None:
            scheduler = scheduler(optimizer, **scheduler_args)

        return scheduler
    def get_scheduler_args(self, args, *vars, **kwargs):
        lr_lambda = make_lr_lambda(total_epochs=args.epochs, warmup_epochs=int(args.warmup_percent * args.epochs),
                                   lr_initial=args.warmup_lr_initial, lr_max=args.warmup_lr_max,
                                   decay_style=args.decay_style)
        self.scheduler_args_dict = {
            'None': None,
            'cosine': {'T_max':args.T_max},
            'linear_warmup_linear_decay': {'lr_lambda':lr_lambda},
            'linear_warmup_cosine_decay': {'lr_lambda':lr_lambda},
            'MultiStepLR': {'milestones': [30], 'gamma': 0.1},
        }
        return self.scheduler_args_dict[args.scheduler]



    def get(self, args, *vars, **kwargs):
        optimizer = self.get_optimizer(args)
        optimizer_args = self.get_optimizer_args(args)

        scheduler = self.get_scheduler(args)
        scheduler_args = self.get_scheduler_args(args)
        return optimizer, optimizer_args, scheduler, scheduler_args

def make_lr_lambda(total_epochs, warmup_epochs, lr_initial, lr_max, decay_style='linear'):
    # assert decay_style in ['linear', 'cosine'], "decay_style must be either 'linear' or 'cosine'"
    # define a function for the learning rate schedule
    if lr_initial == 0:
        lr_initial = 1e-7
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            lr = (lr_max / lr_initial) * (epoch / warmup_epochs)
            # print('lr', lr)
            return lr  # linear warmup
        else:
            decay_epochs = epoch - warmup_epochs
            total_decay_epochs = total_epochs - warmup_epochs + 1e-5  # avoid div by zero
            if decay_style == 'linear':
                return ((lr_max / lr_initial) *
                        (1 - decay_epochs / total_decay_epochs))  # linear decay
            elif decay_style == 'cosine':
                cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(decay_epochs / total_decay_epochs) * math.pi))
                return (lr_max / lr_initial) * cosine_decay  # cosine decay

    return lr_lambda


def test_if_make_lr_lambda_works():
    total_epochs = 100
    warmup_factor = 0.1
    warmup_epochs = total_epochs * warmup_factor  # 10% of total epochs
    lr_initial = 0.00001
    lr_max = 0.00005
    final_value = 0.00001  # just an example, you'll need to choose
    decay_style = 'cosine'
    import torch
    model = torch.nn.Linear(10, 1)
    # model = resnet50
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_initial)

    lr_lambda = make_lr_lambda(total_epochs, warmup_epochs, lr_initial, lr_max, decay_style)

    scheduler = LambdaLR(optimizer, lr_lambda)

    import matplotlib
    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lr_values = []

    for epoch in range(total_epochs):
        # Instead of old_training_code, we'll just step the scheduler and record the learning rate
        lr_values.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    print(lr_values)
    plt.plot(lr_values)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.show()


if __name__ == '__main__':
    model = nn.Linear(1,1)

    from model_configs.DCN_new import Configs
    args = Configs()
    optim_scheduler = Manager_Scheduler_Optimizer(args, model)
    optimizer = optim_scheduler.optimizer
    scheduler = optim_scheduler.scheduler

    print(optimizer)
    print(scheduler)
    test_if_make_lr_lambda_works()
    # from models.dcn import DCN

    # Assume you're old_training_code for 100 epochs

