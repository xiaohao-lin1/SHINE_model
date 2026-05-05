import os

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt


def plot_loss(train_loss, valid_loss, train_acc, val_acc, save_dir, fig_name, test_loss=None, test_acc=None):
    fig = plt.figure(figsize=(10, 8))

    if train_loss:
        print(f'train_loss is {train_loss}')
        plt.plot(range(len(train_loss)), train_loss, label='Training Loss')

    if valid_loss:
        print(f'valid_loss is {valid_loss}')
        plt.plot(range(len(valid_loss)), valid_loss, label='Validation Loss')


    if test_loss:
        print(f'test_loss is {test_loss}')
        plt.plot(range(len(test_loss)), test_loss, label='Test Loss')

    if train_acc:
        print(f'train_acc is {train_acc}')
        plt.plot(range(len(train_acc)), train_acc, label='Training Acc')
    if val_acc:
        print(f'val_acc is {val_acc}')
        plt.plot(range(len(val_acc)), val_acc, label='Validation Acc')
        # find position of highest val acc
        max_val_acc = val_acc.index(max(val_acc))
        plt.axvline(max_val_acc, linestyle='--', color='r', label='Max Val Acc Epoch')


    if test_acc:
        print(f'test_acc is {test_acc}')
        plt.plot(range(len(test_acc)), test_acc, label='Test Acc')
        # find position of highest test acc
        max_test_acc = test_acc.index(max(test_acc))
        plt.axvline(max_test_acc, linestyle='--', color='b', label='Max Test Acc Epoch')
    # if s2_train_loss is not None:
    #     plt.plot(range(1, len(s2_train_loss) + 1), train_loss, label='s2_train_loss Loss')
    # if s2_train_acc is not None:
    #     plt.plot(range(1, len(s2_train_acc) + 1), train_loss, label='s2_train_acc')


    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 2.0)  # consistent scale
    plt.xlim(0, len(train_loss))  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    save_path = os.path.join(save_dir, fig_name)
    fig.savefig(f'{save_path}.png', bbox_inches='tight')


if __name__ == '__main__':
    train_loss = [0.5, 0.4, 0.3, 0.2, 0.1]
    valid_loss = []
    train_acc = [0.5, 0.6, 0.7, 0.8, 0.9]
    val_acc = []
    test_loss = [i - 0.1 for i in train_loss]
    test_acc = [i+0.1 for i in train_acc]
    save_dir = './'
    fig_name = 'test'

    plot_loss(train_loss, valid_loss, train_acc, val_acc, save_dir, fig_name, test_loss, test_acc)