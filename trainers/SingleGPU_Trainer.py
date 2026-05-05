import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.profiler import record_function

from trainers.get_criterion import get_criterion
from utils.dataframe_manipulation_utils import save_results_to_pdseries
from utils.get_metrics import get_metrics


class SingleGPU_Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        snapshot_path=None,
        optimizer=None,
        scheduler=None,
        folder_manager=None,
        logger=None,
        args=None,
        device=None,
        *vars, **kwargs,
    ) -> None:
        self.epochs_run = 0
        self.args = args
        self.model = model
        self.model_name = model.__class__.__name__
        if snapshot_path is not None and os.path.exists(snapshot_path):
            print(f"snapshot exists")

        self.device = device
        self.model.cuda()

        self.train_data = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_data = test_dataloader
        self.optimizer = optimizer
        self.snapshot_path = snapshot_path
        self.scheduler = scheduler
        self.clip_grad_norm = self.args.clip_grad_norm
        self.folder_manager = folder_manager
        self.logger = logger

        if hasattr(self.args, 'loss_parameter') and self.args.loss_parameter is not None:
            self.loss = get_criterion(self.args.loss, **self.args.loss_parameter)
            self.logger.log_keyword_arguments(loss=self.args.loss, loss_parameter=self.args.loss_parameter)
        else:
            self.loss = get_criterion(self.args.loss)
            self.logger.log_keyword_arguments(loss=self.args.loss)

    def _save_snapshot(self, epoch):
        if getattr(self.args, 'no_store_wt', False):
            return
        snapshot = {"MODEL_STATE": self.model.state_dict(), "EPOCHS_RUN": epoch}
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train_one_epoch(self, *vars, **kwargs):
        epoch_train_loss = 0
        all_preds, all_labels = [], []
        self.model.train()
        for i, data in enumerate(self.train_data):
            with record_function("data_loading"):
                inputs, labels = data
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            self.optimizer.zero_grad()
            with record_function("forward_pass"):
                outputs = self.model(inputs, *vars, **kwargs)
            if isinstance(outputs, (tuple, list)):
                loss = self.loss(outputs[0], labels)
                _, predictions = torch.max(outputs[0], 1)
            else:
                loss = self.loss(outputs, labels)
                _, predictions = torch.max(outputs, 1)
            loss.backward()
            epoch_train_loss += loss.item()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad_max_norm)
            self.optimizer.step()

            all_labels.append(labels.detach())
            all_preds.append(predictions.detach())

        epoch_train_loss /= len(self.train_data)
        preds = torch.cat(all_preds, 0)
        label_ls = torch.cat(all_labels, 0)
        acc, f1, cm = get_metrics(y_pred=preds.tolist(), y_true=label_ls.tolist())
        return epoch_train_loss, acc, f1, cm

    def train_all_epochs_with_epoch_index(self, fold, subject_dependent=False):
        """Train + validate for the full epoch budget, passing the epoch index
        to the model (used by SHINE's PDG layer-wise decay schedule)."""
        torch.manual_seed(self.args.seed)
        epoch_train_loss_ls, epoch_val_loss_ls, epoch_train_accs, epoch_val_accs, lrs = [], [], [], [], []

        best_val_acc = -np.inf
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        last_epoch = max(self.args.epochs, self.epochs_run + 1) - 1

        for epoch in range(self.epochs_run, max(self.args.epochs, self.epochs_run + 1)):
            train_epoch_loss, train_acc, train_f1, train_cm = self.train_one_epoch(epoch, train=True)
            if self.scheduler is not None:
                self.scheduler.step()
            lrs.append(self.optimizer.param_groups[0]["lr"])

            epoch_train_loss_ls.append(train_epoch_loss)
            epoch_train_accs.append(train_acc)

            if self.val_dataloader is None:
                ls_results = save_results_to_pdseries(
                    model_name=self.model_name, epoch=epoch, train_acc=train_acc,
                    train_f1=train_f1, train_cm=train_cm, params=params,
                )
            else:
                val_epoch_loss, val_acc, val_f1, val_cm = self.validate(train=False)
                epoch_val_loss_ls.append(val_epoch_loss)
                epoch_val_accs.append(val_acc)

                if val_acc > best_val_acc:
                    print(f'Epoch {epoch}, val_acc is {val_acc}')
                    best_val_acc = val_acc
                    ls_results = save_results_to_pdseries(
                        model_name=self.model_name, epoch=epoch, train_acc=train_acc,
                        train_f1=train_f1, train_cm=train_cm, val_acc=val_acc,
                        val_f1=val_f1, val_cm=val_cm, params=params,
                    )

        self._save_snapshot(last_epoch)
        self.plot_learning_rate_curve(lrs)

        return ls_results, epoch_train_loss_ls, epoch_val_loss_ls, epoch_train_accs, epoch_val_accs

    def validate(self, *vars, **kwargs):
        epoch_val_loss = 0
        all_preds, all_labels = [], []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_dataloader):
                with record_function("validation_data_loading"):
                    inputs = data[0].type(torch.float).cuda(non_blocking=True)
                    labels = data[1].type(torch.LongTensor).cuda(non_blocking=True)
                with record_function("validation_forward"):
                    outputs = self.model(inputs, *vars, **kwargs)
                if isinstance(outputs, (tuple, list)):
                    loss = self.loss(outputs[0], labels)
                    _, predictions = torch.max(outputs[0], 1)
                else:
                    loss = self.loss(outputs, labels)
                    _, predictions = torch.max(outputs, 1)

                epoch_val_loss += loss.item()
                all_labels.append(labels.detach())
                all_preds.append(predictions.detach())

            epoch_val_loss /= len(self.val_dataloader)
            preds = torch.cat(all_preds, 0)
            label_ls = torch.cat(all_labels, 0)
            acc, f1, cm = get_metrics(y_pred=preds.tolist(), y_true=label_ls.tolist())
        return epoch_val_loss, acc, f1, cm

    def test(self, *vars, **kwargs):
        label_ls, preds = [], []
        snapshot = torch.load(self.snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Testing starts, loading model saved at {self.snapshot_path} at Epoch {self.epochs_run}")

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                inputs = data[0].type(torch.float).cuda(non_blocking=True)
                labels = data[1].type(torch.LongTensor).cuda(non_blocking=True)

                outputs = self.model(inputs, *vars, **kwargs)
                if isinstance(outputs, (tuple, list)):
                    _, predictions = torch.max(outputs[0], 1)
                else:
                    _, predictions = torch.max(outputs, 1)
                label_ls.extend(labels.data.tolist())
                preds.extend(predictions.data.tolist())

            acc, f1, cm = get_metrics(y_pred=preds, y_true=label_ls)
            test_results = save_results_to_pdseries(test_acc=acc, test_f1=f1, test_cm=cm)
        return test_results

    def plot_learning_rate_curve(self, lrs, name='lr'):
        fig, axs = plt.subplots(1)
        axs.plot(lrs)
        fig.savefig(os.path.join(self.folder_manager.LOSS_CURVES, f'{name}.png'))
