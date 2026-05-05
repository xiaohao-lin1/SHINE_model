"""Subject-dependent n-fold cross-validation trainer.

The training loop is `n_fold_cross_validation`: for each subject, run
stratified n-fold CV; for each fold train the model for the full epoch
budget and report per-epoch train/val/test accuracy. The decay schedule
inside the model (Progressive Decay Graph) is driven by the current epoch.
"""
import math
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from trainers.EegDataset import EegDataset
from trainers.FolderManager import FolderManager
from trainers.Logger import Logger
from trainers.MakeDataset import MakeDataset_OpenClose
from trainers.Manager_Scheduler_Optimizer import Manager_Scheduler_Optimizer
from trainers.get_model import get_model
from utils.get_trainer import get_trainer
from utils.parser_args_utils import get_folder_name
from utils.plot_loss import plot_loss


class Trainer:
    """Subject-dependent n-fold CV trainer. Entry point: ``run()``."""

    def __init__(self, args, model_name, dataset_name, parser_args,
                 folder_name=None, trainer_name='SingleGPU_Trainer',
                 torch_dataset_object=EegDataset):
        self.args = args
        self.model_name = model_name
        self.parser_args = parser_args

        self.dataset = MakeDataset_OpenClose(
            dataset_name,
            torch_dataset_object=torch_dataset_object,
        )
        self.trainer = get_trainer(trainer_name)
        print('trainer_name', trainer_name)

        if folder_name is None:
            folder_name = get_folder_name(
                model_name, parser_args.subject_training_method,
                dataset_name, params_to_change_json=None,
            )
        self.folder_manager = FolderManager(folder_name=folder_name, results_path='./results')
        self.logger = Logger(self.folder_manager.LOGGER_FOLDER)

    def run(self):
        """Single public entry point — runs subject-dependent n-fold CV with per-epoch logging."""
        return self.n_fold_cross_validation()

    def n_fold_cross_validation(self):
        """For each subject in [start, end), run StratifiedKFold CV with per-epoch tracking."""
        if self.dataset.n_sessions != 1:
            raise NotImplementedError(
                f"n_fold_cross_validation expects single-session datasets; "
                f"got n_sessions={self.dataset.n_sessions}"
            )

        session_no = 1
        subject_accs = {}
        for subject_id in range(self.parser_args.start, self.parser_args.end):
            mean_test_acc = self._run_one_subject(subject_id, session_no, self.parser_args.nfolds)
            subject_accs[f'sub{subject_id}'] = mean_test_acc

        self.folder_manager.combine_all_csv_files_for_sub_dependent_cv(
            self.folder_manager.EXP_DATA, nfolds=self.parser_args.nfolds,
        )
        self.logger.close()
        return subject_accs

    def _run_one_subject(self, subject_id, session_no, nfolds):
        """Run nfolds StratifiedKFold CV for a single subject; return mean test accuracy."""
        self.dataset.load_one_subject_data_and_label(subject_id)

        subject_train_df = pd.DataFrame(columns=list(range(1, nfolds + 1)))
        subject_test_df = pd.DataFrame(columns=list(range(1, nfolds + 1)))

        data = deepcopy(self.dataset.data)
        labels = deepcopy(self.dataset.labels)

        # Sync schedule + variance window in samples to current epochs setting.
        if hasattr(self.args, 'T_max'):
            self.args.T_max = self.args.epochs
        if hasattr(self.args, 'variance_window_ms'):
            self.args.variance_window_samples = math.floor(self.args.variance_window_ms / 1000 * 250)
            print(f'variance_window_samples = {self.args.variance_window_samples}')
            print(f'variance_window_ms      = {self.args.variance_window_ms}')

        kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=1)
        trial_idx = np.arange(len(labels))

        for fold, (train_val_idx, test_idx) in enumerate(kf.split(trial_idx, labels)):
            self.logger.log(f"Subject {subject_id+1}, fold {fold+1} started")

            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.1, stratify=labels[train_val_idx],
            )

            train_loader, val_loader, test_loader = self._build_dataloaders(
                data, labels, train_idx, val_idx, test_idx,
            )

            model = get_model(self.model_name, args=self.args, dataset=self.dataset)
            optimizer_and_scheduler, optimizer, scheduler = self._optimizer_and_scheduler(model)

            if fold == 0:
                self.logger.log_keyword_arguments(
                    trainer=self.trainer.__name__, dataset=self.dataset,
                    folder_name=self.folder_manager.folder_name, model=model,
                    args=self.args, optimizer=optimizer, scheduler=scheduler,
                )

            trainer_args = {
                'model': model,
                'train_dataloader': train_loader,
                'val_dataloader': val_loader,
                'test_dataloader': test_loader,
                'snapshot_path': self._snapshot_path(subject_id, fold, session_no),
                'optimizer': optimizer,
                'scheduler': scheduler,
                'folder_manager': self.folder_manager,
                'logger': self.logger,
                'args': self.args,
            }
            trainer = self.trainer(**trainer_args)

            (train_results, epoch_train_loss, epoch_val_loss,
             epoch_train_acc, epoch_val_acc) = \
                trainer.train_all_epochs_with_epoch_index(fold=fold, subject_dependent=True)

            self._save_curves(subject_id, fold, epoch_train_loss, epoch_val_loss,
                              epoch_train_acc, epoch_val_acc)

            plot_loss(
                epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc,
                save_dir=self.folder_manager.LOSS_CURVES,
                fig_name=f'sub_{subject_id}_session_{session_no}_fold{fold}',
            )

            test_results = trainer.test(train=False)
            subject_train_df[fold] = train_results
            subject_test_df[fold] = test_results

        self.folder_manager.save_one_subject_data_for_sub_dependent_10fold_cv(
            subject_id, subject_train_df, subject_test_df,
            self.folder_manager.EXP_DATA, self.folder_manager.folder_name, session_no,
        )
        self.logger.log(f"Subject {subject_id+1} ended")
        return subject_test_df.loc['test_acc'].mean()

    def _build_dataloaders(self, data, labels, train_idx, val_idx, test_idx):
        train, val, test, train_y, val_y, test_y = \
            self.dataset.get_normalised_train_val_test_data_pipeline(
                data, labels, train_idx, val_idx, test_idx,
            )
        return self.dataset.make_train_val_test_dataloaders(
            self.args.batch_size, train, val, test, train_y, val_y, test_y,
        )

    def _optimizer_and_scheduler(self, model):
        manager = Manager_Scheduler_Optimizer(self.args, model)
        optimizer = manager.get_optimizer(self.args)
        scheduler = manager.get_scheduler(self.args, optimizer)
        return manager, optimizer, scheduler

    def _snapshot_path(self, subject_id, fold, session_no):
        name = f'sub_{subject_id}_fold{fold}_session_no{session_no}.pt'
        return os.path.join(self.folder_manager.MODELS_WEIGHTS, name)

    def _save_curves(self, subject_id, fold, train_loss, val_loss, train_acc, val_acc):
        for arr, name in [
            (train_acc, 'train_accs'), (val_acc, 'val_accs'),
            (train_loss, 'train_loss'), (val_loss, 'val_loss'),
        ]:
            path = os.path.join(self.folder_manager.ACC_LOSS,
                                f'{name}_sub{subject_id}_fold{fold}.npy')
            np.save(path, np.array(arr))
