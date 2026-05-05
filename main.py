"""Entry point: subject-dependent n-fold CV training of SHINE.

Usage:
    python main.py                              # default: 1 subject, defaults from configs/
    python main.py start=0 end=19               # all 19 subjects
    python main.py dataset=patients_rest_close_ica
    python main.py model.alpha=0.3 epochs=50

Hydra composes configs/{config,model/shine,dataset/...,trainer/default,optimizer/adamw}.yaml.
The training method is fixed at ``n_fold_cross_validation`` (subject-dependent
StratifiedKFold with per-epoch test logging — required for the Progressive
Decay Graph's epoch-driven decay schedule).
"""
import os
import time

import hydra
import torch
from torch.profiler import profile, ProfilerActivity
from omegaconf import DictConfig, OmegaConf

from trainers.subject_dependent_trainer import Trainer
from utils.get_trainer_name import get_trainer_name
from utils.parser_args_utils import get_folder_name, get_training_args
from utils.seed_all import seed_all


def _build_folder_name(cfg: DictConfig) -> str:
    if cfg.folder_name is not None:
        return cfg.folder_name
    # 'None' (the literal string) tells get_folder_name to skip the params suffix.
    params_json = cfg.params_to_change_json or 'None'
    return get_folder_name(
        cfg.model_name,
        cfg.subject_training_method,
        cfg.dataset_name,
        params_json,
        cfg.nfolds,
    )


def _apply_overrides(args, cfg: DictConfig):
    """Push top-level epochs override + every non-null cfg.model.* into the Configs object."""
    if cfg.epochs is not None:
        args.epochs = cfg.epochs
        print(f"Overriding epochs to {cfg.epochs}")
    for key, value in cfg.model.items():
        if key == 'name' or value is None:
            continue
        setattr(args, key, value)


def _run_with_optional_profile(trainer_instance, profile_enabled: bool):
    if not profile_enabled:
        trainer_instance.run()
        return

    print("PROFILING ENABLED — wrapping trainer_instance.run()")
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        trainer_instance.run()

    sort_keys = [
        ("self CPU time", "self_cpu_time_total"),
        ("self CUDA time", "self_cuda_time_total"),
        ("self CPU memory", "self_cpu_memory_usage"),
        ("self CUDA memory", "self_cuda_memory_usage"),
    ]
    for label, sort_by in sort_keys:
        print(f"\n=== Top 10 ops by {label} ===")
        print(prof.key_averages().table(sort_by=sort_by, row_limit=10))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    seed_all(torch_deterministic=True, seed=cfg.seed)

    args = get_training_args(cfg.model_name)
    _apply_overrides(args, cfg)

    folder_name = _build_folder_name(cfg)
    print(f"Folder name: {folder_name}")
    if cfg.wait > 0:
        print(f"Pausing for {cfg.wait} seconds to allow verification...")
        time.sleep(cfg.wait)

    trainer_instance = Trainer(
        args=args,
        model_name=cfg.model_name,
        dataset_name=cfg.dataset_name,
        parser_args=cfg,
        folder_name=folder_name,
        trainer_name=get_trainer_name(cfg.model_name),
    )

    _run_with_optional_profile(trainer_instance, cfg.profile)
    print("Resolved config:\n" + OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
