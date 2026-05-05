"""Per-fold trainer factory.

Only the single-GPU trainer is supported in the public release. Parameter
kept as a string to mirror the resolution flow in ``utils.get_trainer_name``.
"""
from trainers.SingleGPU_Trainer import SingleGPU_Trainer


_TRAINERS = {
    'SingleGPU_Trainer': SingleGPU_Trainer,
}


def get_trainer(trainer_name: str = 'SingleGPU_Trainer'):
    return _TRAINERS[trainer_name]
