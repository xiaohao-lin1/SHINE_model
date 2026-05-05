"""Resolve a model name to its per-fold trainer class name."""


def get_trainer_name(model_name: str) -> str:
    """SHINE uses the single-GPU trainer; kept as a function for forward compatibility."""
    return 'SingleGPU_Trainer'
