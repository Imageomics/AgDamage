"""Runtime selection shared without coupling preprocessing to model backends."""


def select_device():
    """CUDA when this torch build can actually run on the visible GPU."""
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu")
    capability = torch.cuda.get_device_capability()
    arch = f"sm_{capability[0]}{capability[1]}"
    built = set(torch.cuda.get_arch_list())
    if built and arch not in built:
        raise RuntimeError(f"GPU needs {arch}; torch has {', '.join(sorted(built))}")
    return torch.device("cuda")
