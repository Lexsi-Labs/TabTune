from ..model import loading as _v2_loading
from ..version import resolve_arch_line, resolve_arch_version


def _check_v2(ckpt):
    if resolve_arch_line(resolve_arch_version(ckpt)) != "v2_0":
        raise ValueError(
            "This checkpoint is a LimiX v1 model; load it with model_name='Limix' instead of 'LimiXv2'."
        )


def load_model(model_path, mask_prediction: bool = False, deterministic: bool = False):
    import torch
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    return load_from_checkpoint(ckpt, mask_prediction=mask_prediction, deterministic=deterministic)


def load_from_checkpoint(ckpt, mask_prediction: bool = False, deterministic: bool = False):
    _check_v2(ckpt)
    return _v2_loading.load_from_checkpoint(
        ckpt, mask_prediction=mask_prediction, deterministic=deterministic
    )
