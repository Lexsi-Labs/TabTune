# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os
from abc import ABC, abstractmethod
from typing import TypeVar

import torch

T = TypeVar("T", bound="PretrainedModel")


def parse_hf_repo_id(path):
    parts = path.split("/")
    return "/".join(parts[0:2])


class PretrainedModel(ABC):
    REGISTRY: dict[str, "PretrainedModel"] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.REGISTRY[cls.register_name()] = cls

    @classmethod
    def from_pretrained(cls: type[T], path: str, device: str = "cpu", hf_kwargs=None) -> T:
        if hf_kwargs is None:
            hf_kwargs = {}
        if os.path.isdir(path):
            checkpoint_path = os.path.join(path, "model.ckpt")
        elif os.path.exists(path):
            checkpoint_path = path
        else:
            from huggingface_hub import hf_hub_download

            repo_id = parse_hf_repo_id(path)
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.ckpt", **hf_kwargs)

        # load lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model: T = cls(**checkpoint["hyper_parameters"])
        model.on_load_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = model.to(device)
        return model.eval()

    @classmethod
    @abstractmethod
    def register_name(cls) -> str:
        pass

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        pass


def load_model(path: str, device: str = "cpu", hf_kwargs=None) -> PretrainedModel:
    """Loads a TiRex model. This function attempts to load the specified model.

    Args:
        path (str): Hugging Face path to the model (e.g. NX-AI/TiRex), a local directory containing
            `model.ckpt`, or a local checkpoint file.
        device (str, optional): The device on which to load the model (e.g., "cuda:0", "cpu").
        hf_kwargs (dict, optional): Keyword arguments to pass to the Hugging Face Hub download method.

    Returns:
        PretrainedModel: The loaded model.

    Examples:
        model: ForecastModel = load_model("NX-AI/TiRex")
    """
    return PretrainedModel.REGISTRY["TiRex"].from_pretrained(path, device=device, hf_kwargs=hf_kwargs)
