import gc
import os
import sys
from contextlib import contextmanager

import torch
from torch.nn.functional import pad


# TODO: this sucks
COMFY_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))

from folder_paths import (
    models_dir,
    get_output_directory,
    get_temp_directory,
    get_save_image_path,
)


def do_cleanup(cuda_cache=True):
    gc.collect()
    if cuda_cache:
        torch.cuda.empty_cache()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def tensors_to(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    if hasattr(tensors, "__dict__"):
        return object_to(tensors, device, empty_cuda_cache=False)
    if isinstance(tensors, list):
        return [tensors_to(x, device) for x in tensors]
    if isinstance(tensors, dict):
        return {k: tensors_to(v, device) for k, v in tensors.items()}
    if isinstance(tensors, set):
        return {tensors_to(x, device) for x in tensors}
    return tensors


def tensors_to_cuda(tensors):
    return tensors_to(tensors, "cuda")


def tensors_to_cpu(tensors):
    return tensors_to(tensors, "cpu")


def object_to(obj, device=None, exclude=None, empty_cuda_cache=True, verbose=False):
    """
    recurse through an object and move any pytorch tensors/parameters/modules to the given device.
    if device is None, cpu is used by default. if the device is a CUDA device and empty_cuda_cache is
    enabled, this will also free unused CUDA memory cached by pytorch.
    """

    if not hasattr(obj, "__dict__"):
        return obj

    classname = type(obj).__name__
    exclude = exclude or set()
    device = device or "cpu"

    def _move_and_recurse(o, name=""):
        for k, v in vars(o).items():
            moved = False
            cur_name = f"{name}.{k}" if name != "" else k
            if cur_name in exclude:
                continue
            if isinstance(v, (torch.nn.Module, torch.nn.Parameter, torch.Tensor)):
                setattr(o, k, v.to(device))
                moved = True
            elif hasattr(v, "__dict__"):
                v = _move_and_recurse(v, name=cur_name)
                setattr(o, k, v)
                moved = True
            if verbose and moved: print(f"moved {classname}.{cur_name} to {device}")
        return o
    
    if isinstance(obj, torch.nn.Module):
        obj = obj.to(device)

    obj = _move_and_recurse(obj)
    if "cuda" in device and empty_cuda_cache:
        torch.cuda.empty_cache()
    return obj


@contextmanager
def obj_on_device(model, src="cpu", dst="cuda", empty_cuda_cache=True, verbose_move=False):
    model = object_to(model, dst, empty_cuda_cache=empty_cuda_cache, verbose=verbose_move)
    yield model
    model = object_to(model, src, empty_cuda_cache=empty_cuda_cache, verbose=verbose_move)


@contextmanager
def on_device(model, src="cpu", dst="cuda", empty_cuda_cache=True, **kwargs):
    model = model.to(dst)
    yield model
    model = model.to(src)
    if empty_cuda_cache:
        torch.cuda.empty_cache()


def stack_audio_tensors(tensors, mode="pad"):
    # assert all(len(x.shape) == 2 for x in tensors)
    sizes = [x.shape[-1] for x in tensors]

    if mode in {"pad_l", "pad_r", "pad"}:
        # pad input tensors to be equal length
        dst_size = max(sizes)
        stack_tensors = (
            [pad(x, pad=(0, dst_size - x.shape[-1])) for x in tensors]
            if mode == "pad_r"
            else [pad(x, pad=(dst_size - x.shape[-1], 0)) for x in tensors]
        )
    elif mode in {"trunc_l", "trunc_r", "trunc"}:
        # truncate input tensors to be equal length
        dst_size = min(sizes)
        stack_tensors = (
            [x[:, x.shape[-1] - dst_size:] for x in tensors]
            if mode == "trunc_r"
            else [x[:, :dst_size] for x in tensors]
        )
    else:
        assert False, 'unknown mode "{pad}"'

    return torch.stack(stack_tensors)
