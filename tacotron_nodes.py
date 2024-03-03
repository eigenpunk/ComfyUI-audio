import json
import os
import sys
from glob import glob

import torch


base_incl_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "include")

sys.path = [
    os.path.join(base_incl_path, "hifi-gan"),
] + sys.path

from denoiser import Denoiser as HifiGANDenoiser
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator as HifiGAN

sys.path = [
    os.path.join(base_incl_path, "tacotron2"),
    os.path.join(base_incl_path, "tacotron2", "waveglow"),
] + sys.path

from hparams import create_hparams
from model import Tacotron2
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser as WaveGlowDenoiser


from .util import do_cleanup, get_device, models_dir, object_to, obj_on_device

BIGINT = 2 ** 32

MODELS_PATH = os.path.join(models_dir, "tacotron2")
WAVEGLOW_MODELS_PATH = os.path.join(models_dir, "waveglow")
HIFIGAN_MODELS_PATH = os.path.join(models_dir, "hifigan")
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(WAVEGLOW_MODELS_PATH, exist_ok=True)
os.makedirs(HIFIGAN_MODELS_PATH, exist_ok=True)

MODELS = {
    x.removeprefix(MODELS_PATH)[1:]: x
    for x in sorted(glob(os.path.join(MODELS_PATH, "*.pt")))
}
WAVEGLOW_MODELS = {
    x.removeprefix(WAVEGLOW_MODELS_PATH)[1:]: x
    for x in sorted(glob(os.path.join(WAVEGLOW_MODELS_PATH, "*")))
}
HIFIGAN_MODELS = {
    x.removeprefix(HIFIGAN_MODELS_PATH)[1:]: x
    for x in sorted(glob(os.path.join(HIFIGAN_MODELS_PATH, "*")))
}
HIFIGAN_CONFIGS = {
    os.path.basename(x): x
    for x in glob(os.path.join(base_incl_path, "hifi-gan", "config_*.json"))
}


class Tacotron2Loader:
    """
    loads a Tacotron2 model
    """
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model_name": (list(MODELS.keys()),),}
        }

    RETURN_NAMES = ("TT2_MODEL", "SR")
    RETURN_TYPES = ("TT2_MODEL", "INT")
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self, model_name):
        if self.model is not None:
            self.model = object_to(self.model, empty_cuda_cache=False)
            del self.model
            do_cleanup()
            print("Tacotron2Loader: unloaded model")

        print("Tacotron2Loader: loading model")
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        path = MODELS[model_name]

        self.model = load_model(hparams)
        sd = torch.load(path, map_location="cpu")["state_dict"]
        self.model.load_state_dict(sd)
        self.model.device = "cpu"
        self.model.eval().half()

        return self.model, 22050


class WaveGlowLoader:
    """
    loads a WaveGlow model
    """
    def __init__(self):
        self.model = None
        self.denoiser = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (list(WAVEGLOW_MODELS.keys()),),}}

    RETURN_TYPES = ("WG_MODEL",)
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self, model_name):
        if self.model is not None:
            self.model = object_to(self.model, empty_cuda_cache=False)
            self.denoiser = object_to(self.denoiser, empty_cuda_cache=False)
            del self.model, self.denoiser
            do_cleanup()
            print("WaveGlowLoader: unloaded model")

        print("WaveGlowLoader: loading model")
        path = WAVEGLOW_MODELS[model_name]

        self.model = torch.load(path, map_location="cpu")["model"]
        self.model.eval().half()
        for k in self.model.convinv:
            k.float()
        self.denoiser = WaveGlowDenoiser(self.model)

        return (self.model, self.denoiser),


class HifiGANLoader:
    """
    loads a HifiGAN model
    """
    def __init__(self):
        self.model = None
        self.denoiser = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(HIFIGAN_MODELS.keys()),),
                "config": (list(HIFIGAN_CONFIGS.keys()),),
            }
        }

    RETURN_TYPES = ("HG_MODEL",)
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self, model_name, config):
        if self.model is not None:
            self.model = object_to(self.model, empty_cuda_cache=False)
            self.denoiser = object_to(self.denoiser, empty_cuda_cache=False)
            del self.model, self.denoiser
            do_cleanup()
            print("HifiGANLoader: unloaded model")

        print("HifiGANLoader: loading model")

        with open(HIFIGAN_CONFIGS[config], "r") as f:
            cfg = AttrDict(json.load(f))

        path = HIFIGAN_MODELS[model_name]

        # model insists on choosing device itself
        device = HifiGANDenoiser.device
        self.model = HifiGAN(cfg).to(device)

        sd = torch.load(path, map_location=device)["generator"]
        self.model.load_state_dict(sd)
        self.model.eval()
        self.model.remove_weight_norm()

        self.denoiser = HifiGANDenoiser(self.model, mode="normal")

        self.model.cpu()
        self.denoiser.cpu()
        self.model.device = "cpu"
        self.denoiser.device = "cpu"

        return (self.model, self.denoiser, cfg),


class Tacotron2Generate:
    """
    generates speech mels from text using Tacotron2
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TT2_MODEL",),
                "text": ("STRING", {"default": "hello world", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_NAMES = ("mel_outputs", "postnet_outputs")
    RETURN_TYPES = ("MEL_TENSOR", "MEL_TENSOR")

    FUNCTION = "generate"

    CATEGORY = "audio"

    def generate(
        self,
        model: Tacotron2,
        text: str = "",
        seed: int = 0,
    ):
        device = get_device()

        sequence = text_to_sequence(text, ['basic_cleaners'])

        with (
            torch.no_grad(),
            torch.random.fork_rng(),
            obj_on_device(model, dst=device, verbose_move=True) as m
        ):
            prev_device = m.device
            m.device = device
            torch.manual_seed(seed)
            sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
            mel_outputs, mel_outputs_postnet, *_ = m.inference(sequence)
            m.device = prev_device

        do_cleanup()
        return mel_outputs, mel_outputs_postnet


class WaveGlowApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mels": ("MEL_TENSOR",),
                "model": ("WG_MODEL",),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "denoiser_strength": ("FLOAT", {"default": 0.06, "min": 0}),
            },
        }

    RETURN_NAMES = ("raw_audio",)
    RETURN_TYPES = ("AUDIO_TENSOR",)

    FUNCTION = "apply"

    CATEGORY = "audio"

    def apply(
        self,
        mels,
        model,
        sigma: float = 1.0,
        denoiser_strength: float = 0.06,
    ):
        device = get_device()
        waveglow, denoiser = model

        with (
            torch.no_grad(),
            torch.random.fork_rng(),
            obj_on_device(waveglow, dst=device, verbose_move=True) as wg,
            obj_on_device(denoiser, dst=device, verbose_move=True) as dn,
        ):
            prev_device = wg.device
            wg.device = dn.device = device

            mels = mels.to(device)
            audio = wg.infer(mels, sigma=sigma)
            mels.cpu()

            if denoiser_strength != 0.0:
                audio = dn(audio, denoiser_strength=denoiser_strength)
            audio = audio.cpu().unbind(0)
            wg.device = dn.device = prev_device

        do_cleanup()
        return audio,


class HifiGANApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mels": ("MEL_TENSOR",),
                "model": ("HG_MODEL",),
                "denoiser_strength": ("FLOAT", {"default": 0.06, "min": 0.0, "step": 0.001}),
            },
        }

    RETURN_NAMES = ("raw_audio",)
    RETURN_TYPES = ("AUDIO_TENSOR",)

    FUNCTION = "apply"

    CATEGORY = "audio"

    def apply(self, mels, model, denoiser_strength: float = 0.06):
        device = get_device()
        hifigan, denoiser, cfg = model

        with (
            torch.no_grad(),
            torch.random.fork_rng(),
            obj_on_device(hifigan, dst=device, verbose_move=True) as hg,
            obj_on_device(denoiser, dst=device, verbose_move=True) as dn,
        ):
            prev_device = hg.device
            hg.device = dn.device = device

            mels = mels.to(device)
            audio = hg(mels.float())
            mels.cpu()

            if denoiser_strength != 0.0:
                audio *= MAX_WAV_VALUE
                audio = dn(audio.squeeze(1), denoiser_strength) 
                audio /= MAX_WAV_VALUE

            audio = list(audio.cpu().unbind(0))
            hg.device = dn.device = prev_device

        do_cleanup()
        return audio,


class ToMelSpectrogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "sample_rate": ("INT", {"default": 22050, "min": 1, "max": BIGINT}),
                "n_fft": ("INT", {"default": 1024, "min": 1, "max": BIGINT}),
                "n_mels": ("INT", {"default": 80, "min": 1}),
                "hop_len": ("INT", {"default": 256, "min": 1, "max": BIGINT}),
                "win_len": ("INT", {"default": 1024, "min":1, "max": BIGINT}),
                "fmin": ("INT", {"default": 0, "min": 0, "max": BIGINT}),
                "fmax": ("INT", {"default": 8000, "min": 0, "max": BIGINT}),
            },
        }

    RETURN_NAMES = ("mels",)
    RETURN_TYPES = ("MEL_TENSOR",)

    FUNCTION = "apply"

    CATEGORY = "audio"

    def apply(self, audio, sample_rate: int, n_fft: int, n_mels: int, hop_len: int, win_len: int, fmin: int, fmax: int):
        with torch.no_grad():
            mels = [mel_spectrogram(clip, n_fft, n_mels, sample_rate, hop_len, win_len, fmin, fmax) for clip in audio]
            mels = torch.cat(mels)

        do_cleanup()
        return mels,


class HifiGANModelParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("HG_MODEL",)},
        }

    RETURN_NAMES = ("sr", "n_mels", "n_fft", "hop_len", "win_len", "fmin", "fmax")
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "INT")

    FUNCTION = "get"

    CATEGORY = "audio"

    def get(self, model):
        *_, cfg = model
        return cfg.sampling_rate, cfg.num_mels, cfg.n_fft, cfg.hop_size, cfg.win_size, cfg.fmin, cfg.fmax


NODE_CLASS_MAPPINGS = {
    "Tacotron2Loader": Tacotron2Loader,
    "Tacotron2Generate": Tacotron2Generate,
    "HifiGANLoader": HifiGANLoader,
    "HifiGANModelParams": HifiGANModelParams,
    "HifiGANApply": HifiGANApply,
    "WaveGlowLoader": WaveGlowLoader,
    "WaveGlowApply": WaveGlowApply,
    "ToMelSpectrogram": ToMelSpectrogram,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Tacotron2Loader": "Tacotron2 Loader",
    "Tacotron2Generate": "Tacotron2 Generator",
    "HifiGANLoader": "HifiGAN Loader",
    "HifiGANModelParams": "Get HifiGAN Model Parameters",
    "HifiGANApply": "Apply HifiGAN",
    "WaveGlowLoader": "WaveGlow Loader",
    "WaveGlowApply": "Apply WaveGlow",
    "ToMelSpectrogram": "Audio to Mel Spectrogram",
}
