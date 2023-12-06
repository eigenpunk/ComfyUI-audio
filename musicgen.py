import os
from glob import glob
from typing import Optional

import torch
from audiocraft.models import MusicGen


MODEL_NAMES = [
    "musicgen-small",
    "musicgen-medium",
    "musicgen-melody",
    "musicgen-large",
    "musicgen-melody-large",
    # TODO: stereo models seem not to be working out of the box
    # "musicgen-stereo-small",
    # "musicgen-stereo-medium",
    # "musicgen-stereo-melody",
    # "musicgen-stereo-large",
    # "musicgen-stereo-melody-large",
]


class MusicgenLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (MODEL_NAMES,)}}

    RETURN_NAMES = ("MODEL", "SR")
    RETURN_TYPES = ("MUSICGEN_MODEL", "INT")
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self, model_name):
        print(f"MusicgenLoader: loading {model_name}")

        model_name = "facebook/" + model_name
        model = MusicGen.get_pretrained(model_name)
        sr = model.sample_rate
        return model, sr


class MusicgenGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MUSICGEN_MODEL",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 300, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "top_k": ("INT", {"default": 250, "min": 0, "max": 10000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {"audio": ("AUDIO_TENSOR",)},
        }

    RETURN_NAMES = ("RAW_AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)

    FUNCTION = "generate"

    CATEGORY = "audio"

    def generate(
        self,
        model: MusicGen,
        text: str = "",
        batch_size: int = 1,
        duration: int = 10,
        cfg: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        seed: int = 0,
        audio: Optional[torch.Tensor] = None,
    ):
        model.set_generation_params(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            duration=duration,
            cfg_coef=cfg,
        )
        if torch.cuda.is_available():
            model.lm = model.lm.cuda()
            model.compression_model = model.compression_model.cuda()
        if text == "":
            text = None
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            if audio is not None or text is not None:
                batched_input = [text] * batch_size
                audio_out = (
                    model.generate_continuation(
                        audio, model.sample_rate, batched_input, progress=True
                    )
                    if audio is not None
                    else model.generate(batched_input, progress=True)
                )
            else:
                audio_out = model.generate_unconditional(batch_size, progress=True)
        model.lm = model.lm.cpu()
        model.compression_model = model.compression_model.cpu()

        return audio_out.cpu(),


NODE_CLASS_MAPPINGS = {
    "MusicgenGenerate": MusicgenGenerate,
    "MusicgenLoader": MusicgenLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusicgenGenerate": "Musicgen Generator",
    "MusicgenLoader": "Musicgen Loader",
}
