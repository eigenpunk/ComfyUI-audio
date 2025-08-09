from typing import Optional, Union

import torch
from audiocraft.models import AudioGen, MusicGen

from .util import do_cleanup, object_to, obj_on_device, tensors_to, tensors_to_cpu


MODEL_NAMES = [
    "musicgen-small",
    "musicgen-medium",
    "musicgen-melody",
    "musicgen-large",
    "musicgen-melody-large",
    "musicgen-stereo-small",
    "musicgen-stereo-medium",
    "musicgen-stereo-melody",
    "musicgen-stereo-large",
    "musicgen-stereo-melody-large",
    "audiogen-medium",
]


class MusicgenLoader:
    def __init__(self):
        self.model = None
        self.name = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (MODEL_NAMES,)}}

    RETURN_NAMES = ("musicgen_model", "sample_rate")
    RETURN_TYPES = ("MUSICGEN_MODEL", "INT")
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self, model_name: str):
        self.unload()

        print(f"MusicgenLoader: loading {model_name}")

        self.name = "facebook/" + model_name
        model_class = AudioGen if "audiogen" in self.name else MusicGen

        self.model = model_class.get_pretrained(self.name)
        sr = self.model.sample_rate
        return self.model, sr

    def unload(self):
        if self.model is not None:
            # force move to cpu, delete/collect, clear cache
            self.model = object_to(self.model, empty_cuda_cache=False)
            del self.model
            do_cleanup()
            print("MusicgenLoader: unloaded model")


class MusicgenGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MUSICGEN_MODEL",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 300.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "top_k": ("INT", {"default": 250, "min": 0, "max": 10000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio"

    def generate(
        self,
        model: Union[AudioGen, MusicGen],
        text: str = "",
        batch_size: int = 1,
        duration: float = 10.0,
        cfg: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        seed: int = 0,
        audio = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # empty string = unconditional generation
        if text == "":
            text = None

        model.set_generation_params(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            duration=duration,
            cfg_coef=cfg,
        )
        with torch.random.fork_rng(), obj_on_device(model, dst=device, verbose_move=True) as m:
            torch.manual_seed(seed)
            text_input = [text] * batch_size
            if audio is not None:
                # do continuation with input audio and (optional) text prompting
                audio_in = audio["waveform"]

                if audio_in.shape[0] < batch_size:
                    # (try to) expand batch if smaller than requested
                    audio_in = audio_in.expand(batch_size, -1, -1)
                elif audio_in.shape[0] > batch_size:
                    # truncate batch if larger than requested
                    audio_in = audio_in[:batch_size]

                audio_input = tensors_to(audio_in, device)
                audio_out = m.generate_continuation(audio_input, model.sample_rate, text_input, progress=True)
            elif text is not None:
                # do text-to-music
                audio_out = m.generate(text_input, progress=True)
            else:
                # do unconditional music generation
                audio_out = m.generate_unconditional(batch_size, progress=True)

            audio_out = tensors_to_cpu(audio_out)

        do_cleanup()
        return {"waveform": audio_out, "sample_rate": model.sample_rate},


NODE_CLASS_MAPPINGS = {
    "MusicgenGenerate": MusicgenGenerate,
    "MusicgenLoader": MusicgenLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusicgenGenerate": "Musicgen Generator",
    "MusicgenLoader": "Musicgen Loader",
}
