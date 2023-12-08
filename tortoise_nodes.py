import gc
import os

import torch

from .util import do_cleanup, models_dir, move_object_tensors_to_device, obj_on_device

MODELS_PATH = os.path.join(models_dir, "tortoise")
VOICES_PATH = os.path.join(MODELS_PATH, "voices")

os.makedirs(VOICES_PATH, exist_ok=True)

from tortoise import api, api_fast
from tortoise.utils.audio import get_voices, load_voice


VOICES = get_voices(extra_voice_dirs=[VOICES_PATH])


class TortoiseTTSLoader:
    """
    loads the Tortoise TTS "model", which is actually just the tortoise tts api

    TODO: figure out why deepspeed no work
    """
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kv_cache": ("BOOLEAN", {"default": True}),
                # "use_deepspeed": ("BOOLEAN", {"default": True}),
                "half": ("BOOLEAN", {"default": True}),
                "use_fast_api": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_NAMES = ("MODEL", "SR")
    RETURN_TYPES = ("TORTOISE_TTS",)
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self, kv_cache=False, half=False, use_fast_api=False):
        if self.model is not None:
            self.model = move_object_tensors_to_device(self.model, empty_cuda_cache=False)
            del self.model
            do_cleanup()
            print("TortoiseTTSLoader: unloaded model")

        print("TortoiseTTSLoader: loading model")
        kwargs = {"kv_cache": kv_cache, "use_deepspeed": False, "half": half}
        self.model = (
            api_fast.TextToSpeech(**kwargs, models_dir=MODELS_PATH)
            if use_fast_api
            else api.TextToSpeech(**kwargs, models_dir=MODELS_PATH)
        )

        return self.model,


class TortoiseTTSGenerate:
    """
    generates speech from text using tortoise. custom voices are supported; just add short clips of speech to a
    subdirectory of "ComfyUI/models/tortoise/voices".
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TORTOISE_TTS",),
                "voice": (["random", *list(VOICES.keys())],),
                "text": ("STRING", {"default": "hello world", "multiline": True}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "num_autoregressive_samples": ("INT", {"default": 80, "min": 0, "max": 10000, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.001}),
                "length_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "top_p": ("FLOAT", {"default": 0.001, "min": 0.001, "max": 1.0, "step": 0.001}),
                "max_mel_tokens": ("INT", {"default": 500, "min": 1, "max": 600, "step": 1}),
                "cvvp_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "diffusion_steps": ("INT", {"default": 100, "min": 0, "max": 4000}),
                "cond_free": ("BOOLEAN", {"default": True}),
                "cond_free_k": ("FLOAT", {"default": 2.0, "min": 0.0}),
                "diffusion_temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_NAMES = ("RAW_AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)

    FUNCTION = "generate"

    CATEGORY = "audio"

    def generate(
        self,
        model: api.TextToSpeech | api_fast.TextToSpeech,
        text: str = "",
        voice: str = "random",
        batch_size: int = 1,
        num_autoregressive_samples: int = 80,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        repetition_penalty: float = 2.0,
        top_p: float = 1.0,
        max_mel_tokens: int = 500,
        cvvp_amount: float = 0.0,
        diffusion_steps: int = 100,
        cond_free: bool = False,
        cond_free_k: float = 0.0,
        diffusion_temperature: float = 1.0,
        seed: int = 0,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        voice_samples, voice_latents = load_voice(voice, extra_voice_dirs=[VOICES_PATH])
        kwargs = dict(
            voice_samples=voice_samples,
            conditioning_latents=voice_latents,
            k=batch_size,
            verbose=True,
            num_autoregressive_samples=num_autoregressive_samples,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            max_mel_tokens=max_mel_tokens,
            cvvp_amount=cvvp_amount,
            diffusion_iterations=diffusion_steps,
            cond_free=cond_free,
            cond_free_k=cond_free_k,
            diffusion_temperature=diffusion_temperature,
            use_deterministic_seed=seed,
        )

        # fast api excludes diffusion parameters
        if isinstance(model, api_fast.TextToSpeech):
            for k in ["diffusion_iterations", "cond_free", "cond_free_k", "diffusion_temperature"]:
                kwargs.pop(k)

        with torch.random.fork_rng(), obj_on_device(model, dst=device) as m:
            torch.manual_seed(seed)
            audio_out = m.tts(text, **kwargs)
            if batch_size > 1:
                audio_out = [x.squeeze(0).cpu() for x in audio_out]
            else:
                audio_out = [audio_out.squeeze(0).cpu()]

        do_cleanup()
        return audio_out,


NODE_CLASS_MAPPINGS = {
    "TortoiseTTSGenerate": TortoiseTTSGenerate,
    "TortoiseTTSLoader": TortoiseTTSLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TortoiseTTSGenerate": "Tortoise TTS Generator",
    "TortoiseTTSLoader": "Tortoise TTS Loader",
}