import os

import torch
import torch.nn.functional as F
from tortoise.api import TextToSpeech, pick_best_batch_size_for_gpu
from tortoise.api_fast import TextToSpeech as FastTextToSpeech
from tortoise.models.cvvp import CVVP
from tortoise.utils.audio import get_voices, load_voice

from .util import do_cleanup, get_device, models_dir, object_to, obj_on_device


MODELS_PATH = os.path.join(models_dir, "tortoise")
VOICES_PATH = os.path.join(MODELS_PATH, "voices")
os.makedirs(VOICES_PATH, exist_ok=True)

VOICES = get_voices(extra_voice_dirs=[VOICES_PATH])


def _load_cvvp(self):
    from urllib.request import urlretrieve
    from tortoise.api import MODELS
    self.cvvp = CVVP(
        model_dim=512,
        transformer_heads=8,
        dropout=0,
        mel_codes=8192,
        conditioning_enc_depth=8,
        cond_mask_percentage=0,
        speech_enc_depth=8,
        speech_mask_percentage=0,
        latent_multiplier=1,
    )
    self.cvvp.eval()
    ckpt_path = os.path.join(MODELS_PATH, "cvvp.pth")
    if not os.path.exists(ckpt_path):
        urlretrieve(MODELS["cvvp.pth"], ckpt_path)
    cvvp_sd = torch.load(ckpt_path, map_location="cpu")
    self.cvvp.load_state_dict(cvvp_sd)


class TextToSpeech(TextToSpeech):
    load_cvvp = _load_cvvp


class FastTextToSpeech(FastTextToSpeech):
    load_cvvp = _load_cvvp
    def tts(
        self, text, voice_samples=None, k=1, verbose=True, use_deterministic_seed=None,
        # autoregressive generation parameters follow
        num_autoregressive_samples=512, temperature=.8, length_penalty=1, repetition_penalty=2.0, 
        top_p=.8, max_mel_tokens=500,
        # CVVP parameters follow
        cvvp_amount=.0,
        **hf_generate_kwargs,
    ):
        """function adapted from the original tortoise implementation by neonbjb."""
        self.deterministic_state(seed=use_deterministic_seed)

        text_tokens = torch.IntTensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.device)
        text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
        assert text_tokens.shape[-1] < 400, 'Too much text provided. Break the text up into separate segments and re-try inference.'
        if voice_samples is not None:
            auto_conditioning = self.get_conditioning_latents(voice_samples, return_mels=False)
        else:
            auto_conditioning  = self.get_random_conditioning_latents()
        auto_conditioning = auto_conditioning.to(self.device)

        with torch.no_grad():
            if verbose:
                print("Generating autoregressive samples..")
            with torch.autocast(
                    device_type="cuda" , dtype=torch.float16, enabled=self.half
                ):
                codes = self.autoregressive.inference_speech(
                    auto_conditioning,
                    text_tokens,
                    top_k=num_autoregressive_samples,
                    top_p=top_p,
                    temperature=temperature,
                    do_sample=True,
                    num_beams=1,
                    num_return_sequences=1,
                    length_penalty=float(length_penalty),
                    repetition_penalty=float(repetition_penalty),
                    output_attentions=False,
                    output_hidden_states=True,
                    **hf_generate_kwargs,
                )
                gpt_latents = self.autoregressive(
                    auto_conditioning.repeat(k, 1),
                    text_tokens.repeat(k, 1),
                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                    codes.repeat(k, 1),
                    torch.tensor([codes.shape[-1]*self.autoregressive.mel_length_compression], device=text_tokens.device),
                    return_latent=True,
                    clip_inputs=False
                )
            if verbose:
                print("generating audio..")
            wav_gen = self.hifi_decoder.inference(gpt_latents.to(self.device), auto_conditioning)
            return wav_gen.cpu()


class TortoiseTTSLoader:
    """
    loads the Tortoise TTS "model", which is actually just the tortoise tts api
    """
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kv_cache": ("BOOLEAN", {"default": True}),
                "half": ("BOOLEAN", {"default": False}),
                "use_deepspeed": ("BOOLEAN", {"default": False}),
                "use_fast_api": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_NAMES = ("tortoise_tts_model", "sample_rate")
    RETURN_TYPES = ("TORTOISE_TTS", "INT")
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self, kv_cache=True, half=False, use_deepspeed=False, use_fast_api=False):
        if self.model is not None:
            self.model = object_to(self.model, empty_cuda_cache=False)
            del self.model
            do_cleanup()
            print("TortoiseTTSLoader: unloaded model")

        print("TortoiseTTSLoader: loading model")
        if use_fast_api:
            print(
                "TortoiseTTSLoader: using fast api; please note that diffusion, CLVP, and CVVP controls will "
                "not be used, num_autoregressive_samples is fixed to 50, and max_mel_tokens will be ignored."
            )
        ctor = FastTextToSpeech if use_fast_api else TextToSpeech
        self.model = ctor(
            models_dir=MODELS_PATH,
            half=half,
            kv_cache=kv_cache,
            use_deepspeed=use_deepspeed,
        )

        return self.model, 24000


class TortoiseTTSGenerate:
    """
    generates speech from text using tortoise. custom voices are supported; just add short clips of speech to a
    subdirectory of "ComfyUI/models/tortoise/voices".
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TORTOISE_TTS",),
                "voice": (["random", *list(VOICES.keys())],),
                "text": ("STRING", {"default": "hello world", "multiline": True}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "num_autoregressive_samples": ("INT", {"default": 20, "min": 0, "max": 10000, "step": 1}),
                "autoregressive_batch_size": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.001, "step": 0.001}),
                "length_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "repetition_penalty": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.001, "max": 1.0, "step": 0.001}),
                "max_mel_tokens": ("INT", {"default": 500, "min": 1, "max": 600, "step": 1}),
                "cvvp_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "diffusion_steps": ("INT", {"default": 20, "min": 0, "max": 4000}),
                "cond_free": ("BOOLEAN", {"default": True}),
                "cond_free_k": ("FLOAT", {"default": 2.0, "min": 0.0, "step": 0.01}),
                "diffusion_temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio"

    def generate(
        self,
        model: TextToSpeech,
        text: str = "",
        voice: str = "random",
        batch_size: int = 1,
        num_autoregressive_samples: int = 80,
        autoregressive_batch_size: int = 8,
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
        device = get_device()
        voice_samples, voice_latents = load_voice(voice, extra_voice_dirs=[VOICES_PATH])

        if autoregressive_batch_size == 0:
            autoregressive_batch_size = pick_best_batch_size_for_gpu()

        model.autoregressive_batch_size = autoregressive_batch_size

        diffusion_kwargs = {
            "diffusion_iterations": diffusion_steps,
            "cond_free": cond_free,
            "cond_free_k": cond_free_k,
            "diffusion_temperature": diffusion_temperature,
        } if not isinstance(model, FastTextToSpeech) else {}

        with (
            torch.random.fork_rng(),
            obj_on_device(model, dst=device, exclude={"rlg_auto", "rlg_diffusion"}, verbose_move=True) as m
        ):
            prev_device = m.device
            m.device = device
            torch.manual_seed(seed)
            audio_out = m.tts(
                text,
                voice_samples=voice_samples,
                conditioning_latents=voice_latents,
                k=batch_size,
                verbose=True,
                num_autoregressive_samples=num_autoregressive_samples,
                temperature=float(temperature),
                length_penalty=float(length_penalty),
                repetition_penalty=float(repetition_penalty),
                top_p=top_p,
                max_mel_tokens=max_mel_tokens,
                cvvp_amount=cvvp_amount,
                use_deterministic_seed=seed,
                **diffusion_kwargs,
            )

            lengths = [x.shape[-1] for x in audio_out]
            max_len = max(lengths)
            audio_out = [F.pad(x, [0, max_len - x.shape[-1]]) for x in audio_out]
            audio_out = torch.cat(audio_out, dim=0)

            m.device = prev_device

        do_cleanup()
        return {"waveform": audio_out, "length": lengths, "sample_rate": 24000},


NODE_CLASS_MAPPINGS = {
    "TortoiseTTSGenerate": TortoiseTTSGenerate,
    "TortoiseTTSLoader": TortoiseTTSLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TortoiseTTSGenerate": "Tortoise TTS Generator",
    "TortoiseTTSLoader": "Tortoise TTS Loader",
}
