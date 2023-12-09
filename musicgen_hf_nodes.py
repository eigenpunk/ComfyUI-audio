import torch
from typing import Optional

from transformers import MusicgenForConditionalGeneration, MusicgenProcessor

from .util import do_cleanup, object_to, obj_on_device, on_device, tensors_to_cpu, tensors_to
from .musicgen_nodes import MODEL_NAMES as _ACM_MODEL_NAMES


# remove unsupported audiogen models from list
MODEL_NAMES = [x for x in _ACM_MODEL_NAMES if "audiogen" not in x]


class MusicgenHFLoader:
    def __init__(self):
        self.model = None
        self.processor = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (MODEL_NAMES,)}}

    RETURN_NAMES = ("MODEL", "PROCESSOR", "SR")
    RETURN_TYPES = ("MUSICGEN_HF_MODEL", "MUSICGEN_HF_PROC", "INT")
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self, model_name: str):
        if self.model is not None:
            self.model = object_to(self.model, empty_cuda_cache=False)
            self.processor = object_to(self.processor, empty_cuda_cache=False)
            del self.model, self.processor
            do_cleanup()
            print("MusicgenHFLoader: unloaded model")

        print(f"MusicgenHFLoader: loading {model_name}")
        model_name = "facebook/" + model_name
        self.processor = MusicgenProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        sr = self.model.config.audio_encoder.sampling_rate
        return self.model, self.processor, sr


MILLISECONDS_PER_TOKEN = 20


class MusicgenHFGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MUSICGEN_HF_MODEL",),
                "processor": ("MUSICGEN_HF_PROC",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 300.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.001}),
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
        model: MusicgenForConditionalGeneration,
        processor: MusicgenProcessor,
        text: str = "",
        batch_size: int = 1,
        duration: float = 10.0,
        cfg: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
        seed: int = 0,
        audio: Optional[torch.Tensor] = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sr = model.config.audio_encoder.sampling_rate
        # model = model.to(device)

        # empty string = unconditional generation
        if text == "":
            text = None

        max_new_tokens = int(duration * 1000.0 / MILLISECONDS_PER_TOKEN)

        with (
            torch.random.fork_rng(),
            obj_on_device(processor, dst=device, verbose_move=True) as p,
            on_device(model, dst=device) as m,
        ):
            torch.manual_seed(seed)

            # create conditioning inputs for models: using encodec for audio, t5 for text
            if audio is not None or text is not None:
                text_input = [text] * batch_size if text is not None else text
                audio_input = (
                    [x.squeeze().numpy() for x in audio] if audio is not None else audio
                )
                inputs = p(
                    text=text_input,
                    audio=audio_input,
                    sampling_rate=sr,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = m.get_unconditional_inputs(batch_size)
                cfg = inputs.guidance_scale

            # move to device, remove redundant guidance scale
            inputs = dict(inputs)
            inputs = tensors_to(inputs, device)
            inputs.pop("guidance_scale", None)

            samples = m.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                guidance_scale=cfg
            )
            inputs = tensors_to_cpu(inputs)
            del inputs

        # model = model.cpu()
        samples = samples.cpu().unsqueeze(1) if samples.dim == 2 else samples.cpu()
        do_cleanup()

        return samples,


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MusicgenHFGenerate": MusicgenHFGenerate,
    "MusicgenHFLoader": MusicgenHFLoader,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MusicgenHFGenerate": "Musicgen (HF) Generator",
    "MusicgenHFLoader": "Musicgen (HF) Loader",
}
