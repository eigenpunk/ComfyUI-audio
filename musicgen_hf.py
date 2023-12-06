import torch

from transformers import MusicgenForConditionalGeneration, MusicgenProcessor

from .musicgen import MODEL_NAMES


class MusicgenHFLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (MODEL_NAMES,),
            },
        }

    RETURN_NAMES = (
        "MODEL",
        "PROCESSOR",
        "SR",
    )
    RETURN_TYPES = (
        "MUSICGEN_HF_MODEL",
        "MUSICGEN_HF_PROC",
        "INT",
    )

    FUNCTION = "load"

    CATEGORY = "audio"

    def load(self, model_name):
        print(f"MusicgenHFLoader: loading {model_name}")
        model_name = "facebook/" + model_name
        processor = MusicgenProcessor.from_pretrained(model_name)
        musicgen = MusicgenForConditionalGeneration.from_pretrained(model_name)
        return musicgen, processor, musicgen.config.audio_encoder.sampling_rate


def tensor_dict_to_cuda(tensors):
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}


def tensor_dict_to_cpu(tensors):
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}


class MusicgenHFGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MUSICGEN_HF_MODEL",),
                "processor": ("MUSICGEN_HF_PROC",),
                "text": ("STRING", {"multiline": True, "default": "rock and roll song about how rap is back"}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "max_new_tokens": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 10.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_NAMES = ("RAW_AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)

    FUNCTION = "generate"

    CATEGORY = "audio"

    def generate(self, model, processor, text, batch_size, max_new_tokens, cfg, top_k, top_p, temperature, seed):
        model = model.cuda()
        conditioning = (
            processor(text=[text] * batch_size, padding=True, return_tensors="pt")
            if text != ""
            else model.get_unconditional_input(batch_size)
        )
        conditioning = tensor_dict_to_cuda(conditioning)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            samples = model.generate(
                **conditioning,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                guidance_scale=cfg
            )
        conditioning = tensor_dict_to_cpu(conditioning)
        del conditioning
        model = model.cpu()
        if samples.dim() == 2:
            samples = samples.unsqueeze(1)
        return samples.cpu(),


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
