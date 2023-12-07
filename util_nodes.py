import os
import random

from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

from .util import get_output_directory, get_temp_directory, get_save_image_path


class ConvertAudio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "from_rate": ("INT", {"default": 44100}),
                "to_rate": ("INT", {"default": 32000}),
                "to_channels": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "convert"
    CATEGORY = "audio"

    def convert(self, audio, from_rate, to_rate, to_channels):
        for i, clip in enumerate(audio):
            expand_dim = len(clip.shape) == 2
            if expand_dim: clip = clip.unsqueeze(0)
            conv_clip = convert_audio(clip, from_rate, to_rate, to_channels)
            audio[i] = conv_clip.squeeze(0) if expand_dim else conv_clip
        return audio,


class SaveAudio:
    def __init__(self):
        self.output_dir = get_output_directory()
        self.output_type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "sr": ("INT", {"default": 32000}),
                "file_format": (["wav", "mp3", "ogg", "flac"],),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def save_audio(
        self,
        audio,
        sr,
        file_format,
        filename_prefix="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
    ):
        filename_prefix += self.prefix_append
        dur = audio[0].shape[-1] // sr
        channels = audio[0].shape[-2]
        full_outdir, base_fname, count, subdir, filename_prefix = get_save_image_path(
            filename_prefix, self.output_dir, dur, channels
        )

        mimetype = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
        }[file_format]

        results = []
        for clip in audio:
            name = f"{base_fname}_{count:05}_"
            stem_name = os.path.join(full_outdir, name)
            path = audio_write(stem_name, clip, sr, format=file_format)
            result = {
                "filename": path.name,
                "subfolder": subdir,
                "type": self.output_type,
                "format": mimetype,
            }
            results.append(result)
            count += 1

        return {"ui": {"clips": results}}


class PreviewAudio(SaveAudio):
    r"""
    NOTE: this doesn't actually do anything yet, need to write js extension code
    """
    def __init__(self):
        self.output_dir = get_temp_directory()
        self.output_type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5)
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "sr": ("INT", {"default": 32000}),
                "file_format": (["wav", "mp3", "ogg", "flac"],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    

NODE_CLASS_MAPPINGS = {
    "SaveAudio": SaveAudio,
    "ConvertAudio": ConvertAudio,
    # "PreviewAudio": PreviewAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAudio": "Save Audio",
    "ConvertAudio": "Convert Audio",
    # "PreviewAudio": "Preview Audio",
}
