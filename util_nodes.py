import math
import os
import random
import shutil
import subprocess
import torch
from torch import hann_window

import numpy as np
import torchaudio.functional as TAF
from audiocraft.data.audio import audio_write, audio_read
from audiocraft.data.audio_utils import convert_audio
from PIL import Image

from comfy.cli_args import args

from .util import (
    do_cleanup,
    get_device,
    get_output_directory,
    get_temp_directory,
    get_save_image_path,
    on_device,
)


class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"path": ("STRING", {"default": ""})}}

    RETURN_NAMES = ("AUDIO", "SR", "DURATION")
    RETURN_TYPES = ("AUDIO_TENSOR", "INT", "FLOAT")
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self, path):
        if not os.path.isabs(path):
            path = os.path.join(get_output_directory(), path)
        audio, sr = audio_read(path)
        return [audio], sr, audio.shape[-1] / sr


class ClipAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "sr": ("INT", {"default": 32000}),
                "from_s": ("FLOAT", {"default": 0.0}),
                "to_s": ("FLOAT", {"default": 0.0}),
            }
        }
    
    RETURN_NAMES = ("AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "clip_audio"
    CATEGORY = "audio"

    def clip_audio(self, audio, sr, from_s, to_s):
        from_sample = int(from_s * sr)
        to_sample = int(to_s * sr)
        clipped_audio = []
        for a in audio:
            a_clipped = a[..., from_sample:to_sample]
            clipped_audio.append(a_clipped)
        return clipped_audio,


class FlattenAudioBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio_batch": ("AUDIO_TENSOR",)}}
    
    RETURN_NAMES = ("AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "concat_audio"
    CATEGORY = "audio"

    def concat_audio(self, audio_batch):
        return [torch.concat(audio_batch, dim=-1)],


class ConcatAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch1": ("AUDIO_TENSOR",),
                "batch2": ("AUDIO_TENSOR",),
            }
        }
    
    RETURN_NAMES = ("AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "concat_audio"
    CATEGORY = "audio"

    def concat_audio(self, batch1, batch2):
        if len(batch1) == len(batch2) and len(batch2) == 1:
            return torch.concat([batch1[0], batch2[0]], dim=-1)

        b1 = batch1.copy()
        b2 = batch2.copy()

        if len(b1) == 1:
            b1 = b1 * len(b2)
        elif len(b2) == 1:
            b2 = b2 * len(b1)

        batch = [torch.concat([x1, x2], dim=-1) for x1, x2 in zip(b1, b2)]

        return batch,


class BatchAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch1": ("AUDIO_TENSOR",),
                "batch2": ("AUDIO_TENSOR",),
            }
        }
    
    RETURN_NAMES = ("AUDIO_BATCH",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "batch_audio"
    CATEGORY = "audio"

    def batch_audio(self, batch1, batch2):
        return batch1 + batch2,


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


def logyscale(img_array):
    height, width = img_array.shape[:2]

    def _remap(x, y):
        return min(int(math.log(x + 1) * height / math.log(height)), height - 1), min(y, width - 1)
    v_remap = np.vectorize(_remap)

    indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    indices = v_remap(*indices)
    img_array = img_array[indices]

    return img_array


class SpectrogramImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "n_fft": ("INT", {"default": 200}),
                "hop_len": ("INT", {"default": 50}),
                "win_len": ("INT", {"default": 100}),
                "power": ("FLOAT", {"default": 1.0}),
                "normalized": ("BOOLEAN", {"default": False}),
                "logy": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_spectrogram"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def make_spectrogram(self, audio, n_fft=400, hop_len=50, win_len=100, power=1.0, normalized=False, logy=True):
        hop_len = n_fft // 4 if hop_len == 0 else hop_len
        win_len = n_fft if win_len == 0 else win_len

        results = []
        for clip in audio:
            spectro = TAF.spectrogram(
                clip,
                0,
                window=hann_window(win_len),
                n_fft=n_fft,
                hop_length=hop_len,
                win_length=win_len,
                power=power,
                normalized=normalized,
                center=True,
                pad_mode="reflect",
                onesided=True,
            )
            spectro = spectro[0].flip(0)

            if logy:
                spectro = clip.new_tensor(logyscale(spectro.numpy()))
                
            results.append(spectro)

        return results,


class CombineImageWithAudio:
    def __init__(self):
        self.output_dir = get_output_directory()
        self.output_type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO_TENSOR",),
                "sr": ("INT", {"default": 32000}),
                "file_format": (["webm", "mp4"],),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image_with_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def save_image_with_audio(self, image, audio, sr, file_format, filename_prefix):
        filename_prefix += self.prefix_append
        dur = audio[0].shape[-1] // sr
        channels = audio[0].shape[-2]
        full_outdir, base_fname, count, subdir, filename_prefix = get_save_image_path(
            filename_prefix, self.output_dir, dur, channels
        )

        audio_results = []
        video_results = []
        for image_tensor, clip in zip(image, audio):
            name = f"{base_fname}_{count:05}_"
            stem_name = os.path.join(full_outdir, name)
            audio_path = audio_write(stem_name, clip, sr, format="wav")

            image = image_tensor.mul(255.0).clip(0, 255).byte().numpy()
            image = Image.fromarray(image)

            image_path = os.path.join(full_outdir, f"{name}.png")
            image.save(image_path, compress_level=4)

            video_path = os.path.join(full_outdir, f"{name}.{file_format}")

            proc_args = [
                shutil.which("ffmpeg"), "-y", "-i", image_path, "-i", str(audio_path)
            ]
            if file_format == "webm":
                proc_args += ["-c:v", "vp8", "-c:a", "opus", "-strict", "-2", video_path]
            else:  # file_format == "mp4"
                proc_args += ["-pix_fmt", "yuv420p", video_path]
                
            subprocess.run(proc_args)

            d = {"subfolder": subdir, "type": self.output_type}
            audio_results.append({
                **d, "filename": f"{name}.wav", "format": "audio/wav",
            })
            video_results.append({
                **d,
                "filename": f"{name}.{file_format}",
                "format": "video/webm" if file_format == "webm" else "video/mpeg",
            })
            count += 1

        return {"ui": {"clips": audio_results, "videos": video_results}}


class ApplyVoiceFixer:
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO_TENSOR",),}}
    
    FUNCTION = "apply"
    RETURN_TYPES = ("AUDIO_TENSOR",)
    CATEGORY = "audio"

    def apply(self, audio):
        device = get_device()
        if self.model is None:
            from voicefixer import VoiceFixer
            self.model = VoiceFixer()

        results = []
        with on_device(self.model, dst=device) as model:
            for clip in audio:
                output = model.restore_inmem(clip.squeeze(0).numpy(), cuda=device == "cuda")
                results.append(clip.new_tensor(output))

        do_cleanup()
        return results,


NODE_CLASS_MAPPINGS = {
    "LoadAudio": LoadAudio,
    "SaveAudio": SaveAudio,
    "ConvertAudio": ConvertAudio,
    "ClipAudio": ClipAudio,
    "ConcatAudio": ConcatAudio,
    "BatchAudio": BatchAudio,
    "FlattenAudioBatch": FlattenAudioBatch,
    "SpectrogramImage": SpectrogramImage,
    "CombineImageWithAudio": CombineImageWithAudio,
    "ApplyVoiceFixer": ApplyVoiceFixer,
    # "PreviewAudio": PreviewAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudio": "Load Audio",
    "SaveAudio": "Save Audio",
    "ConvertAudio": "Convert Audio",
    "ClipAudio": "Clip Audio",
    "ConcatAudio": "Concatenate Audio",
    "BatchAudio": "Batch Audio",
    "FlattenAudioBatch": "Flatten Audio Batch",
    "SpectrogramImage": "Spectrogram Image",
    "CombineImageWithAudio": "Combine Image with Audio",
    "ApplyVoiceFixer": "Apply VoiceFixer",
    # "PreviewAudio": "Preview Audio",
}
