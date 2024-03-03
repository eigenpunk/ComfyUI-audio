import math
import os
import random
import shutil
import subprocess
import librosa
import torch
from torch import hann_window

import numpy as np
import scipy
import resampy
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


# filters that only require width
FILTER_WINDOWS = {
    x.__name__.split(".")[-1]: x for x in [
        scipy.signal.windows.boxcar,
        scipy.signal.windows.triang,
        scipy.signal.windows.blackman,
        scipy.signal.windows.hamming,
        scipy.signal.windows.hann,
        scipy.signal.windows.bartlett,
        scipy.signal.windows.flattop,
        scipy.signal.windows.parzen,
        scipy.signal.windows.bohman,
        scipy.signal.windows.blackmanharris,
        scipy.signal.windows.nuttall,
        scipy.signal.windows.barthann,
        scipy.signal.windows.cosine,
        scipy.signal.windows.exponential,
        scipy.signal.windows.tukey,
        scipy.signal.windows.taylor,
        scipy.signal.windows.lanczos,
    ]
}
MAX_WAV_VALUE = 32768.0


class LoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
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


class NormalizeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "power": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
    
    RETURN_NAMES = ("AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "normalize_audio"
    CATEGORY = "audio"

    def normalize_audio(self, audio, power):
        normed_audio = []
        for clip in audio:
            normed_clip = clip * (1.0 / clip.abs().max()) ** power
            normed_audio.append(normed_clip)
        return normed_audio,


class ClipAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "sr": ("INT", {"default": 32000, "min": 0, "max": 2 ** 32}),
                "from_s": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "to_s": ("FLOAT", {"default": 0.0, "step": 0.001}),
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


class TrimAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "sr": ("INT", {"default": 32000, "min": 0, "max": 2 ** 32}),
                "s_from_start": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "s_from_end": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }
    
    RETURN_NAMES = ("AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "clip_audio"
    CATEGORY = "audio"

    def clip_audio(self, audio, sr, s_from_start, s_from_end):
        from_sample = int(s_from_start * sr)
        to_sample = (int(s_from_end * sr) + 1)
        clipped_audio = []
        for a in audio:
            a_clipped = a[..., from_sample:-to_sample]
            clipped_audio.append(a_clipped)
        return clipped_audio,


class TrimAudioSamples:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "from_start": ("INT", {"default": 0, "min": 0, "max": 2 ** 32, "step": 1}),
                "from_end": ("INT", {"default": 0, "min": 0, "max": 2 ** 32, "step": 1}),
            }
        }
    
    RETURN_NAMES = ("AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "clip_audio"
    CATEGORY = "audio"

    def clip_audio(self, audio, from_start, from_end):
        from_sample = from_start
        to_sample = from_end + 1
        clipped_audio = []
        for a in audio:
            a_clipped = a[..., from_sample:-to_sample]
            clipped_audio.append(a_clipped)
        return clipped_audio,


class FlattenAudioBatch:
    """
    flatten a batch of audio into a single audio tensor
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio_batch": ("AUDIO_TENSOR",)}}
    
    RETURN_NAMES = ("AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "concat_audio"
    CATEGORY = "audio"

    def concat_audio(self, audio_batch):
        return [torch.concat(audio_batch, dim=-1)],


class ConcatAudio:
    """
    concatenate two batches of audio along their time dimensions

    mismatched batch sizes are not supported unless one of the batches is size 1: if a batch has only
    one item it will be repeated to match the size of the other batch if necessary.
    """
    @classmethod
    def INPUT_TYPES(cls):
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
    """
    combine two AUDIO_TENSOR batches together.
    """
    @classmethod
    def INPUT_TYPES(cls):
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
    """
    convert an AUDIO_TENSOR's sample rate and number of channels
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "from_rate": ("INT", {"default": 44100, "min": 1, "max": 2 ** 32}),
                "to_rate": ("INT", {"default": 32000, "min": 1, "max": 2 ** 32}),
                "to_channels": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "convert"
    CATEGORY = "audio"

    def convert(self, audio, from_rate, to_rate, to_channels):
        converted = []
        for clip in audio:
            expand_dim = len(clip.shape) == 2
            if expand_dim:
                clip = clip.unsqueeze(0)
            conv_clip = convert_audio(clip, from_rate, to_rate, to_channels)
            conv_clip = conv_clip.squeeze(0) if expand_dim else conv_clip
            converted.append(conv_clip)
        return converted,


class ResampleAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "from_rate": ("INT", {"default": 44100, "min": 1, "max": 2 ** 32}),
                "to_rate": ("INT", {"default": 32000, "min": 1, "max": 2 ** 32}),
                "filter": (["sinc_window", "kaiser_best", "kaiser_fast"], ),
                "window": (list(FILTER_WINDOWS.keys()),),
                "num_zeros": ("INT", {"default": 64, "min": 1, "max": 2 ** 32})
            }
        }

    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "convert"
    CATEGORY = "audio"

    def convert(self, audio, from_rate, to_rate, filter, window, num_zeros):
        converted = []
        w = FILTER_WINDOWS[window]
        for clip in audio:
            new_clip = resampy.resample(clip.numpy(), from_rate, to_rate, filter=filter, window=w, num_zeros=num_zeros, parallel=False)
            converted.append(torch.from_numpy(new_clip))
        return converted,


class SaveAudio:
    """
    save an AUDIO_TENSOR to disk. if the input is a batch, each item will be saved separately.
    """
    def __init__(self):
        self.output_dir = get_output_directory()
        self.output_type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
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
    """
    create spectrogram images from audio.
    """
    @classmethod
    def INPUT_TYPES(cls):
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


class BlendAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_to": ("AUDIO_TENSOR",),
                "audio_from": ("AUDIO_TENSOR",),
                "audio_to_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "blend"
    CATEGORY = "audio"

    def blend(self, audio_to, audio_from, audio_to_strength):
        blended_audio = []

        for a_to, a_from in zip(audio_to, audio_from):
            to_n = a_to.shape[-1]
            from_n = a_from.shape[-1]

            if to_n > from_n:
                leftover = a_to[..., from_n:]
                a_to = a_to[..., :from_n]

            elif from_n > to_n:
                leftover = a_from[..., to_n:]
                a_from = a_from[..., :to_n]

            else:
                leftover = torch.empty(0, dtype=a_to.dtype)

            new_a = audio_to_strength * a_to + (1 - audio_to_strength) * a_from
            blended_audio.append(torch.cat((new_a, leftover), dim=-1))

        return blended_audio,


class InvertPhase:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
            }
        }
    
    RETURN_NAMES = ("AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "invert"
    CATEGORY = "audio"

    def invert(self, audio):
        normed_audio = []
        for clip in audio:
            normed_clip = -clip
            normed_audio.append(normed_clip)
        return normed_audio,



class FilterAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "numtaps": ("INT", {"default": 101, "min": 1, "max": 2 ** 32}),
                "cutoff": ("INT", {"default": 10500, "min": 1, "max": 2 ** 32}),
                "width": ("INT", {"default": 0, "min": 0, "max": 2 ** 32}),
                "window": (list(FILTER_WINDOWS.keys()),),
                "pass_zero": ("BOOLEAN", {"default": True}),
                "scale": ("BOOLEAN", {"default": True}),
                "fs": ("INT", {"default": 32000, "min": 1, "max": 2 ** 32}),
            }
        }
    
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "filter_audio"
    CATEGORY = "audio"

    def filter_audio(self, audio, numtaps, cutoff, width, window, pass_zero, scale, fs):
        if width == 0:
            width = None

        filtered = []
        f = scipy.signal.firwin(numtaps, cutoff, width=width, window=window, pass_zero=pass_zero, scale=scale, fs=fs)
        for clip in audio:
            dtype = clip.dtype
            filtered_clip = scipy.signal.lfilter(f, [1.0], clip.numpy() * MAX_WAV_VALUE)
            filtered.append(torch.from_numpy(filtered_clip / MAX_WAV_VALUE).to(dtype=dtype))

        return filtered,


class CombineImageWithAudio:
    """
    combine an image and audio into a video clip.
    """
    def __init__(self):
        self.output_dir = get_output_directory()
        self.output_type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
        return {
            "required": 
                {
                    "audio": ("AUDIO_TENSOR",),
                    "mode": ("INT", {"default": 0, "min": 0, "max": 2}),
                },
            }
    
    FUNCTION = "apply"
    RETURN_TYPES = ("AUDIO_TENSOR",)
    CATEGORY = "audio"

    def apply(self, audio, mode):
        device = get_device()
        if self.model is None:
            from voicefixer import VoiceFixer
            self.model = VoiceFixer()

        results = []
        with on_device(self.model, dst=device) as model:
            for clip in audio:
                output = model.restore_inmem(clip.squeeze(0).numpy(), cuda=device == "cuda", mode=mode)
                results.append(clip.new_tensor(output))

        do_cleanup()
        return results,


class TrimSilence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO_TENSOR",),
                "top_db": ("FLOAT", {"default": 0.0}),
            }
        }
    
    FUNCTION = "trim"
    RETURN_TYPES = ("AUDIO_TENSOR",)
    CATEGORY = "audio"

    def trim(self, audio, top_db=6.0):
        trimmed_audio = []
        for clip in audio:
            trimmed_clip, _ = librosa.effects.trim(clip, top_db=top_db, frame_length=256, hop_length=128)
            trimmed_audio.append(trimmed_clip)
        return trimmed_audio,


NODE_CLASS_MAPPINGS = {
    "LoadAudio": LoadAudio,
    "SaveAudio": SaveAudio,
    "ConvertAudio": ConvertAudio,
    "FilterAudio": FilterAudio,
    "ResampleAudio": ResampleAudio,
    "ClipAudioRegion": ClipAudio,
    "InvertAudioPhase": InvertPhase,
    "TrimAudio": TrimAudio,
    "TrimAudioSamples": TrimAudioSamples,
    "ConcatAudio": ConcatAudio,
    "BlendAudio": BlendAudio,
    "BatchAudio": BatchAudio,
    "FlattenAudioBatch": FlattenAudioBatch,
    "SpectrogramImage": SpectrogramImage,
    "CombineImageWithAudio": CombineImageWithAudio,
    "ApplyVoiceFixer": ApplyVoiceFixer,
    "TrimSilence": TrimSilence,
    "NormalizeAudio": NormalizeAudio,
    # "PreviewAudio": PreviewAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudio": "Load Audio",
    "SaveAudio": "Save Audio",
    "ConvertAudio": "Convert Audio",
    "FilterAudio": "Filter Audio",
    "ResampleAudio": "Resample Audio",
    "ClipAudioRegion": "Clip Audio Region",
    "InvertAudioPhase": "Invert Audio Phase",
    "TrimAudio": "Trim Audio",
    "TrimAudioSamples": "Trim Audio (by samples)",
    "ConcatAudio": "Concatenate Audio",
    "BlendAudio": "Blend Audio",
    "BatchAudio": "Batch Audio",
    "FlattenAudioBatch": "Flatten Audio Batch",
    "SpectrogramImage": "Spectrogram Image",
    "CombineImageWithAudio": "Combine Image with Audio",
    "ApplyVoiceFixer": "Apply VoiceFixer",
    "TrimSilence": "Trim Silence",
    "NormalizeAudio": "Normalize Audio",
    # "PreviewAudio": "Preview Audio",
}
