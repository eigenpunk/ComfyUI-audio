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
import torchaudio
import torchaudio.functional as TAF
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


class NormalizeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "power": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "normalize_audio"
    CATEGORY = "audio"

    def normalize_audio(self, audio, power):
        clip = audio["waveform"]
        normed_clip = clip * (1.0 / clip.abs().max(dim=-1, keepdim=True)[0]) ** power
        return {"waveform": normed_clip, "sample_rate": audio["sample_rate"]},


class ClipAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "from_s": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "to_s": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "clip_audio"
    CATEGORY = "audio"

    def clip_audio(self, audio, from_s, to_s):
        sr = audio["sample_rate"]
        from_sample = int(from_s * sr)
        to_sample = int(to_s * sr)
        return {"waveform": audio["waveform"][..., from_sample:to_sample], "sample_rate": sr},


class TrimAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "s_from_start": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "s_from_end": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "clip_audio"
    CATEGORY = "audio"

    def clip_audio(self, audio, s_from_start, s_from_end):
        sr = audio["sample_rate"]
        from_sample = int(s_from_start * sr)
        to_sample = (int(s_from_end * sr) + 1)
        return {"waveform": audio["waveform"][..., from_sample:-to_sample], "sample_rate": sr},


class TrimAudioSamples:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "from_start": ("INT", {"default": 0, "min": 0, "max": 2 ** 32, "step": 1}),
                "from_end": ("INT", {"default": 0, "min": 0, "max": 2 ** 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "clip_audio"
    CATEGORY = "audio"

    def clip_audio(self, audio, from_start, from_end):
        from_sample = from_start
        to_sample = from_end + 1
        return {"audio": audio["waveform"][..., from_sample:-to_sample], "sample_rate": audio["sample_rate"]},


class FlattenAudioBatch:
    """
    flatten a batch of audio into a single audio tensor
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio_batch": ("AUDIO",)}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concat_audio"
    CATEGORY = "audio"

    def concat_audio(self, audio_batch):
        audio = audio_batch["waveform"]
        n, c, t = audio.shape
        audio = audio.permute(0, 2, 1)
        return {"waveform": audio.reshape(1, -1, c).permute(0, 2, 1), "sample_rate": audio["sample_rate"]},


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
                "batch1": ("AUDIO",),
                "batch2": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concat_audio"
    CATEGORY = "audio"

    def concat_audio(self, batch1, batch2):
        # TODO: validate that the sample rates are the same
        b1 = batch1["waveform"]
        b2 = batch2["waveform"]

        if len(b1) == 1 and len(b2) != 1:
            b1 = b1.expand(len(b2), -1, -1)
        elif len(b2) == 1 and len(b1) != 1:
            b2 = b2.expand(len(b1), -1, -1)

        return {"waveform": torch.concat([b1, b2], dim=-1), "sample_rate": batch1["sample_rate"]},


class BatchAudio:
    """
    combine two AUDIO batches together.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch1": ("AUDIO",),
                "batch2": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "batch_audio"
    CATEGORY = "audio"

    def batch_audio(self, batch1, batch2):
        batch = torch.cat([batch1["waveform"], batch2["waveform"]], dim=0)
        return {"waveform": batch, "sample_rate": batch1["sample_rate"]},


class ConvertAudio:
    """
    convert audio sample rate and/or number of channels
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "to_rate": ("INT", {"default": 32000, "min": 1, "max": 2 ** 32}),
                "to_channels": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "convert"
    CATEGORY = "audio"

    def convert(self, audio, to_rate, to_channels):
        from_rate = audio["sample_rate"]
        waveform = audio["waveform"]
        waveform = TAF.resample(waveform, from_rate, to_rate)
        if to_channels == 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        elif to_channels == 2 and waveform.shape[1] == 1:
            waveform = waveform.expand(-1, to_channels, -1)

        return {"waveform": waveform, "sample_rate": to_rate},


class ResampleAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "from_rate": ("INT", {"default": 44100, "min": 1, "max": 2 ** 32}),
                "to_rate": ("INT", {"default": 32000, "min": 1, "max": 2 ** 32}),
                "filter": (["sinc_window", "kaiser_best", "kaiser_fast"], ),
                "window": (list(FILTER_WINDOWS.keys()),),
                "num_zeros": ("INT", {"default": 64, "min": 1, "max": 2 ** 32})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "convert"
    CATEGORY = "audio"

    def convert(self, audio, from_rate, to_rate, filter, window, num_zeros):
        converted = []
        w = FILTER_WINDOWS[window]
        for clip in audio["waveform"]:
            new_clip = resampy.resample(clip.numpy(), from_rate, to_rate, filter=filter, window=w, num_zeros=num_zeros, parallel=False)
            converted.append(torch.from_numpy(new_clip))
        return {"waveform": torch.stack(converted, dim=0), "sample_rate": to_rate},


def logyscale(img_array):
    height, width = img_array.shape

    def _remap(y, x):
        return min(int(math.log(y + 1) * height / math.log(height)), height - 1), min(x, width - 1)
    v_remap = np.vectorize(_remap)

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    indices = v_remap(y, x)
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
                "audio": ("AUDIO",),
                "n_fft": ("INT", {"default": 200}),
                "hop_len": ("INT", {"default": 50}),
                "win_len": ("INT", {"default": 100}),
                "power": ("FLOAT", {"default": 1.0}),
                "normalized": ("BOOLEAN", {"default": False}),
                "logy": ("BOOLEAN", {"default": True}),
                "width": ("INT", {"default": 640, "min": 0}),
                "height": ("INT", {"default": 320, "min": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_spectrogram"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def make_spectrogram(
        self,
        audio,
        n_fft=400,
        hop_len=50,
        win_len=100,
        power=1.0,
        normalized=False,
        logy=True,
        width=640,
        height=320,
    ):
        hop_len = n_fft // 4 if hop_len == 0 else hop_len
        win_len = n_fft if win_len == 0 else win_len

        waveform_batch = audio["waveform"]
        results = []
        for clip in waveform_batch:
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
            )  # yields a 1xCxT tensor
            spectro = spectro[0].squeeze().flip(0)  # CxT
            if logy:
                spectro = clip.new_tensor(logyscale(spectro.numpy()))
            results.append(
                torch.nn.functional.interpolate(spectro[None, None], (height, width), mode="bilinear")
                if width != 0 and height != 0
                else spectro[None, None]
            )
        results = torch.cat(results, dim=0).permute(0, 2, 3, 1).expand(-1, -1, -1, 3)
        return results,


class BlendAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_to": ("AUDIO",),
                "audio_from": ("AUDIO",),
                "audio_to_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "blend"
    CATEGORY = "audio"

    def blend(self, audio_to, audio_from, audio_to_strength):
        a_to = audio_to["waveform"]
        a_from = audio_from["waveform"]
        a_to = a_to.float() * MAX_WAV_VALUE
        a_from = a_from.float() * MAX_WAV_VALUE
        to_n = a_to.shape[-1]
        from_n = a_from.shape[-1]

        if to_n > from_n:
            leftover = a_to[..., from_n:]
            a_to = a_to[..., :from_n]
        elif from_n > to_n:
            leftover = a_from[..., to_n:]
            a_from = a_from[..., :to_n]
        else:
            leftover = torch.empty(0, dtype=torch.float)

        new_a = audio_to_strength * a_to + (1 - audio_to_strength) * a_from
        blended_audio = torch.cat((new_a, leftover), dim=-1) / MAX_WAV_VALUE

        return {"waveform": blended_audio, "sample_rate": audio_to["sample_rate"]},


class InvertPhase:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "invert"
    CATEGORY = "audio"

    def invert(self, audio):
        return {"waveform": -audio["waveform"], "sample_rate": audio["sample_rate"]},


class FilterAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "numtaps": ("INT", {"default": 101, "min": 1, "max": 2 ** 32}),
                "cutoff": ("INT", {"default": 10500, "min": 1, "max": 2 ** 32}),
                "width": ("INT", {"default": 0, "min": 0, "max": 2 ** 32}),
                "window": (list(FILTER_WINDOWS.keys()),),
                "pass_zero": ("BOOLEAN", {"default": True}),
                "scale": ("BOOLEAN", {"default": True}),
                "fs": ("INT", {"default": 32000, "min": 1, "max": 2 ** 32}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "filter_audio"
    CATEGORY = "audio"

    def filter_audio(self, audio, numtaps, cutoff, width, window, pass_zero, scale, fs):
        if width == 0:
            width = None

        filtered = []
        f = scipy.signal.firwin(numtaps, cutoff, width=width, window=window, pass_zero=pass_zero, scale=scale, fs=fs)
        for clip in audio["waveform"]:
            filtered_clip = scipy.signal.lfilter(f, [1.0], clip.numpy() * MAX_WAV_VALUE)
            filtered.append(torch.from_numpy(filtered_clip / MAX_WAV_VALUE).float())

        return {"waveform": torch.stack(filtered, dim=0), "sample_rate": audio["sample_rate"]},


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
                "audio": ("AUDIO",),
                "file_format": (["webm", "mp4"],),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image_with_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def save_image_with_audio(self, image, audio, file_format, filename_prefix):
        filename_prefix += self.prefix_append
        sr = audio["sample_rate"]
        full_outdir, base_fname, count, subdir, filename_prefix = get_save_image_path(
            filename_prefix, self.output_dir
        )

        audio_results = []
        video_results = []

        waveform = audio["waveform"]
        for image_tensor, clip in zip(image, waveform):
            name = f"{base_fname}_{count:05}_"
            tmp_dir = get_temp_directory()

            wav_basename = f"{name}.wav"
            wav_fname = os.path.join(full_outdir, wav_basename)
            torchaudio.save(wav_fname, clip, sr, format="wav")

            image = image_tensor.mul(255.0).clip(0, 255).byte().numpy()
            image = Image.fromarray(image)

            image_basename = f"{name}.png"
            image_fname = os.path.join(tmp_dir, image_basename)
            image.save(image_fname, compress_level=4)

            video_basename = f"{name}.{file_format}"
            video_fname = os.path.join(full_outdir, video_basename)

            proc_args = [
                shutil.which("ffmpeg"), "-y", "-i", image_fname, "-i", str(wav_fname)
            ]
            if file_format == "webm":
                proc_args += ["-c:v", "vp8", "-c:a", "opus", "-strict", "-2", video_fname]
            else:  # file_format == "mp4"
                proc_args += ["-pix_fmt", "yuv420p", video_fname]

            subprocess.run(proc_args)

            audio_results.append({
               "filename": wav_basename,
               "format": "audio/wav",
               "subfolder": subdir,
               "type": "output",
            })
            video_results.append({
                "filename": video_basename,
                "format": "video/webm" if file_format == "webm" else "video/mpeg",
                "subfolder": subdir,
                "type": "output",
            })
            count += 1

        return {"ui": {"audio": audio_results, "video": video_results}}


class ApplyVoiceFixer:
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":
                {
                    "audio": ("AUDIO",),
                    "mode": ("INT", {"default": 0, "min": 0, "max": 2}),
                },
            }

    FUNCTION = "apply"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"

    def apply(self, audio, mode):
        device = get_device()
        if self.model is None:
            from voicefixer import VoiceFixer
            self.model = VoiceFixer()

        results = []
        with on_device(self.model, dst=device) as model:
            for clip in audio["waveform"]:
                output = model.restore_inmem(clip.squeeze(0).numpy(), cuda=device == "cuda", mode=mode)
                results.append(clip.new_tensor(output))

        do_cleanup()
        return {"waveform": torch.stack(results), "sample_rate": audio["sample_rate"]},


class TrimSilence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "top_db": ("FLOAT", {"default": 0.0}),
            }
        }

    FUNCTION = "trim"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"

    def trim(self, audio, top_db=6.0):
        if audio["waveform"].shape[0] != 1:
            raise ValueError("Can only trim one audio clip at a time")
        trimmed_clip, _ = librosa.effects.trim(audio["waveform"], top_db=top_db, frame_length=256, hop_length=128)
        return {"waveform": trimmed_clip, "sample_rate": audio["sample_rate"]},


class AudioSampleRate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    FUNCTION = "get_sample_rate"
    RETURN_TYPES = ("INT",)
    CATEGORY = "audio"

    def get_sample_rate(self, audio):
        return audio["sample_rate"],


NODE_CLASS_MAPPINGS = {
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
    "AudioSampleRate": AudioSampleRate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
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
    "AudioSampleRate": "Get Audio Sample Rate",
}
