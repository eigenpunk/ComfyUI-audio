from dataclasses import dataclass, field
from glob import glob
import os
import sys
from urllib.request import urlretrieve

import numpy as np
import torch


from .util import (
    models_dir,
    do_cleanup,
    object_to,
    obj_on_device,
    on_device,
    tensors_to,
)

import langid
from audiocraft.data.audio_utils import normalize_loudness
# from vallex.data import AudioTokenizer, tokenize_audio
from encodec.model import EncodecModel
from vallex.data.collation import get_text_token_collater, TextTokenCollater
from vallex.models.vallex import VALLE
from vallex.utils.g2p import PhonemeBpeTokenizer
from vallex.utils.generation import url as VALLEX_CKPT_URL
from vallex.utils.macros import *
from vallex.utils.prompt_making import make_transcript
from vocos import Vocos


MODELS_PATH = os.path.join(models_dir, "vall_e_x")
VOICES_PATH = os.path.join(MODELS_PATH, "voices")
os.makedirs(VOICES_PATH, exist_ok=True)

VOICES = {
    os.path.splitext(os.path.basename(x))[0]: x
    for x in sorted(glob(os.path.join(VOICES_PATH, "*.npz")))
}
ACCENTS = ["none", *list(lang2token.keys())]

VALLEX_CKPT_PATH = os.path.join(MODELS_PATH, "vallex-checkpoint.pt")
VALLEX_TOKENIZER_PATH = os.path.join(MODELS_PATH, "bpe_69.json")
VALLEX_TOKENIZER_URL = "https://raw.githubusercontent.com/korakoe/VALL-E-X/main/vallex/utils/g2p/bpe_69.json"
VALLEX_VOICEPROMPTS = ["null", *VOICES]


@dataclass
class VALLEXModel:
    valle: VALLE
    encodec: EncodecModel
    vocos: Vocos
    tokenizer: PhonemeBpeTokenizer
    collater: TextTokenCollater


# NOTE: the following function is adapted from Plachtaa's implementation of VALL-E X:
# https://github.com/Plachtaa/VALL-E-X


@torch.no_grad()
def generate_audio(
    model,
    text_prompt,
    voice_prompt,
    language="auto",
    accent="no-accent",
    topk=100,
    temperature=1.0,
    best_of=8,
    length_penalty=1.0,
    use_vocos=True,
    device=None,
):
    valle: VALLE = model.valle
    vocoder = model.vocos if use_vocos else model.encodec
    text_tokenizer = model.tokenizer
    text_collater = model.collater

    text = text_prompt.replace("\n", "").strip(" ")

    # detect language
    if language == "auto":
        language = langid.classify(text)[0]
    lang_token = lang2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # load prompt
    audio_prompts, text_prompts, lang_pr = voice_prompt

    enroll_x_lens = text_prompts.shape[-1]
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
    text_tokens, text_tokens_lens = text_collater([phone_tokens])
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens

    # accent control
    lang = lang if accent == "no-accent" else accent
    encoded_frames = valle.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts.to(device),
        enroll_x_lens=enroll_x_lens,
        top_k=topk,
        temperature=temperature,
        prompt_language=lang_pr,
        text_language=langs if accent == "no-accent" else lang,
        best_of=best_of,
        length_penalty=length_penalty,
    )

    # decode
    if use_vocos:
        frames = encoded_frames.permute(2, 0, 1)
        features = vocoder.codes_to_features(frames)
        samples = vocoder.decode(features, bandwidth_id=torch.tensor([2], device=device))
    else:
        samples = vocoder.decode([(encoded_frames.transpose(2, 1), None)])

    return samples.squeeze().cpu().numpy()


class VALLEXLoader:
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_NAMES = ("model", "sr")
    RETURN_TYPES = ("VALLEX_MODEL", "INT")
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self):
        if self.model is not None:
            self.model = object_to(self.model, "cpu")
            del self.model
            do_cleanup()
            print("VALLEXLoader: unloaded models")

        print("VALLEXLoader: loading models")

        if not os.path.exists(VALLEX_CKPT_PATH):
            print("fetching VALL-E X checkpoint...", end="")
            urlretrieve(VALLEX_CKPT_URL, VALLEX_CKPT_PATH)
            print("done.")

        if not os.path.exists(VALLEX_TOKENIZER_PATH):
            print("fetching VALL-E X phoneme tokenizer...", end="")
            urlretrieve(VALLEX_TOKENIZER_URL, VALLEX_TOKENIZER_PATH)
            print("done.")

        valle = VALLE(
            N_DIM,
            NUM_HEAD,
            NUM_LAYERS,
            norm_first=True,
            add_prenet=False,
            prefix_mode=PREFIX_MODE,
            share_embedding=True,
            nar_scale_factor=1.0,
            prepend_bos=True,
            num_quantizers=NUM_QUANTIZERS,
        )
        ckpt = torch.load(VALLEX_CKPT_PATH, map_location="cpu")
        valle.load_state_dict(ckpt["model"], strict=True)
        valle.eval()

        encodec = EncodecModel.encodec_model_24khz()
        encodec.eval()

        vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
        vocos.eval()

        tokenizer = PhonemeBpeTokenizer(VALLEX_TOKENIZER_PATH)

        model = VALLEXModel(valle, encodec, vocos, tokenizer, get_text_token_collater())
        sr = 24000

        do_cleanup()
        return model, sr


class VALLEXGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VALLEX_MODEL",),
                "voice_prompt": ("VALLEX_VPROMPT",),
                "text_prompt": ("STRING", {"default": "", "multiline": True}),
                "language": (["auto", *list(lang2token.keys())],),
                "accent": (ACCENTS,),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.001}),
                "topk": ("INT", {"default": 100, "step": 1}),
                "best_of": ("INT", {"default": 8}),
                "length_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_NAMES = ("audio",)
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio"

    def generate(
        self,
        model,
        voice_prompt,
        text_prompt: str = None,
        language: str = "auto",
        accent: str = "none",
        temperature: float = 1.0,
        topk: int = 100,
        best_of: int = 8,
        length_penalty: float = 1.0,
        seed: int = 0,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        accent = "no-accent" if accent == "none" else accent

        with torch.random.fork_rng(), obj_on_device(model, dst=device) as m:
            torch.manual_seed(seed)
            audio = generate_audio(
                m,
                text_prompt,
                voice_prompt,
                language=language,
                accent=accent,
                topk=-topk,
                temperature=temperature,
                best_of=best_of,
                length_penalty=length_penalty,
                device=device,
            )

        do_cleanup()
        return normalize_loudness(torch.from_numpy(audio).unsqueeze(0), 24000, loudness_compressor=True),


class VALLEXVoicePromptLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice": (VALLEX_VOICEPROMPTS,),
            }
        }

    RETURN_TYPES = ("VALLEX_VPROMPT",)
    FUNCTION = "load_prompt"
    CATEGORY = "audio"

    def load_prompt(self, voice):
        if voice != "null":
            name = VOICES[voice]
            prompt_path = name
            if not os.path.exists(prompt_path):
                prompt_path = os.path.join(VOICES_PATH, "presets", name + ".npz")
            if not os.path.exists(prompt_path):
                prompt_path = os.path.join(VOICES_PATH, "customs", name + ".npz")
            if not os.path.exists(prompt_path):
                raise ValueError(f"Cannot find prompt {name}")
            prompt_data = np.load(prompt_path)
            audio_prompts = prompt_data["audio_tokens"]
            text_prompts = prompt_data["text_tokens"]
            lang_pr = prompt_data["lang_code"]
            lang_pr = code2lang[int(lang_pr)]

            # numpy to tensor
            audio_prompts = torch.tensor(audio_prompts).type(torch.int32)
            text_prompts = torch.tensor(text_prompts).type(torch.int32)
        else:
            audio_prompts = torch.zeros([1, 0, NUM_QUANTIZERS]).type(torch.int32)
            text_prompts = torch.zeros([1, 0]).type(torch.int32)
            lang_pr = "en"

        return (audio_prompts, text_prompts, lang_pr),


class VALLEXVoicePromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VALLEX_MODEL",),
                "transcript": ("STRING", {"default": "", "multiline": True}),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("VALLEX_VPROMPT",)
    FUNCTION = "make_prompt"
    CATEGORY = "audio"

    def make_prompt(self, model, audio, transcript=None):
        encodec: EncodecModel = model.encodec
        tokenizer: PhonemeBpeTokenizer = model.tokenizer
        text_collater: TextTokenCollater = model.collater

        device = "cuda" if torch.cuda.is_available() else "cpu"
        wav_pr = audio["waveform"]

        print(wav_pr)
        print(wav_pr.shape)

        if wav_pr.size(0) == 2:
            wav_pr = wav_pr.mean(0, keepdim=True)

        wav_pr = wav_pr.unsqueeze(0)

        text, lang = make_transcript("_temp_prompt", wav_pr, encodec.sample_rate, transcript)

        with torch.no_grad(), on_device(encodec, dst=device) as e, obj_on_device(tokenizer, dst=device) as t:
            # tokenize audio
            wav_pr = wav_pr.to(device)
            encoded_frames = e.encode(wav_pr)
            audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu()

            # tokenize text
            phonemes, _ = t.tokenize(text=f"{text}".strip())
            text_tokens, _ = text_collater([phonemes])
            wav_pr = wav_pr.cpu()

        do_cleanup()

        return (audio_tokens, text_tokens, lang),


NODE_CLASS_MAPPINGS = {
    "VALLEXLoader": VALLEXLoader,
    "VALLEXGenerator": VALLEXGenerator,
    "VALLEXVoicePromptLoader": VALLEXVoicePromptLoader,
    "VALLEXVoicePromptFromAudio": VALLEXVoicePromptGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VALLEXLoader": "VALL-E X Loader",
    "VALLEXGenerator": "VALL-E X Generator",
    "VALLEXVoicePromptLoader": "VALL-E X Voice Prompt Loader",
    "VALLEXVoicePromptFromAudio": "VALL-E X Voice Prompt from Audio",
}
