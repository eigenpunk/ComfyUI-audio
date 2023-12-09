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
    stack_audio_tensors,
    tensors_to,
    tensors_to_cpu,
)

import langid
from audiocraft.data.audio_utils import normalize_loudness
from vallex.data import AudioTokenizer
from vallex.data.collation import get_text_token_collater
from vallex.models.vallex import VALLE
from vallex.utils.g2p import PhonemeBpeTokenizer
from vallex.utils.generation import url as VALLEX_CKPT_URL
from vallex.utils.macros import *
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


text_collater = get_text_token_collater()


# NOTE: the following function is adapted from Plachtaa's implementation of VALL-E X:
# https://github.com/Plachtaa/VALL-E-X


@torch.no_grad()
def generate_audio(
    model,
    codec,
    vocos,
    text_tokenizer,
    text,
    prompt,
    language="auto",
    accent="no-accent",
    topk=100,
    temperature=1.0,
    length_penalty=1.0,
    device=None,
):
    text = text.replace("\n", "").strip(" ")
    # detect language
    if language == "auto":
        language = langid.classify(text)[0]
    lang_token = lang2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # load prompt
    if prompt is not None:
        prompt_path = prompt
        if not os.path.exists(prompt_path):
            prompt_path = os.path.join(VOICES_PATH, "presets", prompt + ".npz")
        if not os.path.exists(prompt_path):
            prompt_path = os.path.join(VOICES_PATH, "customs", prompt + ".npz")
        if not os.path.exists(prompt_path):
            raise ValueError(f"Cannot find prompt {prompt}")
        prompt_data = np.load(prompt_path)
        audio_prompts = prompt_data["audio_tokens"]
        text_prompts = prompt_data["text_tokens"]
        lang_pr = prompt_data["lang_code"]
        lang_pr = code2lang[int(lang_pr)]

        # numpy to tensor
        audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
        text_prompts = torch.tensor(text_prompts).type(torch.int32)
    else:
        audio_prompts = torch.zeros([1, 0, NUM_QUANTIZERS]).type(torch.int32).to(device)
        text_prompts = torch.zeros([1, 0]).type(torch.int32)
        lang_pr = lang if lang != "mix" else "en"

    enroll_x_lens = text_prompts.shape[-1]
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
    text_tokens, text_tokens_lens = text_collater([phone_tokens])
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens
    # accent control
    lang = lang if accent == "no-accent" else accent
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        top_k=-topk,
        temperature=temperature,
        prompt_language=lang_pr,
        text_language=langs if accent == "no-accent" else lang,
        length_penalty=length_penalty,
    )
    # Decode with Vocos
    frames = encoded_frames.permute(2, 0, 1)
    features = vocos.codes_to_features(frames)
    samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

    return samples.squeeze().cpu().numpy()


class VALLEXLoader:
    def __init__(self):
        self.model = None
        self.codec = None
        self.vocoder = None
        self.tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_NAMES = ("VALLEX_MODEL", "SR")
    RETURN_TYPES = ("VALLEX_MODEL", "INT")
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self):
        if self.model is not None:
            self.model = self.model.cpu()
            self.codec = object_to(self.codec, "cpu")
            self.vocoder = self.vocoder.cpu()
            del self.model, self.codec, self.vocoder, self.tokenizer
            do_cleanup()
            print("VALLEXLoader: unloaded models")

        print("VALLEXLoader: loading models")

        if not os.path.exists(VALLEX_CKPT_PATH):
            # TODO: fetch ckpt
            print("fetching VALL-E X checkpoint...", end="")
            urlretrieve(VALLEX_CKPT_URL, VALLEX_CKPT_PATH)
            print("done.")

        if not os.path.exists(VALLEX_TOKENIZER_PATH):
            # TODO: fetch ckpt
            print("fetching VALL-E X phoneme tokenizer...", end="")
            urlretrieve(VALLEX_TOKENIZER_URL, VALLEX_TOKENIZER_PATH)
            print("done.")

        self.tokenizer = PhonemeBpeTokenizer(VALLEX_TOKENIZER_PATH)
        self.model = VALLE(
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
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()
        self.codec = AudioTokenizer()
        self.vocoder = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
        sr = 24000

        return (self.model, self.codec, self.vocoder, self.tokenizer), sr


class VALLEXGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("VALLEX_MODEL",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "prompt": (VALLEX_VOICEPROMPTS,),
                "language": (["auto", *list(lang2token.keys())],),
                "accent": (ACCENTS,),
                # "mode": (["sliding-window", "fixed-prompt"]),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.001}),
                "topk": ("INT", {"default": 100, "step": 1}),
                "length_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_NAMES = ("RAW_AUDIO",)
    RETURN_TYPES = ("AUDIO_TENSOR",)
    FUNCTION = "generate"
    CATEGORY = "audio"

    def generate(
        self,
        model,
        text: str = None,
        prompt: str = "null",
        language: str = "auto",
        accent: str = "none",
        temperature: float = 1.0,
        topk: int = 100,
        length_penalty: float = 1.0,
        seed: int = 0,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, codec, vocoder, tokenizer = model

        prompt = None if prompt == "null" else VOICES[prompt]
        accent = "no-accent" if accent == "none" else accent

        with (
            torch.random.fork_rng(),
            on_device(model, dst=device) as m,
            obj_on_device(codec, dst=device) as c,
            on_device(vocoder, dst=device) as v
        ):
            torch.manual_seed(seed)
            audio = generate_audio(
                m,
                c,
                v,
                tokenizer,
                text,
                prompt,
                language=language,
                accent=accent,
                topk=-topk,
                temperature=temperature,
                length_penalty=length_penalty,
                device=device,
            )

        return [normalize_loudness(torch.from_numpy(audio).unsqueeze(0), 24000, loudness_compressor=True)],


NODE_CLASS_MAPPINGS = {
    "VALLEXLoader": VALLEXLoader,
    "VALLEXGenerator": VALLEXGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VALLEXLoader": "VALL-E X Loader",
    "VALLEXGenerator": "VALL-E X Generator",
}
