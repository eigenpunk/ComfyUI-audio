# ComfyUI-audio

generative audio tools for ComfyUI. highly experimental&mdash;expect things to break and/or change frequently.

## features
- [musicgen text-to-music + audiogen text-to-sound](https://facebookresearch.github.io/audiocraft/docs/MUSICGEN.html)
    - audiocraft and transformers implementations
    - supports audio continuation, unconditional generation
- [tortoise text-to-speech](https://github.com/neonbjb/tortoise-tts)
    - uses [152334h's fork](https://github.com/152334H/tortoise-tts-fast)
- [vall-e x text-to-speech](https://github.com/Plachtaa/VALL-E-X)
    - uses [korakoe's fork](https://github.com/korakoe/VALL-E-X)
- [voicefixer2](https://github.com/voicefixer/voicefixer)
- audio utility nodes
    - save audio, convert audio

## installation
```bash
# make sure you have activated the python environment used by ComfyUI
cd ComfyUI/custom_nodes
git clone https://github.com/eigenpunk/ComfyUI-audio
cd ComfyUI-audio
# pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118  # for cuda 11.8
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121  # for cuda 12.1
pip install -U audiocraft --no-deps
```

## would be nice to have maybe
- audio uploads
- audio previews
- prompt weights for text-to-music/audio
- stereo musicgen
- ~~audiogen~~
- multi-band diffusion
- more/faster tts model support
    - [tacotron2](https://github.com/NVIDIA/tacotron2)
    - [vits](https://huggingface.co/docs/transformers/model_doc/vits)?
    - ~~[vall-e x](https://github.com/Plachtaa/VALL-E-X)~~
    <!-- 
    these implementations exist but seem not to have trained checkpoints:
    - [voicebox](https://github.com/lucidrains/voicebox-pytorch)?
    - [naturalspeech](https://github.com/lucidrains/naturalspeech2-pytorch)?
    -->
    - ???
- split generator nodes by model stages
    <!-- - for tortoise, could split the node into:
        - autoregressor
        - clvp/cvvp
        - spectrogram diffusion
    - musicgen components:
        - t5 text encoder
        - encodec audio encoder
        - decoder -->
- more audio generation models
    <!-- - [audiolm](https://github.com/lucidrains/audiolm-pytorch)/[musiclm](https://github.com/lucidrains/musiclm-pytorch) -->
- demucs
