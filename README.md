# ComfyUI-audio

generative audio tools for ComfyUI. highly experimental&mdash;expect things to break.

## features
- [musicgen](https://facebookresearch.github.io/audiocraft/docs/MUSICGEN.html)
    - audiocraft and transformers implementations
- [tortoise tts](https://github.com/neonbjb/tortoise-tts)
- audio utility nodes
    - save audio, convert audio

## installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/eigenpunk/ComfyUI-audio
cd ComfyUI-audio
pip install -r requirements.txt
pip install -U audiocraft --no-deps
```

## would be nice to have maybe
- audio uploads
- audio previews
- stereo musicgen
- audiogen
- multi-band diffusion
