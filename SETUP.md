# Using seed-vc as the `seed` package

This fork is installed as a Python package named `seed` and used by `consumer_code` for
voice conversion (as the primary or fallback VC step in P4). The public entry point is
`seed.clone_voice`.

## 1. Install (editable)

The pipeline uses an editable install, so changes in this repo apply immediately without
reinstalling.

```bash
git clone git@github.com:neuralgarage/seed-vc-fork-2.git /home/neural/development_repos/lib/seed-vc-fork-2
conda activate hyperp_env          # the consumer_code environment (Python 3.11)
cd /home/neural/development_repos/lib/seed-vc-fork-2
pip install -e .
```

`setup.py` installs the dependencies from `requirements-mac.txt` (torch 2.4.0,
transformers 4.46.3, huggingface-hub, librosa, ...), not `requirements.txt`.

Check the install points at this repo:

```bash
python -c "import seed.voice_cloner as v; print(v.__file__)"
# -> /home/neural/development_repos/lib/seed-vc-fork-2/seed/voice_cloner.py
```

## 2. Model files

All weights live under one cache directory. `consumer_code` uses
`<external_checkpoint_dir>/seed_vc`, where `external_checkpoint_dir` comes from
`consumer_code/conf/.env` (default `/home/neural/model_checkpoints/external_checkpoints`).

```
seed_vc/
├── seed_vc_large.pth                                  # fine-tuned DiT checkpoint (passed as `checkpoint`)
├── config_dit_mel_seed_uvit_whisper_large_wavenet.yml # its config (passed as `config`)
├── models--vasista22--whisper-hindi-large-v2/         # speech tokenizer, downloaded from HF
├── models--nvidia--bigvgan_v2_22khz_80band_256x/      # vocoder, downloaded from HF
├── models--funasr--campplus/                          # speaker embedding, downloaded from HF
└── models--Plachta--Seed-VC/                          # default DiT, only used when no checkpoint is given
```

`seed_vc_large.pth` and its yml must be copied in by hand. The `models--*` folders are
downloaded from Hugging Face into `cache_dir` on the first run (about 7 GB, mostly whisper),
so the first run on a new machine needs internet access.

## 3. Calling it directly

```python
from seed import clone_voice

out_path = clone_voice(
    src_voice="input.wav",          # audio to convert
    ref_voice="target_speaker.wav", # voice to convert into (first 25 s are used)
    output_dir="out/",
    checkpoint="/home/neural/model_checkpoints/external_checkpoints/seed_vc/seed_vc_large.pth",
    config="/home/neural/model_checkpoints/external_checkpoints/seed_vc/config_dit_mel_seed_uvit_whisper_large_wavenet.yml",
    cache_dir="/home/neural/model_checkpoints/external_checkpoints/seed_vc",
    # optional: diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7,
    #           f0_condition=False, auto_f0_adjust=False, semi_tone_shift=0, fp16=False
)
```

It returns the path of the written wav (`vc_<src>_<ref>_<length_adjust>_<steps>_<cfg>.wav`
inside `output_dir`).

## 4. How consumer_code uses it

Call chain:

```
executors/master_executor.py
  -> P4Executor.exec -> generate_audio_vc            (executors/p4_executor.py)
  -> AudioGenerator.generate_vc                      (editors/audio.py)
  -> seed_vc_voice_clone                             (external/eleven_labs_tts.py)
  -> seed.voice_cloner.clone_voice
```

`seed_vc_voice_clone` supplies `cache_dir` itself and passes the split's `params` dict as
keyword arguments to `clone_voice`. So `params` may hold any `clone_voice` argument
except `src_voice`, `ref_voice`, `output_dir` and `cache_dir`. Passing `cache_dir` in
`params` causes a duplicate-argument `TypeError`.

`seed_vc_voice_clone` catches every exception, logs it and returns `None`, so a seed-vc
failure only shows up as an error log followed by a later failure on the missing file.

The reference voice comes from the payload key `ref_voice`.

### Config

Setting `"do_voice_conversion": "seedvc"` or `"fallback_params": {"vc_method": "seed_vc"}` in
the pipeline input only makes the config generator write a placeholder:

```json
"vc_params": {"src": "add fallback seed-vc model", "params": {}}
```

Edit each audio split in the generated config so it reads:

```json
"fallback_params": {
  "n_retries": 2,
  "tts_params": {"id": "hi-IN-ArjunNeural", "src": "azure", "params": {}},
  "vc_params": {
    "src": "seed_vc",
    "params": {
      "checkpoint": "/home/neural/model_checkpoints/external_checkpoints/seed_vc/seed_vc_large.pth",
      "config": "/home/neural/model_checkpoints/external_checkpoints/seed_vc/config_dit_mel_seed_uvit_whisper_large_wavenet.yml"
    }
  }
}
```

The same `vc_params` block works at split level for seed-vc as the primary VC.

Do not leave the placeholder in place. In fallback params, any `src` other than
`eleven_labs_tts` is parsed as seed-vc with empty params, which silently loads the default
`Plachta/Seed-VC` model (downloading it if missing) instead of your checkpoint.

## 5. Models stay in memory after the first call

`load_models` is wrapped in `functools.lru_cache(maxsize=1)`. Within one Python process:

- The first `clone_voice` call takes about 6–8 s: about 5.5 s to load the models plus CUDA
  warm-up. It is slower when the model files are not already in the OS file cache.
- Later calls with the same `checkpoint`, `config`, `f0_condition`, `fp16` and `cache_dir`
  reuse the loaded models (see the timings in section 6).
- The models keep about 2.2 GiB of GPU memory until the process exits, peaking at
  2.3–2.7 GiB during a call.
- Only one model configuration is kept. Calling with a different `checkpoint`, `config`,
  `f0_condition` or `fp16` replaces it and triggers a full reload.

Nothing persists across processes, so each `master_executor.py` run pays the load once.

To free the GPU memory early:

```python
import torch
from seed.voice_cloner import load_models

load_models.cache_clear()
torch.cuda.empty_cache()
```

## 6. Timings

Measured on an RTX 4090 with the `seed_vc_large.pth` config, a 6 s reference voice and the
default 30 diffusion steps, with the models already loaded:

| Input audio | Time per call |
|---|---|
| 1–3 s | 0.39–0.46 s |
| 6 s | 0.67 s |
| 12 s | 1.0 s |
| 24 s | 2.2 s |

- The diffusion step is 75–80% of each call. The whisper encoder costs a flat ~50 ms because
  it pads every input to 30 s.
- The reference voice length counts as much as the input length: with a 24 s reference, a
  2 s input takes 1.36 s instead of 0.44 s. Keep references around 6 s.
- Time grows linearly with `diffusion_steps`, about 11 ms per step for a 2 s input:
  10 → 0.22 s, 20 → 0.32 s, 30 → 0.44 s, 50 → 0.66 s.
- `fp16: true` cuts 30–40% (0.31 s for a 2 s input, 0.40 s for 6 s) but reloads the models
  on its first call, since it changes the cache key.
- Only speed was measured. Check output quality before lowering `diffusion_steps` or
  turning on `fp16`.

## 7. Things to know

- `cache_dir` is the download cache on disk. It avoids re-downloading weights but does not
  keep models in memory; the `lru_cache` above does that.
- On each fresh load, Hugging Face still checks online whether a newer revision exists and
  downloads it if so. Set `HF_HUB_OFFLINE=1` to skip this once all weights are cached.
  It applies to every Hugging Face model in the process.
- Importing `seed` imports torch, picks the device (CUDA, then MPS, then CPU) and sets
  `HF_HUB_CACHE=./checkpoints/hf_cache`. The explicit `cache_dir` argument overrides that
  path.
- The `hifigan` vocoder option reads `configs/hifigan.yml` relative to the working directory,
  so it only works when run from this repo's root. The configs used by the pipeline use
  `bigvgan` and are not affected.
