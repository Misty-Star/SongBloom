"""提取参考音频片段（prompt_wav）。

用法:
    python -m training.preprocess.extract_prompt \
        --audio-dir /path/to/audio \
        --output-dir /path/to/output
"""

import argparse
import os

import torchaudio
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--prompt-len", type=float, default=10.0, help="参考音频时长(秒)")
    args = parser.parse_args()

    prompt_samples = int(args.prompt_len * args.sr)
    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith((".wav", ".flac", ".mp3"))]

    for fname in tqdm(audio_files, desc="Extracting prompt"):
        stem = os.path.splitext(fname)[0]
        out_dir = os.path.join(args.output_dir, stem)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, "prompt_wav.flac")
        if os.path.exists(out_path):
            continue

        wav, sr = torchaudio.load(os.path.join(args.audio_dir, fname))
        if sr != args.sr:
            wav = torchaudio.functional.resample(wav, sr, args.sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # 截取前 prompt_len 秒
        prompt = wav[:, :prompt_samples]
        torchaudio.save(out_path, prompt, args.sr)


if __name__ == "__main__":
    main()
