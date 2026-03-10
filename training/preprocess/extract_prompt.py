"""提取参考音频片段（prompt_wav）。

用法:
    python -m training.preprocess.extract_prompt \
        --audio-dir /path/to/audio \
        --output-dir /path/to/output
"""

import argparse
import json
import os
import typing as tp

import torchaudio
from tqdm import tqdm


def _load_structure_map(path: tp.Optional[str]) -> tp.Dict[str, tp.List[dict]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    raise ValueError("structure-json must be a JSON object keyed by sample id.")


def choose_prompt_window(
    wav,
    sample_rate: int,
    prompt_len: float,
    structure_segments: tp.Optional[tp.Sequence[dict]] = None,
) -> tp.Tuple[int, int]:
    total_samples = wav.shape[-1]
    prompt_samples = int(prompt_len * sample_rate)
    if total_samples <= prompt_samples:
        return 0, total_samples

    structure_segments = list(structure_segments or [])
    vocal_segments = [
        segment
        for segment in structure_segments
        if segment["label"] in {"[verse]", "[chorus]", "[bridge]"}
    ]
    candidate_start = 0
    if vocal_segments:
        lead = vocal_segments[0]
        window_center = max(float(lead["start"]) + 2.0, 0.0)
        candidate_start = int(window_center * sample_rate)
    else:
        energy = wav.abs().mean(dim=0)
        stride = max(sample_rate // 2, 1)
        scores = []
        for start in range(0, max(total_samples - prompt_samples, 1), stride):
            end = min(start + prompt_samples, total_samples)
            scores.append((float(energy[start:end].mean()), start))
        if scores:
            candidate_start = max(scores)[1]

    candidate_start = min(candidate_start, max(total_samples - prompt_samples, 0))
    return candidate_start, min(candidate_start + prompt_samples, total_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--prompt-len", type=float, default=10.0, help="参考音频时长(秒)")
    parser.add_argument("--structure-json", type=str, default="", help="可选，样本级结构 JSON 映射")
    args = parser.parse_args()

    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith((".wav", ".flac", ".mp3"))]
    structure_map = _load_structure_map(args.structure_json or None)

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

        start, end = choose_prompt_window(
            wav=wav,
            sample_rate=args.sr,
            prompt_len=args.prompt_len,
            structure_segments=structure_map.get(stem),
        )
        prompt = wav[:, start:end]
        torchaudio.save(out_path, prompt, args.sr)


if __name__ == "__main__":
    main()
