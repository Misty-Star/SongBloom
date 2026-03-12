"""SongBloom 训练预处理总管线。

输入：原始歌曲 manifest（至少包含 `id`/`idx`、`audio_path`、`lyrics_raw` 或 `lyrics`）
输出：`training/dataset.py` 可直接读取的样本目录。
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import traceback
import typing as tp

import torch
import torch.nn.functional as F
import torchaudio

from .align_lyrics import process_item as process_alignment_item
from .common import (
    command_exists,
    ensure_dir,
    get_audio_path,
    get_item_id,
    get_raw_lyrics,
    load_jsonl,
    merge_adjacent_segments,
    save_json,
    split_sentences,
    write_jsonl,
)
from .extract_prompt import choose_prompt_window
from .extract_structure import process_item as process_structure_item


def convert_audio(
    input_path: str,
    output_path: str,
    sample_rate: int,
    mono: bool = False,
) -> tp.Tuple[str, float]:
    wav, sr = torchaudio.load(input_path)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif not mono and wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    duration = wav.shape[-1] / sample_rate
    ensure_dir(os.path.dirname(output_path))
    torchaudio.save(output_path, wav, sample_rate)
    return output_path, duration


def run_demucs(
    audio_path: str,
    output_dir: str,
    demucs_cmd: str,
    model_name: str,
) -> tp.Tuple[str, str]:
    command = shlex.split(demucs_cmd)
    if not command:
        raise ValueError("`demucs_cmd` is empty.")
    argv = list(command) + ["-n", model_name, "--two-stems=vocals", "-o", output_dir, audio_path]
    try:
        subprocess.run(argv, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Demucs command not found: {command[0]!r}. "
            "Install demucs, set --demucs-cmd to the correct executable, "
            "or run prepare_assets first so the manifest already contains vocals_path/no_vocals_path."
        ) from exc

    stem = os.path.splitext(os.path.basename(audio_path))[0]
    model_dir = os.path.join(output_dir, model_name, stem)
    vocals_path = None
    accompaniment_path = None
    for root, _dirs, files in os.walk(model_dir):
        for name in files:
            lower = name.lower()
            path = os.path.join(root, name)
            if lower.startswith("vocals.") or lower == "vocals.wav":
                vocals_path = path
            elif "no_vocals" in lower:
                accompaniment_path = path
    if not vocals_path or not accompaniment_path:
        raise FileNotFoundError(f"Demucs stems not found under {model_dir}")
    return vocals_path, accompaniment_path


def allocate_sentences_to_segments(sentences: tp.Sequence[str], segments: tp.Sequence[dict]) -> tp.List[tp.List[str]]:
    vocal_segments = [seg for seg in segments if seg["label"] in {"[verse]", "[chorus]", "[bridge]"}]
    allocations = [[] for _ in segments]
    if not vocal_segments:
        return allocations
    if not sentences:
        for idx, seg in enumerate(segments):
            if seg["label"] in {"[verse]", "[chorus]", "[bridge]"}:
                allocations[idx] = []
        return allocations

    vocal_indices = [idx for idx, seg in enumerate(segments) if seg["label"] in {"[verse]", "[chorus]", "[bridge]"}]
    durations = [max(seg["end"] - seg["start"], 1e-6) for seg in vocal_segments]
    total_duration = sum(durations)
    desired = [max(1, int(round(len(sentences) * dur / total_duration))) for dur in durations]

    while sum(desired) > len(sentences):
        largest = max(range(len(desired)), key=lambda idx: desired[idx])
        if desired[largest] > 1:
            desired[largest] -= 1
        else:
            break
    while sum(desired) < len(sentences):
        largest = max(range(len(durations)), key=lambda idx: durations[idx])
        desired[largest] += 1

    cursor = 0
    for local_index, segment_index in enumerate(vocal_indices):
        next_cursor = min(cursor + desired[local_index], len(sentences))
        allocations[segment_index] = list(sentences[cursor:next_cursor])
        cursor = next_cursor

    if cursor < len(sentences):
        allocations[vocal_indices[-1]].extend(sentences[cursor:])
    return allocations


def build_lyrics_and_structure(
    cleaned_lyrics: str,
    structure_payload: dict,
) -> tp.Tuple[str, tp.List[tp.List[tp.Union[str, float]]]]:
    segments = merge_adjacent_segments(structure_payload.get("segments", []))
    sentences = split_sentences(cleaned_lyrics)
    sentence_allocations = allocate_sentences_to_segments(sentences, segments)

    parts = []
    structure_duration = []
    for idx, segment in enumerate(segments):
        label = segment["label"]
        start = round(float(segment["start"]), 3)
        end = round(float(segment["end"]), 3)
        structure_duration.append([label, start, end])

        if label in {"[verse]", "[chorus]", "[bridge]"}:
            assigned = sentence_allocations[idx]
            if not assigned:
                assigned = [""]
            text = ". ".join(item.strip(" .") for item in assigned if item.strip())
            text = text.strip()
            parts.append(f"{label} {text}".strip())
        else:
            parts.append(label)

    lyrics = " , ".join(part.strip() for part in parts if part.strip())
    return lyrics, structure_duration


def save_prompt_audio(
    audio_path: str,
    output_path: str,
    sample_rate: int,
    prompt_len: float,
    structure_segments: tp.Sequence[dict],
    prompt_path: tp.Optional[str] = None,
    prompt_start_sec: tp.Optional[float] = None,
) -> None:
    wav, sr = torchaudio.load(prompt_path or audio_path)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if prompt_start_sec is not None:
        start = max(int(float(prompt_start_sec) * sample_rate), 0)
        end = min(start + int(prompt_len * sample_rate), wav.shape[-1])
    else:
        start, end = choose_prompt_window(
            wav=wav,
            sample_rate=sample_rate,
            prompt_len=prompt_len,
            structure_segments=structure_segments,
        )
    torchaudio.save(output_path, wav[:, start:end], sample_rate)


class FeatureExtractorBundle:
    def __init__(self, args):
        self.args = args
        self._vae = None
        self._muq = None
        self._vq = None

    @property
    def vae(self):
        if self._vae is None:
            from SongBloom.models.vae_frontend import StableVAE

            self._vae = StableVAE(
                vae_cfg=self.args.vae_cfg,
                vae_ckpt=self.args.vae_ckpt,
                sr=self.args.sample_rate,
            ).eval().to(self.args.device)
        return self._vae

    @property
    def muq(self):
        if self._muq is None:
            from .extract_sketch import load_muq_model

            self._muq = load_muq_model(self.args.muq_model, self.args.device)
        return self._muq

    @property
    def vq(self):
        if self._vq is None:
            from .extract_sketch import SingleLayerVQ

            self._vq = SingleLayerVQ.from_pretrained(self.args.vq_ckpt).eval().to(self.args.device)
        return self._vq

    def extract_latent(self, audio_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(audio_path)
        if sr != self.args.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.args.sample_rate)(wav)
        with torch.no_grad():
            latent = self.vae.encode(wav.unsqueeze(0).to(self.args.device))
        return latent.squeeze(0).cpu()

    def extract_sketch(self, audio_path: str) -> torch.Tensor:
        from .extract_sketch import extract_muq_embeddings

        wav, sr = torchaudio.load(audio_path)
        if sr != self.args.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.args.sample_rate)(wav)
        embeddings = extract_muq_embeddings(self.muq, wav, self.args.sample_rate)
        target_frames = int(wav.shape[-1] / self.args.sample_rate * self.args.target_fps)
        if embeddings.shape[1] != target_frames:
            embeddings = F.interpolate(
                embeddings.transpose(1, 2),
                size=target_frames,
                mode="nearest",
            ).transpose(1, 2)
        with torch.no_grad():
            sketch_tokens = self.vq.encode(embeddings)
        return sketch_tokens.squeeze(0).cpu()


def resolve_stems(item: dict, audio_path: str, work_dir: str, args) -> tp.Tuple[str, str]:
    if item.get("vocals_path") and item.get("no_vocals_path"):
        return str(item["vocals_path"]), str(item["no_vocals_path"])
    if args.skip_demucs:
        return audio_path, audio_path
    demucs_dir = ensure_dir(os.path.join(work_dir, "demucs"))
    return run_demucs(
        audio_path=audio_path,
        output_dir=demucs_dir,
        demucs_cmd=args.demucs_cmd,
        model_name=args.demucs_model,
    )


def _resolve_command_name(command: str) -> str:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("command must not be empty")
    return argv[0]


def collect_preflight_issues(items: tp.Sequence[dict], args) -> tp.List[str]:
    issues: tp.List[str] = []

    needs_demucs = not args.skip_demucs and any(
        not (item.get("vocals_path") and item.get("no_vocals_path")) for item in items
    )
    if needs_demucs:
        demucs_exec = _resolve_command_name(args.demucs_cmd)
        if not command_exists(demucs_exec):
            issues.append(
                f"Demucs executable {demucs_exec!r} is not in PATH. "
                "Install demucs, pass --demucs-cmd, or run prepare_assets first to enrich the manifest with stems."
            )

    needs_whisperx = any(not item.get("whisperx_json") for item in items)
    if needs_whisperx:
        whisperx_exec = _resolve_command_name(args.whisperx_cmd)
        if not command_exists(whisperx_exec):
            issues.append(
                f"WhisperX executable {whisperx_exec!r} is not in PATH. "
                "Install whisperx, pass --whisperx-cmd, or run prepare_assets first to enrich the manifest with whisperx_json."
            )

    needs_songformer = any(not item.get("structure_json") for item in items)
    if needs_songformer:
        songformer_exec = _resolve_command_name(args.songformer_python)
        infer_path = os.path.join(args.songformer_root, "src", "SongFormer", "infer", "infer.py")
        if not command_exists(songformer_exec):
            issues.append(
                f"SongFormer python launcher {songformer_exec!r} is not in PATH. "
                "Fix --songformer-python before running dataset preprocessing."
            )
        if not os.path.exists(infer_path):
            issues.append(
                f"SongFormer infer script is missing: {infer_path}. "
                "Check --songformer-root or initialize the third_party/SongFormer dependency."
            )

    return issues


def write_processed_lyrics(output_path: str, lyrics: str, lyric_processor: str) -> str:
    from SongBloom.g2p.lyric_common import process_lyric_preserve_labels

    processed = process_lyric_preserve_labels(lyrics, lyric_processor)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(processed)
    return processed


def process_sample(item: dict, args, features: FeatureExtractorBundle) -> dict:
    sample_id = get_item_id(item)
    output_sample_dir = os.path.join(args.output_dir, sample_id)
    work_sample_dir = os.path.join(args.workspace_dir, sample_id)
    ensure_dir(output_sample_dir)
    ensure_dir(work_sample_dir)

    if args.skip_existing and os.path.exists(os.path.join(output_sample_dir, "meta.json")):
        return {"id": sample_id, "status": "skipped", "output_dir": output_sample_dir}

    standardized_audio, duration = convert_audio(
        input_path=get_audio_path(item),
        output_path=os.path.join(work_sample_dir, "full_audio.flac"),
        sample_rate=args.sample_rate,
        mono=False,
    )
    if duration < args.min_duration or duration > args.max_duration:
        raise ValueError(f"duration {duration:.2f}s outside [{args.min_duration}, {args.max_duration}]")

    vocals_path, no_vocals_path = resolve_stems(item, standardized_audio, work_sample_dir, args)

    alignment_item = dict(item)
    alignment_item["audio_path"] = standardized_audio
    alignment_item["vocals_path"] = vocals_path
    alignment = process_alignment_item(
        item=alignment_item,
        output_dir=work_sample_dir,
        whisperx_cmd=args.whisperx_cmd,
        language=args.language,
        model=args.whisperx_model,
        device=args.whisperx_device,
        compute_type=args.whisperx_compute_type,
        similarity_threshold=args.lyrics_similarity_threshold,
        skip_existing=args.skip_existing,
    )
    if alignment["whisperx_quality"]["score"] < args.min_whisperx_score:
        raise ValueError(f"low whisperx score: {alignment['whisperx_quality']['score']:.3f}")

    structure_item = dict(item)
    structure_item["audio_path"] = standardized_audio
    structure_item["songformer_audio_path"] = standardized_audio
    structure_item["lyrics_alignment_json"] = os.path.join(work_sample_dir, sample_id, "lyrics_alignment.json")
    structure_item["duration_hint"] = duration
    structure_payload = process_structure_item(
        item=structure_item,
        output_dir=work_sample_dir,
        ignore_silence=True,
        min_duration=args.min_structure_duration,
        refine_vocals=not args.disable_vocal_refine,
        vocal_threshold=args.vocal_threshold,
        non_vocal_threshold=args.non_vocal_threshold,
        songformer_root=args.songformer_root,
        python_exec=args.songformer_python,
        gpu_num=args.songformer_gpu_num,
        num_thread_per_gpu=args.songformer_threads,
        model=args.songformer_model,
        checkpoint=args.songformer_checkpoint,
        config_path=args.songformer_config,
        no_rule_post_processing=args.songformer_no_rule_post,
        skip_existing=args.skip_existing,
    )
    if len(structure_payload["segments"]) < args.min_structure_segments:
        raise ValueError(f"too few structure segments: {len(structure_payload['segments'])}")

    structured_lyrics, structure_duration = build_lyrics_and_structure(
        cleaned_lyrics=alignment["cleaned_lyrics"] or get_raw_lyrics(item),
        structure_payload=structure_payload,
    )
    if not structured_lyrics.strip():
        raise ValueError("structured lyrics is empty")

    lyrics_out_path = os.path.join(output_sample_dir, "lyrics.txt")
    processed_lyrics = write_processed_lyrics(lyrics_out_path, structured_lyrics, args.lyric_processor)
    if not processed_lyrics.strip():
        raise ValueError("processed lyrics is empty")

    prompt_out = os.path.join(output_sample_dir, "prompt_wav.flac")
    save_prompt_audio(
        audio_path=standardized_audio,
        output_path=prompt_out,
        sample_rate=args.sample_rate,
        prompt_len=args.prompt_len,
        structure_segments=structure_payload["segments"],
        prompt_path=item.get("prompt_path"),
        prompt_start_sec=item.get("prompt_start_sec"),
    )

    x_latent = features.extract_latent(standardized_audio)
    x_sketch = features.extract_sketch(standardized_audio)
    effective_frames = min(
        x_latent.shape[-1],
        x_sketch.shape[0],
        int(duration * args.target_fps),
    )
    effective_frames = (effective_frames // args.block_size) * args.block_size
    if effective_frames <= 0:
        raise ValueError("effective frame length is zero")

    x_latent = x_latent[:, :effective_frames]
    x_sketch = x_sketch[:effective_frames]
    torch.save(x_latent, os.path.join(output_sample_dir, "x_latent.pt"))
    torch.save(x_sketch, os.path.join(output_sample_dir, "x_sketch.pt"))

    meta = {
        "duration": round(effective_frames / args.target_fps, 3),
        "structure_duration": structure_duration,
        "source_audio_path": standardized_audio,
        "vocals_path": vocals_path,
        "no_vocals_path": no_vocals_path,
        "whisperx_quality": alignment["whisperx_quality"],
        "songformer_segments": structure_payload["segments"],
    }
    save_json(os.path.join(output_sample_dir, "meta.json"), meta)

    if not args.keep_intermediate:
        shutil.rmtree(work_sample_dir, ignore_errors=True)

    return {
        "id": sample_id,
        "status": "ok",
        "output_dir": output_sample_dir,
        "duration": meta["duration"],
        "num_structure_segments": len(structure_payload["segments"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--workspace-dir", type=str, default="")
    parser.add_argument("--lyric-processor", type=str, default="phoneme", choices=["pinyin", "phoneme"])
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--target-fps", type=int, default=25)
    parser.add_argument("--min-duration", type=float, default=80.0)
    parser.add_argument("--max-duration", type=float, default=240.0)
    parser.add_argument("--prompt-len", type=float, default=10.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--min-whisperx-score", type=float, default=0.15)
    parser.add_argument("--lyrics-similarity-threshold", type=float, default=0.25)

    parser.add_argument("--skip-demucs", action="store_true")
    parser.add_argument("--demucs-cmd", type=str, default="demucs")
    parser.add_argument("--demucs-model", type=str, default="htdemucs")

    parser.add_argument("--whisperx-cmd", type=str, default="whisperx")
    parser.add_argument("--whisperx-model", type=str, default="large-v3")
    parser.add_argument("--whisperx-device", type=str, default="cuda")
    parser.add_argument("--whisperx-compute-type", type=str, default="float16")
    parser.add_argument("--language", type=str, default=None)

    parser.add_argument("--songformer-root", type=str, default="third_party/SongFormer")
    parser.add_argument("--songformer-python", type=str, default=sys.executable)
    parser.add_argument("--songformer-gpu-num", type=int, default=1)
    parser.add_argument("--songformer-threads", type=int, default=1)
    parser.add_argument("--songformer-model", type=str, default="SongFormer")
    parser.add_argument("--songformer-checkpoint", type=str, default="SongFormer.safetensors")
    parser.add_argument("--songformer-config", type=str, default="SongFormer.yaml")
    parser.add_argument("--songformer-no-rule-post", action="store_true")
    parser.add_argument("--min-structure-duration", type=float, default=1.0)
    parser.add_argument("--min-structure-segments", type=int, default=2)
    parser.add_argument("--disable-vocal-refine", action="store_true")
    parser.add_argument("--vocal-threshold", type=float, default=0.3)
    parser.add_argument("--non-vocal-threshold", type=float, default=0.05)

    parser.add_argument("--vae-cfg", type=str, default="pretrained/stable_audio_1920_vae.json")
    parser.add_argument("--vae-ckpt", type=str, default="pretrained/autoencoder_music_dsp1920.ckpt")
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--muq-model", type=str, default="OpenMuQ/MuQ-large-msd-iter")
    args = parser.parse_args()

    args.output_dir = ensure_dir(args.output_dir)
    args.workspace_dir = ensure_dir(args.workspace_dir or os.path.join(args.output_dir, "_workspace"))
    items = load_jsonl(args.input_jsonl)

    preflight_issues = collect_preflight_issues(items, args)
    if preflight_issues:
        error_message = "Preflight failed: " + " ".join(preflight_issues)
        write_jsonl(
            os.path.join(args.output_dir, "report.jsonl"),
            [{"id": get_item_id(item), "status": "error", "error": error_message} for item in items],
        )
        raise SystemExit(error_message)

    features = FeatureExtractorBundle(args)
    report_rows = []
    for item in items:
        sample_id = get_item_id(item)
        try:
            report_rows.append(process_sample(item, args, features))
        except Exception as exc:
            shutil.rmtree(os.path.join(args.output_dir, sample_id), ignore_errors=True)
            if not args.keep_intermediate:
                shutil.rmtree(os.path.join(args.workspace_dir, sample_id), ignore_errors=True)
            report_rows.append(
                {
                    "id": sample_id,
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )

    write_jsonl(os.path.join(args.output_dir, "report.jsonl"), report_rows)


if __name__ == "__main__":
    main()
