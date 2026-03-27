from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_GEN_TEXT = "Halo, ini adalah contoh inferensi teks bahasa Indonesia menggunakan model Mamba3."
DEFAULT_MODEL = "KCVTTS_Mamba3TTS_Base"


def _sanitize_ssl_env() -> None:
    # Some environments export SSL_CERT_FILE/REQUESTS_CA_BUNDLE/CURL_CA_BUNDLE
    # pointing to deleted files. That breaks huggingface/httpx downloads.
    for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        value = os.environ.get(env_name)
        if value and not Path(value).expanduser().exists():
            print(f"[WARN] {env_name} path tidak valid, dihapus: {value}")
            os.environ.pop(env_name, None)

    # If certifi is available, prefer a valid CA bundle path.
    try:
        import certifi

        cert_path = certifi.where()
        if cert_path and Path(cert_path).exists():
            os.environ.setdefault("SSL_CERT_FILE", cert_path)
    except Exception:
        pass


def _resolve_existing_path(*candidates: Path | str | None) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            return path.resolve()
    return None


def _read_metadata_row(metadata_path: Path, row_idx: int) -> tuple[str, str]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        delimiter = "|" if "|" in first_line else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Metadata kosong: {metadata_path}")
    if row_idx < 0 or row_idx >= len(rows):
        raise IndexError(f"row_idx {row_idx} di luar range metadata (0..{len(rows) - 1})")

    row = rows[row_idx]
    audio_key = next((k for k in ("audio_file", "audio", "wav", "path") if k in row), None)
    text_key = next((k for k in ("text", "transcript", "sentence") if k in row), None)

    if audio_key is None or text_key is None:
        raise KeyError("Kolom metadata tidak valid. Wajib ada kolom audio_file/path dan text/transcript.")

    audio_value = row[audio_key].strip()
    text_value = row[text_key].strip()
    return audio_value, text_value


def _resolve_ref_audio(audio_value: str, metadata_path: Path) -> Path:
    audio_path = Path(audio_value)
    found = _resolve_existing_path(
        audio_path,
        metadata_path.parent / audio_path,
        REPO_ROOT / "data" / audio_path,
    )
    if found is None:
        raise FileNotFoundError(f"File referensi audio tidak ditemukan: {audio_value}")
    return found


def _pick_vocab_file(model_name: str, user_vocab_file: str | None) -> str:
    from omegaconf import OmegaConf

    if user_vocab_file:
        vocab_path = _resolve_existing_path(user_vocab_file)
        if vocab_path is None:
            raise FileNotFoundError(f"vocab_file tidak ditemukan: {user_vocab_file}")
        return str(vocab_path)

    cfg_path = _resolve_existing_path(REPO_ROOT / "src" / "f5_tts" / "configs" / f"{model_name}.yaml")
    if cfg_path is not None:
        cfg = OmegaConf.load(str(cfg_path))
        tokenizer = cfg.model.get("tokenizer")

        if tokenizer == "custom":
            tokenizer_path = cfg.model.get("tokenizer_path")
            if tokenizer_path:
                custom_vocab = _resolve_existing_path(tokenizer_path, cfg_path.parent / tokenizer_path)
                if custom_vocab is not None:
                    return str(custom_vocab)
        elif tokenizer in {"char", "pinyin"}:
            dataset_name = cfg.datasets.get("name")
            inferred = _resolve_existing_path(REPO_ROOT / "data" / f"{dataset_name}_{tokenizer}" / "vocab.txt")
            if inferred is not None:
                return str(inferred)

    for pattern in ("*_char/vocab.txt", "*_pinyin/vocab.txt"):
        candidates = sorted((REPO_ROOT / "data").glob(pattern))
        if candidates:
            return str(candidates[0].resolve())

    return ""


def _detect_vocos_local_path(hf_cache_dir: str | None) -> str | None:
    # 1) explicit local checkpoints folder commonly used by this repo
    local_candidates = [
        REPO_ROOT / "checkpoints" / "vocos-mel-24khz",
        REPO_ROOT.parent / "checkpoints" / "vocos-mel-24khz",
    ]
    for candidate in local_candidates:
        if (candidate / "config.yaml").exists() and (candidate / "pytorch_model.bin").exists():
            return str(candidate.resolve())

    # 2) huggingface cache snapshot folder
    cache_base = None
    if hf_cache_dir:
        cache_base = Path(hf_cache_dir)
    elif os.environ.get("HUGGINGFACE_HUB_CACHE"):
        cache_base = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    elif os.environ.get("HF_HOME"):
        cache_base = Path(os.environ["HF_HOME"]) / "hub"

    if cache_base is None:
        return None

    model_root = cache_base / "models--charactr--vocos-mel-24khz" / "snapshots"
    if not model_root.exists():
        return None

    for snapshot_dir in sorted(model_root.glob("*"), reverse=True):
        if (snapshot_dir / "config.yaml").exists() and (snapshot_dir / "pytorch_model.bin").exists():
            return str(snapshot_dir.resolve())

    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference TTS Bahasa Indonesia dengan checkpoint lokal.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Nama model config di src/f5_tts/configs.")
    parser.add_argument("--ckpt_file", type=str, default="ckpts/model_10000.pt", help="Path checkpoint .pt/.safetensors.")
    parser.add_argument("--vocab_file", type=str, default=None, help="Path vocab.txt (opsional, auto-detect jika kosong).")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv", help="Path metadata CSV.")
    parser.add_argument("--row_idx", type=int, default=0, help="Index baris metadata untuk referensi suara.")
    parser.add_argument("--ref_audio", type=str, default=None, help="Override path audio referensi.")
    parser.add_argument("--ref_text", type=str, default=None, help="Override teks referensi audio.")
    parser.add_argument("--gen_text", type=str, default=DEFAULT_GEN_TEXT, help="Teks Indonesia yang ingin disintesis.")
    parser.add_argument("--output_dir", type=str, default="output", help="Folder output.")
    parser.add_argument("--output_file", type=str, default="hasil_inferensi_indonesia.wav", help="Nama file output wav.")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda/cpu/xpu/mps.")
    parser.add_argument("--nfe_step", type=int, default=32, help="Jumlah denoising step.")
    parser.add_argument("--cfg_strength", type=float, default=2.0, help="Classifier-free guidance.")
    parser.add_argument("--speed", type=float, default=1.0, help="Kecepatan audio output.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed inference.")
    parser.add_argument("--remove_silence", action="store_true", help="Hapus jeda panjang di output.")
    parser.add_argument("--vocoder_local_path", type=str, default=None, help="Path lokal vocoder (opsional).")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="Override HuggingFace cache dir.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _sanitize_ssl_env()

    from f5_tts.api import F5TTS

    ckpt_path = _resolve_existing_path(args.ckpt_file)
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.ckpt_file}")

    metadata_path = _resolve_existing_path(args.metadata)
    if metadata_path is None:
        raise FileNotFoundError(f"Metadata tidak ditemukan: {args.metadata}")

    if args.ref_audio:
        ref_audio_path = _resolve_existing_path(args.ref_audio)
        if ref_audio_path is None:
            raise FileNotFoundError(f"ref_audio tidak ditemukan: {args.ref_audio}")
    else:
        metadata_audio, _ = _read_metadata_row(metadata_path, args.row_idx)
        ref_audio_path = _resolve_ref_audio(metadata_audio, metadata_path)

    if args.ref_text:
        ref_text = args.ref_text
    else:
        _, metadata_text = _read_metadata_row(metadata_path, args.row_idx)
        ref_text = metadata_text

    vocab_file = _pick_vocab_file(args.model, args.vocab_file)
    if vocab_file:
        print(f"[INFO] vocab_file: {vocab_file}")
    else:
        print("[WARN] vocab_file tidak ditemukan otomatis. Inference mungkin gagal jika vocab mismatch.")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    vocoder_local_path = None
    if args.vocoder_local_path:
        resolved_vocoder_path = _resolve_existing_path(args.vocoder_local_path)
        if resolved_vocoder_path is None:
            raise FileNotFoundError(f"vocoder_local_path tidak ditemukan: {args.vocoder_local_path}")
        vocoder_local_path = str(resolved_vocoder_path)
    else:
        vocoder_local_path = _detect_vocos_local_path(args.hf_cache_dir)
        if vocoder_local_path:
            print(f"[INFO] Pakai vocoder lokal/cache: {vocoder_local_path}")

    tts = F5TTS(
        model=args.model,
        ckpt_file=str(ckpt_path),
        vocab_file=vocab_file,
        vocoder_local_path=vocoder_local_path,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
    )

    tts.infer(
        ref_file=str(ref_audio_path),
        ref_text=ref_text,
        gen_text=args.gen_text,
        file_wave=str(output_path),
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        speed=args.speed,
        seed=args.seed,
        remove_silence=args.remove_silence,
    )

    print(f"[OK] Output tersimpan di: {output_path}")


if __name__ == "__main__":
    main()
