
# === CELL 2 ===
from __future__ import annotations

import csv
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_WORKING = Path('/kaggle/working')
KAGGLE_INPUT = Path('/kaggle/input')

# Repo source (project code)
SOURCE_DATASET_CANDIDATES = [
    Path('/kaggle/input/datasets/benedictusryugunawan/f5-tts-mamba-tts-repo'),
    Path('/kaggle/input/f5-tts-mamba-tts-repo'),
    Path('/kaggle/input/f5-mamba-dit-tts'),
]

# Raw training data source
RAW_DATASET_CANDIDATES = [
    Path('/kaggle/input/datasets/benedictusryugunawan/tts-indo/data'),
    Path('/kaggle/input/indonesian-voice-dataset/datasetku'),
    Path('/kaggle/input/datasetku'),
]

# Mapping tegas: metadata tertentu hanya boleh memakai folder wav tertentu.
METADATA_WAV_MAPPING = {
    'metadata_indsp.csv': 'indsp',
    'metadata.csv': 'wavs',
    'metda_data.csv': 'wavs',
    'meta_data.csv': 'wavs',
}
METADATA_CANDIDATES = list(METADATA_WAV_MAPPING.keys())
WAVS_DIR_CANDIDATES = ['indsp', 'wavs']


def find_repo_root(base: Path) -> Path | None:
    if not base.exists():
        return None
    if (base / 'pyproject.toml').exists() and (base / 'src' / 'f5_tts').exists():
        return base
    for child in base.iterdir():
        if child.is_dir() and (child / 'pyproject.toml').exists() and (child / 'src' / 'f5_tts').exists():
            return child
    return None


def search_repo_under_kaggle_input() -> Path | None:
    if not KAGGLE_INPUT.exists():
        return None
    for pyproject in KAGGLE_INPUT.rglob('pyproject.toml'):
        repo = pyproject.parent
        if (repo / 'src' / 'f5_tts').exists():
            return repo
    return None


def get_existing_metadata_wav_pairs(data_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for metadata_name in METADATA_CANDIDATES:
        wav_dir_name = METADATA_WAV_MAPPING[metadata_name]
        meta_path = data_root / metadata_name
        wav_dir = data_root / wav_dir_name
        if meta_path.exists() and wav_dir.exists():
            pairs.append((meta_path, wav_dir))
    return pairs


def validate_raw_layout(data_root: Path) -> tuple[Path | None, list[Path]]:
    pairs = get_existing_metadata_wav_pairs(data_root)
    if not pairs:
        return None, []

    primary_metadata = pairs[0][0]
    wav_dirs: list[Path] = []
    seen = set()
    for _, wav_dir in pairs:
        wav_dir_str = wav_dir.as_posix()
        if wav_dir_str in seen:
            continue
        seen.add(wav_dir_str)
        wav_dirs.append(wav_dir)

    return primary_metadata, wav_dirs


def resolve_raw_data_root() -> Path | None:
    for candidate in RAW_DATASET_CANDIDATES:
        meta, wav_dirs = validate_raw_layout(candidate)
        if meta is not None and wav_dirs:
            return candidate

    # Fallback: scan /kaggle/input for any supported wav dir
    if KAGGLE_INPUT.exists():
        for wav_dir_name in WAVS_DIR_CANDIDATES:
            for p in KAGGLE_INPUT.rglob(wav_dir_name):
                root = p.parent
                meta, wav_dirs = validate_raw_layout(root)
                if meta is not None and wav_dirs:
                    return root

    return None


SOURCE_REPO_ROOT = None
for candidate in SOURCE_DATASET_CANDIDATES:
    found = find_repo_root(candidate)
    if found is not None:
        SOURCE_REPO_ROOT = found
        break

if SOURCE_REPO_ROOT is None:
    SOURCE_REPO_ROOT = search_repo_under_kaggle_input()

RAW_DATA_ROOT = resolve_raw_data_root()

# Kita kerja di /kaggle/working karena /kaggle/input read-only
REPO_ROOT = KAGGLE_WORKING / 'F5-TTS'

DATA_ROOT = RAW_DATA_ROOT  # akan dipakai untuk metadata + wavs mentah
RAW_METADATA = None
RAW_WAV_DIRS: list[Path] = []

if RAW_DATA_ROOT is not None:
    RAW_METADATA, RAW_WAV_DIRS = validate_raw_layout(RAW_DATA_ROOT)

DATASET_NAME = 'indonesian_voice'
TOKENIZER = 'char'
PREPARED_DATA_DIR = REPO_ROOT / 'data' / f'{DATASET_NAME}_{TOKENIZER}'
ABS_METADATA = REPO_ROOT / 'data' / 'metadata_abs.csv'

print('SOURCE_REPO_ROOT    :', SOURCE_REPO_ROOT)
print('RAW_DATA_ROOT       :', RAW_DATA_ROOT)
print('RAW_METADATA        :', RAW_METADATA)
print('RAW_WAV_DIRS        :', [p.as_posix() for p in RAW_WAV_DIRS])
print('REPO_ROOT (working) :', REPO_ROOT)
print('METADATA->WAV map   :', METADATA_WAV_MAPPING)
print('OUTPUT DATASET      :', PREPARED_DATA_DIR)

# === CELL 3 ===
import torch
print(torch.__version__)


import sys
print(f'Python: {sys.version}')
!python -V
!nvcc --version
print("cxx11abi:", torch._C._GLIBCXX_USE_CXX11_ABI)

# === CELL 4 ===
if SOURCE_REPO_ROOT is None:
    raise FileNotFoundError(
        'Source repo tidak ditemukan di /kaggle/input. '
        'Pastikan dataset repo kamu ter-mount dan berisi pyproject.toml + src/f5_tts.'
    )

if not REPO_ROOT.exists():
    shutil.copytree(SOURCE_REPO_ROOT, REPO_ROOT)
    print('Repo copied to writable path:', REPO_ROOT)
else:
    print('Repo already available at:', REPO_ROOT)

if not (REPO_ROOT / 'pyproject.toml').exists():
    raise FileNotFoundError(
        f'pyproject.toml tidak ditemukan di {REPO_ROOT}. Pastikan ini root project F5-TTS.'
    )

# Fallback: cari dataset di dalam repo itu sendiri jika tidak ditemukan di /kaggle/input
if DATA_ROOT is None:
    candidate_roots = [
        REPO_ROOT / 'data',
        SOURCE_REPO_ROOT / 'data',
    ]
    for root in candidate_roots:
        meta, wav_dirs = validate_raw_layout(root)
        if meta is not None and wav_dirs:
            DATA_ROOT = root
            RAW_METADATA = meta
            RAW_WAV_DIRS = wav_dirs
            break

if DATA_ROOT is None or RAW_METADATA is None or not RAW_WAV_DIRS:
    raise FileNotFoundError(
        'Raw dataset tidak ditemukan. Pastikan path berikut berisi metadata + indsp/ atau wavs/:\n'
        '- /kaggle/input/datasets/anekazek/indonesian-voice-dataset/datasetku\n'
        '- /kaggle/input/indonesian-voice-dataset/datasetku'
    )

os.chdir(REPO_ROOT)
print('Working directory:', Path.cwd())
print('DATA_ROOT     :', DATA_ROOT)
print('RAW_METADATA  :', RAW_METADATA)
print('RAW_WAV_DIRS  :', [p.as_posix() for p in RAW_WAV_DIRS])

# === CELL 5 ===
print('Step 1.5 - Preflight checklist (quick fail-fast)')
print('Checklist ini memverifikasi prasyarat minimum sebelum Step 2.')
print()

def _check(name: str, ok: bool, detail: str, critical: bool = False) -> tuple[bool, bool]:
    status = 'PASS' if ok else ('FAIL' if critical else 'WARN')
    print(f'[{status}] {name}: {detail}')
    return ok, critical

results: list[tuple[bool, bool]] = []

results.append(_check(
    'Kaggle runtime paths',
    KAGGLE_WORKING.exists() and KAGGLE_INPUT.exists(),
    f'working={KAGGLE_WORKING.exists()} input={KAGGLE_INPUT.exists()}',
    critical=True,
))

results.append(_check(
    'Source repo detected',
    SOURCE_REPO_ROOT is not None and SOURCE_REPO_ROOT.exists(),
    str(SOURCE_REPO_ROOT),
    critical=True,
))

results.append(_check(
    'Raw data root detected',
    DATA_ROOT is not None and DATA_ROOT.exists(),
    str(DATA_ROOT),
    critical=True,
))

results.append(_check(
    'Raw metadata detected',
    RAW_METADATA is not None and RAW_METADATA.exists(),
    str(RAW_METADATA),
    critical=True,
))

results.append(_check(
    'Raw wav directories detected',
    len(RAW_WAV_DIRS) > 0 and all(p.exists() for p in RAW_WAV_DIRS),
    str([p.as_posix() for p in RAW_WAV_DIRS]),
    critical=True,
))

results.append(_check(
    'Working directory writable',
    REPO_ROOT.parent.exists() and os.access(REPO_ROOT.parent, os.W_OK),
    str(REPO_ROOT.parent),
    critical=True,
))

results.append(_check(
    'Python version',
    sys.version_info >= (3, 10),
    f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
    critical=True,
))

gpu_available = False
gpu_count = 0
try:
    import torch  # local import supaya preflight tetap aman jika torch belum ter-load
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
except Exception as exc:
    results.append(_check('Torch import', False, repr(exc), critical=False))
else:
    results.append(_check(
        'GPU visibility',
        gpu_available,
        f'cuda_available={gpu_available} gpu_count={gpu_count}',
        critical=False,
    ))

critical_failures = sum(1 for ok, critical in results if critical and not ok)
warnings = sum(1 for ok, critical in results if (not ok) and (not critical))

print()
print('Preflight summary:', f'critical_failures={critical_failures}', f'warnings={warnings}')
if critical_failures > 0:
    raise RuntimeError(
        'Preflight gagal: ada prasyarat kritis yang belum terpenuhi. '
        'Perbaiki item [FAIL] sebelum lanjut ke Step 2.'
    )
print('Preflight OK: aman lanjut ke Step 2.')

# === CELL 7 ===
def run(cmd: list[str], cwd=None, extra_env=None):
    print('+', ' '.join(shlex.quote(c) for c in cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, check=True, cwd=str(cwd or REPO_ROOT), env=env)

run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip'])

# ── 1. Verifikasi PyTorch bawaan Kaggle ───────────────────────────────────────
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.version.cuda}')
print(f'GPU     : {torch.cuda.is_available()}')

# ── 2. Deps biasa ─────────────────────────────────────────────────────────────
common_deps = [
    'accelerate>=1.13.0', 'datasets', 'hydra-core>=1.3.0',
    'safetensors', 'cached_path', 'pydub', 'soundfile', 'librosa',
    'pypinyin', 'rjieba', 'ema_pytorch>=0.5.2', 'torchdiffeq',
    'transformers', 'transformers_stream_generator', 'vocos',
    'x_transformers>=1.31.14', 'wandb', 'click', 'tqdm',
]
run([sys.executable, '-m', 'pip', 'install', '-q', *common_deps])

# ── 3. Install causal_conv1d + mamba_ssm dari wheel lokal ────────────────────
import re, shutil, importlib.util

WHEEL_DIR     = Path('/kaggle/input/datasets/benedictusryugunawan/mamba-tts-wheel')
TMP_WHEEL_DIR = Path('/kaggle/working/wheels_fixed')
TMP_WHEEL_DIR.mkdir(parents=True, exist_ok=True)

def fix_wheel_filename(src: Path, dst_dir: Path) -> Path:
    name  = src.name
    fixed = re.sub(r'-(\d+\.\d+[\.\d]*)(cu\d)', r'-\1+\2', name)
    dst   = dst_dir / fixed
    shutil.copy2(src, dst)
    if name != fixed:
        print(f'  Renamed : {name}\n        → : {fixed}')
    else:
        print(f'  Nama OK : {fixed}')
    return dst

causal_wheels = sorted(WHEEL_DIR.glob('causal_conv1d-*.whl'))
mamba_wheels  = sorted(WHEEL_DIR.glob('mamba_ssm-*.whl'))

if not causal_wheels:
    raise FileNotFoundError(f'causal_conv1d wheel tidak ditemukan di {WHEEL_DIR}')
if not mamba_wheels:
    raise FileNotFoundError(f'mamba_ssm wheel tidak ditemukan di {WHEEL_DIR}')

print('Fixing & installing causal_conv1d ...')
causal_fixed = fix_wheel_filename(causal_wheels[0], TMP_WHEEL_DIR)
run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', str(causal_fixed)])
print('✓ causal_conv1d OK')

print('Fixing & installing mamba_ssm ...')
mamba_fixed = fix_wheel_filename(mamba_wheels[0], TMP_WHEEL_DIR)
run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', str(mamba_fixed)])
print('✓ mamba_ssm OK')

# ── 4. Clone mamba main branch untuk mendapatkan mamba_ssm.modules.mamba3 ────
# mamba_ssm 2.3.1 (stable release) belum punya modul mamba3.
# Modul ini hanya ada di main branch. Kita clone dan prepend ke sys.path
# agar import mamba_ssm.modules.mamba3 diarahkan ke source terbaru.
MAMBA_SRC = Path('/kaggle/working/mamba-src')
if not MAMBA_SRC.exists():
    print('Cloning state-spaces/mamba main branch ...')
    run([
        'git', 'clone', '--depth', '1',
        'https://github.com/state-spaces/mamba.git',
        str(MAMBA_SRC),
    ])
else:
    print(f'mamba-src sudah ada: {MAMBA_SRC}')

# Verifikasi mamba3.py ada di clone
mamba3_file = MAMBA_SRC / 'mamba_ssm' / 'modules' / 'mamba3.py'
if not mamba3_file.exists():
    raise FileNotFoundError(
        f'mamba_ssm/modules/mamba3.py tidak ditemukan di clone.\n'
        f'Cek isi: {MAMBA_SRC / "mamba_ssm" / "modules"}'
    )
print(f'✓ mamba3.py ditemukan: {mamba3_file}')

# Prepend ke sys.path agar override installed mamba_ssm yang tidak punya mamba3
mamba_src_str = str(MAMBA_SRC)
if mamba_src_str in sys.path:
    sys.path.remove(mamba_src_str)
sys.path.insert(0, mamba_src_str)
os.environ['PYTHONPATH'] = mamba_src_str + os.pathsep + os.environ.get('PYTHONPATH', '')
print(f'✓ mamba-src ditambahkan ke sys.path (posisi 0)')

# ── 5. Verifikasi semua import ────────────────────────────────────────────────
for pkg in ('torch', 'causal_conv1d', 'mamba_ssm'):
    ok = importlib.util.find_spec(pkg) is not None
    print(f"  {'✓' if ok else '✗ GAGAL'} {pkg}")

# Test spesifik: apakah mamba3 module bisa diimport
try:
    from mamba_ssm.modules.mamba3 import Mamba3
    print(f'  ✓ mamba_ssm.modules.mamba3.Mamba3')
except ImportError as e:
    raise RuntimeError(f'mamba3 masih tidak bisa diimport: {e}')

# ── 6. Verifikasi Mamba3 impl via subprocess ──────────────────────────────────
src_path = str(REPO_ROOT / 'src')
verify_script = f"""
import sys
sys.path.insert(0, '{mamba_src_str}')
sys.path.insert(1, '{src_path}')
import torch
from f5_tts.model.backbones import mamba3 as mamba3_mod
print('torch :', torch.__version__)
print('CUDA  :', torch.version.cuda)
impl = mamba3_mod._resolve_mamba3_impl()
print('Resolved Mamba3 impl:', impl.__module__ + '.' + impl.__name__)
if impl.__name__ == '_FallbackMamba3':
    raise RuntimeError('Fallback terdeteksi! mamba_ssm belum aktif.')
print('OK: Real Mamba3 aktif.')
"""
run([sys.executable, '-c', verify_script])

# ── 7. Set PYTHONPATH ─────────────────────────────────────────────────────────
if src_path not in sys.path:
    sys.path.insert(0, src_path)
prev = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = mamba_src_str + os.pathsep + src_path + (os.pathsep + prev if prev else '')
os.environ['F5_TTS_REQUIRE_MAMBA3'] = '1'

print()
print('Dependency setup selesai (STRICT Mamba3 mode).')
print('PYTHONPATH:', os.environ['PYTHONPATH'])

# === CELL 8 ===
import sys
print(f'Python: {sys.version}')

# === CELL 10 ===
import torch
import accelerate

print('torch version      :', torch.__version__)
print('torch cuda version :', torch.version.cuda)
print('accelerate version :', accelerate.__version__)
print('cuda available     :', torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError('GPU tidak terdeteksi. Aktifkan GPU Accelerator di Kaggle settings.')

gpu_count = torch.cuda.device_count()
print('gpu count          :', gpu_count)
for i in range(gpu_count):
    print(f'  gpu[{i}]          :', torch.cuda.get_device_name(i))

if gpu_count < 2:
    print('WARNING: GPU terdeteksi < 2. Multi-GPU T4x2 tidak aktif di sesi ini.')
else:
    print('OK: Multi-GPU siap dipakai.')


# === CELL 12 ===
import importlib

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('WANDB_SILENT', 'true')
os.environ.setdefault('WANDB__SERVICE_WAIT', '300')
os.environ.setdefault('WANDB_DIR', str(REPO_ROOT / 'wandb'))

WANDB_ENABLED = False

try:
    wandb = importlib.import_module('wandb')
    kaggle_secrets = importlib.import_module('kaggle_secrets')
    UserSecretsClient = getattr(kaggle_secrets, 'UserSecretsClient')

    secret_client = UserSecretsClient()
    wandb_api_key = secret_client.get_secret('WANDB_API_KEY')
    if wandb_api_key:
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb.login(key=wandb_api_key, relogin=True)
        WANDB_ENABLED = True
except Exception as e:
    print('WANDB secret tidak tersedia / gagal dibaca:', e)

if not WANDB_ENABLED:
    os.environ['WANDB_MODE'] = 'offline'

print('WANDB_ENABLED:', WANDB_ENABLED)
print('WANDB_MODE   :', os.getenv('WANDB_MODE', 'online'))


# === CELL 14 ===
if DATA_ROOT is None:
    raise RuntimeError('DATA_ROOT belum terdeteksi. Jalankan Step 1 dulu.')

if not DATA_ROOT.exists():
    raise FileNotFoundError(f'DATA_ROOT tidak ada: {DATA_ROOT}')

# Mapping tegas metadata -> folder wav
rows = []
seen = set()
processed_pairs = []
missing_audio = 0
metadata_hit_counts = {}
wav_dir_hit_counts = {}

AUDIO_COL_CANDIDATES = {'audio_file', 'audio_path', 'wav_path', 'path', 'file'}
TEXT_COL_CANDIDATES = {'text', 'transcript', 'sentence', 'utterance'}

for metadata_name in METADATA_CANDIDATES:
    wav_dir_name = METADATA_WAV_MAPPING[metadata_name]
    meta_path = DATA_ROOT / metadata_name
    mapped_wav_dir = DATA_ROOT / wav_dir_name

    if not meta_path.exists():
        continue

    if not mapped_wav_dir.exists():
        print(
            f"WARNING: {metadata_name} ada, tapi folder mapping '{wav_dir_name}' tidak ada. File ini di-skip."
        )
        continue

    processed_pairs.append((meta_path.name, wav_dir_name))
    metadata_hit_counts[meta_path.name] = 0
    wav_dir_hit_counts.setdefault(wav_dir_name, 0)

    with meta_path.open('r', encoding='utf-8-sig') as _probe:
        header_line = _probe.readline()
    delimiter = '|' if header_line.count('|') >= header_line.count(',') else ','
    print(
        f"Detected metadata: {meta_path.name} -> {wav_dir_name}  (delimiter={repr(delimiter)})"
    )

    with meta_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames_raw = [((name or '').strip()) for name in (reader.fieldnames or [])]
        fieldnames = {name.lower() for name in fieldnames_raw}

        audio_col = next((c for c in AUDIO_COL_CANDIDATES if c in fieldnames), None)
        text_col = next((c for c in TEXT_COL_CANDIDATES if c in fieldnames), None)

        if audio_col is None or text_col is None:
            print(
                f"WARNING: Skipping {meta_path.name} - butuh kolom audio/text. Header terdeteksi: {fieldnames_raw}"
            )
            continue

        for raw_row in reader:
            row = {((k or '').strip().lower()): (v or '') for k, v in raw_row.items()}

            rel_audio = row.get(audio_col, '').strip().strip('"')
            text = row.get(text_col, '').strip()
            if not rel_audio or not text:
                continue

            rel_path = Path(rel_audio)
            resolved_paths = []

            if rel_path.is_absolute():
                if rel_path.exists():
                    resolved_paths.append(rel_path.resolve())
            else:
                # Jika metadata memberi prefix folder lain, tolak agar mapping tetap tegas.
                if rel_path.parts and rel_path.parts[0] in WAVS_DIR_CANDIDATES:
                    if rel_path.parts[0] != wav_dir_name:
                        missing_audio += 1
                        if missing_audio <= 5:
                            print(
                                f"WARNING: {meta_path.name} menunjuk ke folder {rel_path.parts[0]}, "
                                f"padahal wajib ke {wav_dir_name}: {rel_audio}"
                            )
                        continue
                    direct = DATA_ROOT / rel_path
                    if direct.exists():
                        resolved_paths.append(direct.resolve())

                # Hanya cari di folder mapping metadata ini.
                candidate_full = mapped_wav_dir / rel_path
                if candidate_full.exists():
                    resolved_paths.append(candidate_full.resolve())

                candidate_name = mapped_wav_dir / rel_path.name
                if candidate_name.exists():
                    resolved_paths.append(candidate_name.resolve())

            uniq_resolved = []
            seen_paths = set()
            for p in resolved_paths:
                p_posix = p.as_posix()
                if p_posix in seen_paths:
                    continue
                seen_paths.add(p_posix)
                uniq_resolved.append(p)

            if not uniq_resolved:
                missing_audio += 1
                if missing_audio <= 5:
                    print(
                        f"WARNING: file audio tidak ditemukan untuk {meta_path.name} di {wav_dir_name}: {rel_audio}"
                    )
                continue

            for abs_audio in uniq_resolved:
                item = (abs_audio.as_posix(), text)
                if item in seen:
                    continue
                seen.add(item)
                rows.append({'audio_file': item[0], 'text': item[1]})
                metadata_hit_counts[meta_path.name] += 1
                wav_dir_hit_counts[wav_dir_name] += 1

if not processed_pairs:
    raise FileNotFoundError(
        'Tidak ada pasangan metadata-folder yang valid. Pastikan mapping berikut tersedia:\n'
        '- metadata_indsp.csv -> indsp\n'
        '- metadata.csv -> wavs'
    )

if not rows:
    raise ValueError('Tidak ada data valid ditemukan dari pasangan metadata-folder terpetakan.')

ABS_METADATA.parent.mkdir(parents=True, exist_ok=True)
with ABS_METADATA.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['audio_file', 'text'], delimiter='|')
    writer.writeheader()
    writer.writerows(rows)

print('ABS_METADATA         :', ABS_METADATA)
print('Processed pairs      :', processed_pairs)
print('Rows per metadata    :', metadata_hit_counts)
print('Rows per wav dir     :', wav_dir_hit_counts)
print('Total rows written   :', len(rows))
print('Missing audio skipped:', missing_audio)

# === CELL 16 ===
workers = str(min(8, max(1, (os.cpu_count() or 2) // 2)))
prepare_cmd = [
    sys.executable,
    '-m',
    'f5_tts.train.datasets.prepare_csv_wavs',
    str(ABS_METADATA),
    str(PREPARED_DATA_DIR),
    '--pretrain',
    '--workers', workers,
]
run(prepare_cmd)

required_outputs = ['raw.arrow', 'duration.json', 'vocab.txt']
missing_outputs = [name for name in required_outputs if not (PREPARED_DATA_DIR / name).exists()]
if missing_outputs:
    raise RuntimeError(f'Prepare dataset selesai tetapi file wajib hilang: {missing_outputs}')

print()
print('Prepared files:')
for p in sorted(PREPARED_DATA_DIR.glob('*')):
    print('-', p.name, '|', p.stat().st_size, 'bytes')


# === CELL 18 ===
from f5_tts.model.dataset import load_dataset

dataset = load_dataset(DATASET_NAME, TOKENIZER)
print('Loaded dataset type:', type(dataset).__name__)
print('Dataset length     :', len(dataset))


# === CELL 20 ===
import torch

major_cc = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
MIXED_PRECISION = 'bf16' if major_cc >= 8 else 'fp16'
print('Selected mixed precision:', MIXED_PRECISION)

config_name = 'kaggle_indonesian_mamba3.yaml'
config_path = REPO_ROOT / 'src' / 'f5_tts' / 'configs' / config_name
logger_yaml = 'wandb' if WANDB_ENABLED else 'null'

config_text = f"""
hydra:
  run:
    dir: ckpts/${{model.name}}_${{model.mel_spec.mel_spec_type}}_${{model.tokenizer}}_${{datasets.name}}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}

datasets:
  name: {DATASET_NAME}
  batch_size_per_gpu: 1600
  batch_size_type: frame
  max_samples: 64
  num_workers: 2

optim:
  epochs: 50
  learning_rate: 2e-4
  num_warmup_updates: 4000
  grad_accumulation_steps: 2
  max_grad_norm: 1.0
  bnb_optimizer: False
  mixed_precision: {MIXED_PRECISION}

model:
  name: Mamba3TTS_Base
  tokenizer: {TOKENIZER}
  tokenizer_path: null
  backbone: Mamba3Backbone
  arch:
    dim: 1024
    depth: 22
    ff_mult: 2
    text_dim: 512
    text_mask_padding: True
    conv_layers: 4
    checkpoint_activations: False
    dropout: 0.0
    d_state: 128
    headdim: 64
    ngroups: 1
    rope_fraction: 0.5
    is_mimo: False
    mimo_rank: 1
    chunk_size: 64
  mel_spec:
    target_sample_rate: 24000
    n_mel_channels: 100
    hop_length: 256
    win_length: 1024
    n_fft: 1024
    mel_spec_type: vocos
  vocoder:
    is_local: False
    local_path: null

ckpts:
  logger: {logger_yaml}
  wandb_project: Mamba3-TTS
  wandb_run_name: ${{model.name}}_${{model.mel_spec.mel_spec_type}}_${{model.tokenizer}}_${{datasets.name}}
  wandb_resume_id: null
  log_samples: false
  save_per_updates: 10000
  keep_last_n_checkpoints: 3
  last_per_updates: 2000
  save_dir: ckpts/${{model.name}}_${{model.mel_spec.mel_spec_type}}_${{model.tokenizer}}_${{datasets.name}}
""".strip()

config_path.write_text(config_text, encoding='utf-8')
print('Training config written to:', config_path)
print(config_path.read_text(encoding='utf-8').splitlines()[:12])


# === CELL 22 ===
num_gpus = torch.cuda.device_count()
num_processes = 2 if num_gpus >= 2 else 1
distributed_type = 'MULTI_GPU' if num_processes > 1 else 'NO'

# Gunakan precision yang sama dengan config training agar konsisten.
mixed_precision = MIXED_PRECISION

accelerate_cfg_path = REPO_ROOT / 'accelerate_kaggle.yaml'
accelerate_cfg_text = f"""
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: {distributed_type}
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: {mixed_precision}
num_machines: 1
num_processes: {num_processes}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
""".strip()

accelerate_cfg_path.write_text(accelerate_cfg_text, encoding='utf-8')
print('Accelerate config:', accelerate_cfg_path)
print(accelerate_cfg_path.read_text(encoding='utf-8'))


# === CELL 24 ===
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

if torch.cuda.device_count() >= 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

train_cmd = [
    sys.executable,
    '-m', 'accelerate.commands.launch',
    '--config_file', str(accelerate_cfg_path),
    'src/f5_tts/train/train.py',
    '--config-name', config_name,
]
run(train_cmd)

