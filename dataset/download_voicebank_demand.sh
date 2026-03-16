#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config & CLI
# =========================
DATASETS=${1:-vb}           # all | vb | dns3 | dns4 | dns5 (poți lista multiple: "vb dns4")
shift || true

ROOT="./dataset"
VB_TO_16K="true"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --vb-16k) VB_TO_16K="true"; shift ;;
    *) echo "Arg necunoscut: $1"; exit 1 ;;
  esac
done

mkdir -p "$ROOT"
echo ">> ROOT datasets: $ROOT"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Instalează $1" >&2; exit 1; }; }

# cel puțin unul dintre curl/wget
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  echo "Instalează curl sau wget." >&2; exit 1
fi
need_cmd unzip
need_cmd rsync
need_cmd git

fetch() {
  local url="$1" out="$2"
  echo ">>> Download: $url"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -k 1M -c -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$out" "$url"
  else
    wget -O "$out" "$url"
  fi
}

resample_wavs_16k() {
  local in_dir="$1" out_dir="$2"
  mkdir -p "$out_dir"
  echo ">>> Resample la 16 kHz din $in_dir -> $out_dir"
  if command -v sox >/dev/null 2>&1; then
    find "$in_dir" -type f -iname '*.wav' | while read -r f; do
      rel="${f#$in_dir/}"
      out="$out_dir/$rel"
      mkdir -p "$(dirname "$out")"
      sox "$f" -r 16000 -b 16 -c 1 "$out" dither 2>/dev/null
    done
  elif command -v ffmpeg >/dev/null 2>&1; then
    find "$in_dir" -type f -iname '*.wav' | while read -r f; do
      rel="${f#$in_dir/}"
      out="$out_dir/$rel"
      mkdir -p "$(dirname "$out")"
      ffmpeg -nostdin -loglevel error -y -i "$f" -ar 16000 -ac 1 "$out"
    done
  else
    echo "Instalează sox sau ffmpeg pentru resampling." >&2; exit 1
  fi
}

# =========================
# VoiceBank+DEMAND (VB-DEMAND)
# =========================
dl_vb() {
  local vb_root="$ROOT/voicebank-demand"
  local raw="$vb_root/raw_48k"
  mkdir -p "$raw"

  echo ">>> VoiceBank+DEMAND: download zips (48 kHz, perechi oficiale train/test 28 spk)"
  # Linkurile oficiale de la Edinburgh DataShare (pot necesita redirect; păstrăm numele fișierelor)
  fetch "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?isAllowed=y&sequence=2" \
        "$raw/clean_trainset_28spk_wav.zip"
  fetch "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip?isAllowed=y&sequence=6" \
        "$raw/noisy_trainset_28spk_wav.zip"
  fetch "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?isAllowed=y&sequence=1" \
        "$raw/clean_testset_wav.zip"
  fetch "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip?isAllowed=y&sequence=5" \
        "$raw/noisy_testset_wav.zip"
  # (opțional) transcrieri:
  fetch "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/trainset_28spk_txt.zip?isAllowed=y&sequence=9" \
        "$raw/trainset_28spk_txt.zip" || true
  fetch "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/testset_txt.zip?isAllowed=y&sequence=8" \
        "$raw/testset_txt.zip" || true

  echo ">>> Extract..."
  unzip -n "$raw/clean_trainset_28spk_wav.zip" -d "$raw" >/dev/null
  unzip -n "$raw/noisy_trainset_28spk_wav.zip" -d "$raw" >/dev/null
  unzip -n "$raw/clean_testset_wav.zip" -d "$raw" >/dev/null
  unzip -n "$raw/noisy_testset_wav.zip" -d "$raw" >/dev/null
  [ -f "$raw/trainset_28spk_txt.zip" ] && unzip -n "$raw/trainset_28spk_txt.zip" -d "$raw" >/dev/null || true
  [ -f "$raw/testset_txt.zip" ] && unzip -n "$raw/testset_txt.zip" -d "$raw" >/dev/null || true

  # Structură standard 48k
  mkdir -p "$vb_root/48k"
  rsync -a "$raw/clean_trainset_28spk_wav/" "$vb_root/48k/clean_train/"
  rsync -a "$raw/noisy_trainset_28spk_wav/" "$vb_root/48k/noisy_train/"
  rsync -a "$raw/clean_testset_wav/"      "$vb_root/48k/clean_test/"
  rsync -a "$raw/noisy_testset_wav/"      "$vb_root/48k/noisy_test/"

  if [[ "$VB_TO_16K" == "true" ]]; then
    echo ">>> Resample VB-DEMAND la 16 kHz (compatibil VB-DEMAND literature @ 16k)"
    resample_wavs_16k "$vb_root/48k/clean_train" "$vb_root/16k/clean_train"
    resample_wavs_16k "$vb_root/48k/noisy_train" "$vb_root/16k/noisy_train"
    resample_wavs_16k "$vb_root/48k/clean_test"  "$vb_root/16k/clean_test"
    resample_wavs_16k "$vb_root/48k/noisy_test"  "$vb_root/16k/noisy_test"
  else
    echo ">>> Sare peste resampling (folosești 48 kHz)."
  fi

  echo ">>> VB-DEMAND gata în: $vb_root"
}

# =========================
# DNS Challenge v3 / v4 / v5 (single-channel tracks)
# =========================
dl_dns_repo_and_hint() {
  local dns_root="$ROOT/dns_challenge"
  mkdir -p "$dns_root"
  echo ">>> Clone DNS-Challenge (Microsoft) cu scripturile oficiale de download (v1..v5)"
  if [ ! -d "$dns_root/DNS-Challenge" ]; then
    git clone --depth=1 https://github.com/microsoft/DNS-Challenge "$dns_root/DNS-Challenge"
  else
    (cd "$dns_root/DNS-Challenge" && git pull --ff-only || true)
  fi

  cat <<'EOF'

[INFO] DNS v3/v4/v5:
  - În $ROOT/dns_challenge/DNS-Challenge găsești scripturi:
      download-dns-challenge-3.sh
      download-dns-challenge-4.sh
      download-dns-challenge-5-*.sh
  - Implicit, fac "dry-run". Deschide scriptul dorit și urmează comentariile din capul fișierului
    (de-comentează metoda de download) apoi rulează:
      bash download-dns-challenge-3.sh
      bash download-dns-challenge-4.sh
      bash download-dns-challenge-5-headset-training.sh
      bash download-dns-challenge-5-speakerphone-training.sh
  - Atenție la spațiu pe disc — DNS-5 poate ajunge ~1 TB ne-arhivat (≈550 GB arhive). 

EOF
}

dl_dns3() { dl_dns_repo_and_hint; echo ">>> Vezi download-dns-challenge-3.sh în repo-ul Microsoft."; }
dl_dns4() { dl_dns_repo_and_hint; echo ">>> Vezi download-dns-challenge-4.sh în repo-ul Microsoft."; }
dl_dns5() { dl_dns_repo_and_hint; echo ">>> Vezi scripturile download-dns-challenge-5-*.sh (headset/speakerphone)."; }

# =========================
# Dispatch
# =========================
case "$DATASETS" in
  all)   dl_vb; dl_dns3; dl_dns4; dl_dns5 ;;
  vb)    dl_vb ;;
  dns3)  dl_dns3 ;;
  dns4)  dl_dns4 ;;
  dns5)  dl_dns5 ;;
  *)
    # Permite listă separată prin spațiu: "vb dns4"
    for d in $DATASETS; do
      case "$d" in
        vb) dl_vb ;;
        dns3) dl_dns3 ;;
        dns4) dl_dns4 ;;
        dns5) dl_dns5 ;;
        *) echo "Dataset necunoscut: $d" >&2; exit 1 ;;
      esac
    done
  ;;
esac

echo ">> Done."
