#!/usr/bin/env bash
# prepare_splits_voicebank-demand.sh
# Build CSVs with (noisy,clean) pairs for VoiceBank-DEMAND (16k) layout:
# datasets/voicebank-demand/16k/{clean_train,noisy_train,clean_test,noisy_test}
# - Uses absolute paths in the CSV
# - Safe for spaces/unusual chars
# - Warns on duplicate basenames
# - Fails if too few pairs match (90% threshold)

set -euo pipefail
export LC_ALL=C

# --- Config / Paths -----------------------------------------------------------
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
DATA_DIR="${VOICEBANK_DATA_DIR:-${SCRIPT_DIR}/voicebank-demand/16k}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

CLEAN_TRAIN="${DATA_DIR}/clean_train"
NOISY_TRAIN="${DATA_DIR}/noisy_train"
CLEAN_TEST="${DATA_DIR}/clean_test"
NOISY_TEST="${DATA_DIR}/noisy_test"

OUT_TRAIN="${DATA_DIR}/train.csv"
OUT_TEST="${DATA_DIR}/test.csv"

# --- Helpers ------------------------------------------------------------------
die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "NOTE: $*" >&2; }
warn() { echo "WARNING: $*" >&2; }

check_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

pair_split() {
  local noisy_dir="$1"
  local clean_dir="$2"
  local out_csv="$3"

  # Ensure inputs exist
  check_dir "$noisy_dir"
  check_dir "$clean_dir"

  # Create a temp dir that cleans up on exit
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf -- "'"$tmp"'"' RETURN

  local noisy_keys="${tmp}/noisy_keys.tsv"
  local clean_keys="${tmp}/clean_keys.tsv"
  local joined="${tmp}/joined.tsv"

  # Collect files (null-delimited, safe for odd names), sort by basename
  # Fields: <basename>\t<absolute_path>
  find "$noisy_dir" -type f -iname '*.wav' -print0 \
    | xargs -0 -I{} bash -c 'f="{}"; b="$(basename "${f%.*}")"; printf "%s\t%s\n" "$b" "$(readlink -f "$f")"' \
    | sort -k1,1 > "$noisy_keys"

  find "$clean_dir" -type f -iname '*.wav' -print0 \
    | xargs -0 -I{} bash -c 'f="{}"; b="$(basename "${f%.*}")"; printf "%s\t%s\n" "$b" "$(readlink -f "$f")"' \
    | sort -k1,1 > "$clean_keys"

  # Report simple stats
  local n_noisy n_clean
  n_noisy=$(wc -l < "$noisy_keys" | tr -d ' ')
  n_clean=$(wc -l < "$clean_keys" | tr -d ' ')
  echo "Found: $n_noisy noisy files in $noisy_dir"
  echo "Found: $n_clean clean files in $clean_dir"

  # Detect duplicate basenames (can cause wrong joins)
  local dupes_noisy dupes_clean
  dupes_noisy=$(cut -f1 "$noisy_keys" | sort | uniq -d | head -n 1 || true)
  dupes_clean=$(cut -f1 "$clean_keys" | sort | uniq -d | head -n 1 || true)
  if [[ -n "${dupes_noisy:-}" || -n "${dupes_clean:-}" ]]; then
    warn "Duplicate basenames detected. This may mispair files."
    [[ -n "${dupes_noisy:-}" ]] && warn "  e.g., in noisy: ${dupes_noisy}"
    [[ -n "${dupes_clean:-}" ]] && warn "  e.g., in clean: ${dupes_clean}"
  fi

  # Join by basename (inner join) to create aligned pairs
  # Output format: <noisy_abs>\t<clean_abs>
  join -t $'\t' -j 1 -o 1.2,2.2 "$noisy_keys" "$clean_keys" > "$joined" || true

  local n_joined
  n_joined=$(wc -l < "$joined" | tr -d ' ')
  echo "Paired: $n_joined files"

  # Guardrails
  if (( n_joined == 0 )); then
    die "No pairs matched between $(basename "$noisy_dir") and $(basename "$clean_dir"). Check filenames."
  fi
  if (( n_joined < n_noisy )); then
    warn "$((n_noisy - n_joined)) noisy files had no matching clean partner."
    # Show a few missing for quick debugging
    comm -23 <(cut -f1 "$noisy_keys") <(cut -f1 "$clean_keys") | head -n 10 \
      | sed 's/^/  missing clean for basename: /' >&2
  fi
  if (( n_joined < n_clean )); then
    note "$((n_clean - n_joined)) clean files had no matching noisy partner."
  fi
  # Fail if fewer than 90% of noisy files were paired (likely wrong dirs)
  if (( n_joined * 10 < n_noisy * 9 )); then
    die "Too few pairs matched ($n_joined of $n_noisy). Check directory mapping."
  fi

  # Write CSV header + rows with ABSOLUTE paths, CSV-quoted
  # CSV quoting: enclose each field in double quotes and escape internal quotes by doubling them.
  {
    echo "noisy,clean"
    awk -F'\t' '{
      gsub(/"/, "\"\"", $1); gsub(/"/, "\"\"", $2);
      printf "\"%s\",\"%s\"\n", $1, $2
    }' "$joined"
  } > "$out_csv"

  echo "Wrote: $out_csv"
}

# --- Checks -------------------------------------------------------------------
check_dir "$DATA_DIR"
check_dir "$CLEAN_TRAIN"
check_dir "$NOISY_TRAIN"
check_dir "$CLEAN_TEST"
check_dir "$NOISY_TEST"

# --- Build CSVs ---------------------------------------------------------------
echo "Building VoiceBank-DEMAND (16k) splits…"
pair_split "$NOISY_TRAIN" "$CLEAN_TRAIN" "$OUT_TRAIN"
pair_split "$NOISY_TEST"  "$CLEAN_TEST"  "$OUT_TEST"

echo "Done."
