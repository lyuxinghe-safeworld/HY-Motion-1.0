#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/generate_motion.sh --prompt "text prompt" [--duration 3.0] [--output-dir output/generated] [--output-type motion|npz]

Options:
  --prompt       Text prompt describing the desired motion. Required.
  --duration     Motion duration in seconds. Defaults to 3.0.
  --enable-body-model-chunking  Decode wooden mesh in frame chunks to reduce peak VRAM usage.
  --output-dir   Output directory. Defaults to output/generated under the repo root.
  --output-type  Desired final output type: motion or npz. Defaults to motion.
  -h, --help     Show this help message.
EOF
}

slugify() {
  python3 - "$1" <<'PY'
import re
import sys

text = sys.argv[1].lower().strip()
text = re.sub(r"[^\w\s-]", "", text)
text = re.sub(r"[\s_-]+", "_", text)
text = re.sub(r"^-+|-+$", "", text)
print(text[:80])
PY
}

prompt=""
duration="3.0"
enable_body_model_chunking="0"
output_dir="output/generated"
output_type="motion"

while (($#)); do
  case "$1" in
    --prompt)
      if (($# < 2)); then
        echo "Missing value for --prompt" >&2
        usage >&2
        exit 1
      fi
      prompt="$2"
      shift 2
      ;;
    --output-dir)
      if (($# < 2)); then
        echo "Missing value for --output-dir" >&2
        usage >&2
        exit 1
      fi
      output_dir="$2"
      shift 2
      ;;
    --duration)
      if (($# < 2)); then
        echo "Missing value for --duration" >&2
        usage >&2
        exit 1
      fi
      duration="$2"
      shift 2
      ;;
    --enable-body-model-chunking)
      enable_body_model_chunking="1"
      shift
      ;;
    --output-type)
      if (($# < 2)); then
        echo "Missing value for --output-type" >&2
        usage >&2
        exit 1
      fi
      output_type="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$prompt" ]]; then
  echo "--prompt is required" >&2
  usage >&2
  exit 1
fi

if [[ "$output_type" != "motion" && "$output_type" != "npz" ]]; then
  echo "--output-type must be either 'motion' or 'npz'" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

if [[ "$output_dir" != /* ]]; then
  output_dir="$repo_root/$output_dir"
fi

if [[ "$enable_body_model_chunking" == "1" ]]; then
  set -- "$python_bin" "$script_dir/generate_motion.py" \
    --prompt "$prompt" \
    --duration "$duration" \
    --output-dir "$output_dir" \
    --enable-body-model-chunking
else
  set -- "$python_bin" "$script_dir/generate_motion.py" \
    --prompt "$prompt" \
    --duration "$duration" \
    --output-dir "$output_dir"
fi

"$@"

if [[ "$output_type" == "motion" ]]; then
  slug="$(slugify "$prompt")"
  prompt_output_dir="$output_dir/$slug"
  npz_path="$prompt_output_dir/${slug}.npz"

  if [[ ! -f "$npz_path" ]]; then
    echo "Expected generated NPZ not found: $npz_path" >&2
    exit 1
  fi

  "$python_bin" "$script_dir/convert_npz.py" \
    --npz-file "$npz_path" \
    --output-dir "$prompt_output_dir"
fi
