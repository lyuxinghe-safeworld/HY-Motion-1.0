from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "scripts" / "generate_motion.sh"


def slugify(text: str) -> str:
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]


def write_fake_python(tmp_path: Path) -> Path:
    fake_python = tmp_path / "fakepython.sh"
    fake_python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail

            log_file="${FAKEPYTHON_LOG:?}"
            script_path="${1:-}"
            shift || true
            printf '%s %s\n' "$script_path" "$*" >> "$log_file"

            if [[ "$script_path" == *"generate_motion.py" ]]; then
              prompt=""
              output_dir=""
              while (($#)); do
                case "$1" in
                  --prompt)
                    prompt="$2"
                    shift 2
                    ;;
                  --output-dir)
                    output_dir="$2"
                    shift 2
                    ;;
                  *)
                    shift
                    ;;
                esac
              done

              slug="$(python3 -c 'import re, sys; text = sys.argv[1].lower().strip(); text = re.sub(r"[^\\w\\s-]", "", text); text = re.sub(r"[\\s_-]+", "_", text); text = re.sub(r"^-+|-+$", "", text); print(text[:80])' "$prompt")"
              mkdir -p "$output_dir"
              mkdir -p "$output_dir/$slug"
              : > "$output_dir/$slug/${slug}.npz"
            fi
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


def run_wrapper(
    tmp_path: Path,
    output_type: str | None,
    duration: str | None = None,
    enable_body_model_chunking: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    assert WRAPPER_PATH.exists(), f"Expected wrapper script at {WRAPPER_PATH}"

    log_file = tmp_path / "python.log"
    fake_python = write_fake_python(tmp_path)
    output_dir = tmp_path / "generated"

    cmd = [
        "bash",
        str(WRAPPER_PATH),
        "--prompt",
        "A person walks forward",
        "--output-dir",
        str(output_dir),
    ]
    if output_type is not None:
        cmd.extend(["--output-type", output_type])
    if duration is not None:
        cmd.extend(["--duration", duration])
    if enable_body_model_chunking:
        cmd.append("--enable-body-model-chunking")

    env = os.environ.copy()
    env["PYTHON_BIN"] = str(fake_python)
    env["FAKEPYTHON_LOG"] = str(log_file)

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    return completed, log_file


def test_wrapper_skips_conversion_for_npz_output(tmp_path: Path):
    completed, log_file = run_wrapper(tmp_path, "npz")

    assert completed.returncode == 0, completed.stderr
    log_lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    assert str(REPO_ROOT / "scripts" / "generate_motion.py") in log_lines[0]
    assert "convert_npz.py" not in log_lines[0]


def test_wrapper_runs_conversion_for_motion_output_by_default(tmp_path: Path):
    completed, log_file = run_wrapper(tmp_path, None)

    assert completed.returncode == 0, completed.stderr
    log_lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 2
    assert str(REPO_ROOT / "scripts" / "generate_motion.py") in log_lines[0]
    assert "convert_npz.py" in log_lines[1]
    expected_npz = f"{slugify('A person walks forward')}/{slugify('A person walks forward')}.npz"
    assert expected_npz in log_lines[1]


def test_wrapper_forwards_duration_to_generator(tmp_path: Path):
    completed, log_file = run_wrapper(tmp_path, "npz", duration="5.0")

    assert completed.returncode == 0, completed.stderr
    log_lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    assert "--duration 5.0" in log_lines[0]


def test_wrapper_forwards_body_model_chunking_flag(tmp_path: Path):
    completed, log_file = run_wrapper(
        tmp_path,
        "npz",
        enable_body_model_chunking=True,
    )

    assert completed.returncode == 0, completed.stderr
    log_lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    assert "--enable-body-model-chunking" in log_lines[0]
