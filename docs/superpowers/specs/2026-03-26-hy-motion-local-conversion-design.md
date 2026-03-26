# HY-Motion Local Conversion Design

## Summary

Move the HY-Motion `.npz` to ProtoMotions `.motion` conversion entrypoint into this repository so users no longer need the separate `hymotion_isaaclab` repo to run conversion. The converter should remain a standalone script owned by `HY-Motion`, while preserving the existing conversion behavior and keeping the current runtime dependency on a local `ProtoMotions` checkout.

## Current State

- `generate_motion.py` writes HY-Motion `.npz` files and tells users to switch into `~/code/hymotion_isaaclab` to convert them.
- `/home/lyuxinghe/code/hymotion_isaaclab/scripts/convert_npz.py` is only a CLI wrapper.
- The real conversion logic lives in `hymotion_isaaclab/conversion/npz_to_motion.py`.
- The conversion logic already depends on `ProtoMotions` modules and assets at runtime.

## Goals

- Make conversion available directly from this repo.
- Remove the dependency on `hymotion_isaaclab` for conversion.
- Keep the conversion workflow simple enough to call from a shell script.
- Update `generate_motion.py` comments and user-facing output so they reflect the new local conversion path.

## Non-Goals

- Removing the runtime dependency on `ProtoMotions`.
- Refactoring the conversion math or changing the output `.motion` format.
- Changing the generation pipeline in `generate_motion.py` beyond comments and next-step guidance.

## Chosen Approach

Use a single-file local converter at `scripts/convert_npz.py`.

This file will:

- own the CLI arguments for converting one file or a directory,
- inline the conversion logic previously imported from `hymotion_isaaclab`,
- keep the existing `--protomotions-root` argument and default path,
- remain executable as a standalone script from this repo.

This is preferred over splitting the logic into a new Python package module because the immediate use case is script-driven conversion, and a single file keeps setup and invocation simple.

## Wrapper Script Design

Add a bash wrapper script that:

- accepts `--prompt`,
- accepts `--output-dir` with default `output/generated`,
- accepts `--output-type` with values `motion|npz` and default `motion`,
- always runs `generate_motion.py`,
- invokes the local converter after generation when `--output-type motion` is selected.

The wrapper should avoid changing the Python generation code path. It should derive the generated `.npz` path from the prompt slug and the existing `_000.npz` naming convention used by `generate_motion.py`.

## Testing Strategy

Use TDD with `pytest`.

- Add focused tests for converter argument parsing and output filename behavior.
- Add focused tests for the wrapper script behavior around `output-type=motion` versus `output-type=npz`.
- Verify the tests fail before implementation, then pass after implementation.

## Risks

- The converter still requires `ProtoMotions` on `PYTHONPATH` at runtime, so the repo becomes standalone relative to `hymotion_isaaclab`, not standalone relative to all external motion tooling.
- The wrapper script relies on the existing output naming convention from `generate_motion.py`; if that convention changes later, the wrapper and tests must be updated together.
