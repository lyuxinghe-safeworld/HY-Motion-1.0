# HY-Motion Local Conversion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move NPZ-to-motion conversion into this repo, update generation guidance, and add a wrapper script that can return either `.npz` or `.motion`.

**Architecture:** Vendor the conversion logic into a local standalone script at `scripts/convert_npz.py`, keep `generate_motion.py` as the generator, and add a bash wrapper that orchestrates generation plus optional conversion. Preserve the existing ProtoMotions-dependent conversion math and expose only the new local repo entrypoints.

**Tech Stack:** Python, Bash, pytest, existing HY-Motion runtime dependencies, local ProtoMotions checkout

---

## Chunk 1: Converter Ownership

### Task 1: Add a red test for the local converter interface

**Files:**
- Create: `tests/test_convert_npz.py`
- Test: `tests/test_convert_npz.py`

- [ ] **Step 1: Write the failing test**

```python
def test_local_converter_parse_args_accepts_single_file_and_output_dir():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convert_npz.py -q`
Expected: FAIL because `scripts.convert_npz` does not exist in this repo yet.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/convert_npz.py` with the CLI and inlined conversion logic.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convert_npz.py -q`
Expected: PASS

### Task 2: Preserve output naming behavior

**Files:**
- Modify: `tests/test_convert_npz.py`
- Modify: `scripts/convert_npz.py`

- [ ] **Step 1: Write the failing test**

```python
def test_iter_conversion_jobs_emits_motion_output_names():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convert_npz.py -q`
Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add a small helper that resolves input `.npz` files and output `.motion` paths.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convert_npz.py -q`
Expected: PASS

## Chunk 2: Generation and Wrapper Flow

### Task 3: Add a red test for wrapper branching

**Files:**
- Create: `tests/test_generate_motion_wrapper.py`
- Test: `tests/test_generate_motion_wrapper.py`

- [ ] **Step 1: Write the failing test**

```python
def test_wrapper_skips_conversion_for_npz_output():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate_motion_wrapper.py -q`
Expected: FAIL because the wrapper script does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add the bash wrapper script and implement `motion` versus `npz` branching.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generate_motion_wrapper.py -q`
Expected: PASS

### Task 4: Update generator guidance to the local converter

**Files:**
- Modify: `generate_motion.py`

- [ ] **Step 1: Update comments and final status output**

Point users to `HY-Motion/scripts/convert_npz.py` instead of `hymotion_isaaclab/scripts/convert_npz.py`.

- [ ] **Step 2: Verify by inspection**

Run: `sed -n '1,80p' generate_motion.py && sed -n '250,320p' generate_motion.py`
Expected: top docstring and final summary reference the local script.

## Chunk 3: Final Verification

### Task 5: Run the focused verification suite

**Files:**
- Test: `tests/test_convert_npz.py`
- Test: `tests/test_generate_motion_wrapper.py`

- [ ] **Step 1: Run targeted tests**

Run: `pytest tests/test_convert_npz.py tests/test_generate_motion_wrapper.py -q`
Expected: PASS

- [ ] **Step 2: Dry-run the wrapper help**

Run: `bash scripts/generate_motion.sh --help`
Expected: exit 0 with usage text.

- [ ] **Step 3: Dry-run the converter help**

Run: `python scripts/convert_npz.py --help`
Expected: exit 0 with usage text.
