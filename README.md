# SAGEFuzz source artifact

This directory contains the Rust source, bundled RISC-V instruction metadata, configurations, and core plugins needed to build SAGEFuzz. Ara, XiangShan, and Spike simulator binaries are external dependencies; they are not included.

## Requirements

- Rust and Cargo with edition 2024 support, and Python 3 for regenerating the bundled Spike guard data.
- A RISC-V GNU toolchain on `PATH`: `riscv64-unknown-elf-gcc`, `as`, `ld`, and `objdump`.
- For execution, a compatible Ara or XiangShan simulator. Spike is needed when using it as an ISA reference.

All commands below run from this directory. Build outputs remain under this directory's `target/`; generated programs remain under the selected `--run-dir`. The `./sagefuzz` launcher selects the RVV generator and exposes only generation and direct-execution commands.

## Build

```bash
cargo build --locked --release -p riscv_fuzz_test
CARGO_BUILD_JOBS=2 bash scripts/build_plugin_bundle.sh --release \
  riscv-fuzz-core-spike riscv-fuzz-core-ara riscv-fuzz-core-xiangshan
```

The second command places the three shared libraries in `riscv_fuzz_plugins/`. To check the Rust generator without a simulator:

```bash
cargo test --locked -p riscv-fuzz-testcase-engine --lib
python3 -m unittest discover -s scripts -p 'test_extract_spike_guards.py'
```

The generator reads `data/spike_guard_rules.json`, which contains parsed guard expressions from Spike instruction headers and shared RVV macros. To regenerate that file from another Spike checkout, run:

```bash
python3 scripts/extract_spike_guards.py \
  --spike-source /path/to/riscv-isa-sim \
  --out data/spike_guard_rules.json
```

Rebuild the Rust program after changing the guard data. Rules the extractor cannot normalize remain recorded in the JSON and are not silently treated as executable constraints.

## Generate a program

`generate` compiles one program and saves its assembly, ELF, and metadata. It does not execute the program. The seed is in `configs/experiments/rv64_sagefuzz_smoke.toml`; copy that file and edit `seed` to generate a different program.

```bash
./sagefuzz generate \
  --config configs/experiments/rv64_sagefuzz_smoke.toml \
  --overlay configs/overlays/cores_spike_xiangshan_single.toml \
  --core xiangshan \
  --run-dir runs/xiangshan-generate

./sagefuzz generate \
  --config configs/experiments/rv64_sagefuzz_smoke.toml \
  --overlay configs/overlays/cores_spike_ara_single.toml \
  --core ara \
  --run-dir runs/ara-generate
```

Inspect `program.S` and `program.elf` in each run directory. `--core` selects the target's ISA and memory configuration.

## Execute on a DUT

Provide paths to your own simulator builds. `exec-generated` generates and directly executes one new program on the selected DUT. Keep each `--run-dir` unique.

```bash
CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C=/path/to/xiangshan/emu \
CX_RISCV_CORES_XIANGSHAN_DIFFTEST_RV64_1C_SO=/path/to/xiangshan/riscv64-spike-so \
  ./sagefuzz exec-generated \
  --config configs/experiments/rv64_sagefuzz_smoke.toml \
  --overlay configs/overlays/cores_spike_xiangshan_single.toml \
  --core xiangshan \
  --run-dir runs/xiangshan-execute

CX_RISCV_CORES_ARA_RV64_1C=/path/to/ara/Vara_tb_verilator \
CX_RISCV_CORES_SPIKE=/path/to/spike \
  ./sagefuzz exec-generated \
  --config configs/experiments/rv64_sagefuzz_smoke.toml \
  --overlay configs/overlays/cores_spike_ara_single.toml \
  --core ara \
  --run-dir runs/ara-execute
```

For command-line options, run `./sagefuzz --help` or `./sagefuzz generate --help`. To check plugin discovery, run `RISCV_FUZZ_PLUGIN_DIRS="$PWD/riscv_fuzz_plugins" target/release/riscv_fuzz_test plugin doctor`.
