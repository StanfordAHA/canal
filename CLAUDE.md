# CLAUDE.md — canal session notes

Interconnect generator (switch boxes, connection boxes, routing graph) used by garnet.
aha `mek` pins this repo's personal branch `thesis` (see /aha/CLAUDE.md "Which branch to
commit to").

## CI and style (2026-10-16, session 352df910)

- GitHub "CI Test" (`.github/workflows/linux.yml`, stock `stanfordaha/garnet:latest`
  container) copies THIS checkout over the image's `/aha/canal`, installs it with
  `pip install --no-deps -e` in the aha venv (a bare `pip` is the image's conda Python),
  runs pycodestyle, then `aha map` + `aha pnr apps/pointwise` on a 4x4 CGRA. Green since
  4a3ed8a. It used to fail at "Install deps": `git fetch origin` inside the container asks
  for credentials.
- Before pushing, from the repo root: `python -m pycodestyle canal tests *.py` must print
  nothing (setup.cfg ignores only E741 and E501, so W503 AND W504 are both on: no line
  break next to a binary operator). pycodestyle 2.12.1 is in the /aha venv.
- Ready-valid register pre-fill matches register blocks by bit width (`_reg_width_mismatch`,
  c3894d2; garnet records the widths in id_to_metadata).

## Switch boxes without ready-valid (2026-10-17, session cfc4174c)

- `__wire_flush_to_sbs` (circuit.py) wired the tile flush into every switch box with
  registers, but only a ready-valid switch box has a flush port: building a static
  (`ready_valid=False`) fabric died with `KeyError: 'flush'` (since f3955c9, 2025-02).
  It now skips switch boxes without a flush port; ready-valid RTL is unchanged.
- Users: garnet `--thesis static` (garnet_thesis.py: the static-dense thesis fabric, plain
  register + bypass per track) and garnet `--dense-only`. The stock image's canal (aha
  master) still has the bug, so garnet's mflowgen rtl step fetches this branch in the
  container for thesis builds.

## Switch-box registers that are always FIFOs: `fifo_only_regs` (2026-10-19, session cfc4174c)

- **Why:** every ready-valid switch-box register (`FifoRegWrapper` → `RegFIFO`, depth 2) has
  a `<reg>_fifo` config bit. `fifo_en=1` is the FIFO. `fifo_en=0` is the static-mode plain
  register, which canal's own `test_reg_fifo_sr` relies on: push and pop are forced every
  cycle, so data shifts by one clock while `valid_out` and `ready_out` stay at 1.
  - Simulated on the thesis RV `garnet.v` module: it outputs tokens whose `valid_in` was 0,
    and loses tokens while `ready_in` is low.
  - Legal only for static apps on a fabric that runs both kinds.
  - `get_route_bitstream` sets `fifo_en=1` on every register a route uses only when
    `use_fifo` (DENSE_READY_VALID) and `USE_NON_SPLIT_FIFOS=1` hold; otherwise a used
    register silently stays a plain register.
- **What:** `Interconnect(..., fifo_only_regs=True)` → TileCircuit → SB →
  `FifoRegWrapper(fifo_only=True)` → `RegFIFO(fifo_mode_config=False)`. The result has no
  `fifo_en` port, no `<reg>_fifo` config bit and no push/pop muxes; modules are named
  `*_fifo_only`. `configure_fifo` then writes only `bogus_init`. Unused registers are
  bypassed by the register mux, as before.
  - Needs `ready_valid` and `USE_NON_SPLIT_FIFOS=1` (asserted).
  - Off by default. Only garnet `--thesis rv` turns it on (user decision 2026-10-19).
- **Stock RTL is unchanged:** zircon 8x2 (rtl-goldcheck flags) with old vs new canal gives
  the same module set. All 37 canal-built modules are byte-identical; only lake's sparse
  `scanner_pipe` / `write_scanner` differ, in statement order, which `vcompare.sh` sorts
  away.
- **Tests:**
  - `tests/test_logic.py::test_reg_fifo_only_ports`;
  - `test_reg_fifo_only_holds_on_stall` (verilator; skipped without it);
  - `tests/test_interconnect.py::test_fifo_only_regs` (config bits, ports, the
    `configure_fifo` writes, the RTL).
- **Pitfall: no local RV interconnect tests ran before this.** `create_dummy_cgra(...,
  ready_valid=True)` died in finalize with `KeyError: 'flush'`, because ready-valid switch
  boxes take the tile's flush and dummy tiles had none. The helper now gives each tile a
  flush port, so `test_ready_valid_verilog` passes again.
- **No verilator on /aha:** 70 of canal's tests fail here with or without this change
  (fault's verilator runs). Before/after lists: no new failures.
- **Generator cache:** gemstone caches circuits by name, and tiles skip content hashing.
  Two fabrics built in one process that differ only in content share a cached definition;
  tests that build several call `clear_generator_cache()`.

## Network-side FIFO in a core input's CB: `input_fifo_ports` (2026-10-21, session cfc4174c; user chose this over a lake fix)

- **Why:** the RV pond's write `ready` is its arbiter grant, and the request includes the
  port's `valid`, so `ready` depends combinationally on `valid`. A switch-box track's `valid`
  into a CB is ANDed with every enabled sink's `ready`, this CB's included
  (`*_ready_reduction_for_gating`). So a pond fed over the routing network forms a zero-delay
  loop: matmul_tile `ux=4` (unpacked kernel ponds) hung on the thesis RV fabric (4.39 ms
  simulated, no finish). A packed pond, fed from its PE over the inter-core edge, is fine:
  that source's `valid` is the PE's registered output, ungated.
- **What:** `Interconnect(..., input_fifo_ports=[names])` → TileCircuit → `CB(input_fifo=True)`.
  - The mux output for switch-box sources goes through a 2-deep FIFO (`FifoRegWrapper`
    fifo_only) before the core. The `ready` a selected track sees is the FIFO's registered
    `~full`.
  - An inter-core source bypasses the FIFO: `InputFifoSteer` picks it when the CB's own select
    is that source's index, passing its data, `valid` and the core's `ready` straight through.
    Packed latency is unchanged.
  - No new config bits, so routing and bitstreams are unchanged.
  - The FIFO takes the tile flush and stall, has no pre-fill, and is clocked only while the
    CB is enabled.
  - The module is `CB_<port>_input_fifo`; the instance name is unchanged.
  - Off by default. garnet `--thesis rv` enables it for `PondTop_input_width_{dw}_num_0`.
- **Checked** (thesis RV 16x16, 4 KB MEM + 128 B / 6-dim pond):
  - matmul_tile T44 `ux=4` bit-accurate, 2405 ns (three unpacked kernel banks, plus packed
    ones);
  - matmul_rvtry3 13355 ns with the same placement, routing and bitstream as without it;
  - conv_3_3 4150 ns;
  - zircon 8x2: all 241 canal-built modules byte-identical;
  - a standalone CB sim against the stock CB: identical on the loopback path; on a track,
    1 cycle of latency at full throughput, where the stock CB moved 0 tokens.
- **Cost:** local Genus freepdk45 Tile_PE +287 µm² (+0.6%); the pond CB grows 540 → 917.
  Network-fed pond writes take one more cycle.
- **Alternative not taken:** a lake change taking the pond's `ready` from its schedule step
  (`ready = sg_step & resource_ready`). It is area-neutral, but leaves a long combinational
  path from the pond's counters through the switch-box ready network back into the pond
  (-1375 ps locally). The prototype is in scratchpad cfc4174c `pondfix/B` (lake 908bbf07,
  unpushed).
- **Test pitfall:** building and compiling two fabrics that differ only in content in one
  process needs both `clear_generator_cache()` and `magma.frontend.coreir_.ResetCoreIR()`
  (coreir reuses same-named modules). `test_input_fifo_ports` calls both.
