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
