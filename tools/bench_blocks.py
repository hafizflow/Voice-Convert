"""Benchmark RVC streaming configs on this machine: latency vs real-time factor.

    python -m tools.bench_blocks IN.wav --model M.pth --index M.index --seconds 10
Writes each config's output to work/bench/ so quality can be judged by ear.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from engine.rvc import RVCConverter
from simulate import SAMPLE_RATE, boundary_step_ratio, load_48k_mono, simulate

# (block_ms, extra_s, crossfade_s, f0_method, device)
DEFAULT_CONFIGS = [
    (100, 0.5, 0.05, "rmvpe", None),
    (100, 0.15, 0.05, "rmvpe", None),
    (200, 0.3, 0.05, "rmvpe", None),
    (100, 0.15, 0.05, "fcpe", None),
    (100, 0.15, 0.05, "rmvpe", "cpu"),
]


def main() -> None:
    import soundfile as sf

    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--model", required=True)
    p.add_argument("--index")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--out-dir", default="work/bench")
    p.add_argument("--index-rate", type=float, default=0.5)
    p.add_argument("--configs", help="block_ms:extra_s:f0:device,... e.g. 100:0.15:fcpe:cpu")
    a = p.parse_args()
    configs = DEFAULT_CONFIGS
    if a.configs:
        configs = []
        for item in a.configs.split(","):
            b, e, f, d = item.split(":")
            configs.append((int(b), float(e), 0.05, f, None if d == "auto" else d))
    inp, out_dir = str(Path(a.input).resolve()), Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    x = load_48k_mono(inp)[: int(a.seconds * SAMPLE_RATE)]

    print(f"{'block':>6} {'extra':>6} {'f0':>6} {'dev':>5} | {'mean ms':>8} {'p95 ms':>8} {'RTF':>5} "
          f"{'latency':>8} {'step':>5}")
    for block_ms, extra_s, xf, f0, dev in configs:
        try:
            conv = RVCConverter(a.model, a.index, block_ms=block_ms, extra_s=extra_s,
                                crossfade_s=xf, f0_method=f0, device=dev,
                                index_rate=a.index_rate)
            res = simulate(x, conv)
        except Exception as e:  # report and carry on: some combos may be unsupported
            print(f"{block_ms:>6} {extra_s:>6} {f0:>6} {dev or 'auto':>5} | FAILED: {type(e).__name__}: {e}")
            continue
        bm = np.array(res.block_ms) if res.block_ms else np.zeros(1)
        name = f"b{block_ms}_x{extra_s}_{f0}_{dev or 'auto'}.wav"
        sf.write(out_dir / name, res.output, SAMPLE_RATE)
        print(f"{block_ms:>6} {extra_s:>6} {f0:>6} {dev or 'auto':>5} | {bm.mean():>8.0f} "
              f"{np.percentile(bm, 95):>8.0f} {res.rtf:>5.2f} {res.algorithmic_latency_ms:>6.0f}ms "
              f"{boundary_step_ratio(res.output, block_ms * 48):>5.2f}", flush=True)


if __name__ == "__main__":
    main()
