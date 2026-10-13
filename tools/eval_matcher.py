"""    python -m tools.eval_matcher --per-cell 40 --seed 1 [--no-extend] [--margin 0.05]"""
import argparse
from dataclasses import replace

from qari.match.evaluate import format_report, generate, run, threshold_table
from qari.match.matcher import Matcher, MatcherConfig
from qari.text.index import WordIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-extend", action="store_true")
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--show-wrong", type=int, default=5)
    a = ap.parse_args()
    cfg = MatcherConfig()
    if a.no_extend:
        cfg = replace(cfg, edge_extend=0)
    if a.margin is not None:
        cfg = replace(cfg, ambiguity_margin=a.margin)
    idx = WordIndex.from_file()
    res = run(Matcher(idx, cfg), generate(idx, a.per_cell, a.seed))
    print(format_report(res))
    print()
    print(threshold_table(res))
    for q, r in res["wrong"][: a.show_wrong]:
        s, e = idx[q.start], idx[q.end]
        gs, ge = idx[r.start_gid], idx[r.end_gid]
        print(f"\nWRONG {q.category}/{q.noise}: truth {s.surah}:{s.ayah}:{s.pos}-{e.surah}:{e.ayah}:{e.pos} "
              f"got {gs.surah}:{gs.ayah}:{gs.pos}-{ge.surah}:{ge.ayah}:{ge.pos} score={r.score:.2f}\n  {q.text}")


if __name__ == "__main__":
    main()
