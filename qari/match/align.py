"""Word-level Smith-Waterman with character similarity between words."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

MATCH_SIM = 0.85   # similarity at or above this is a "match", below is a "substitution"


def skeleton(norm: str) -> str:
    """Matching key: normalized word with alefs removed (Uthmani and simple script disagree on them)."""
    return norm.replace("ا", "") or norm


@lru_cache(maxsize=1 << 18)
def similarity(a: str, b: str) -> float:
    """1 - normalized Levenshtein distance, in [0, 1]."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


@dataclass(frozen=True)
class Op:
    kind: str            # match | substitution | missing | extra
    query_index: int | None
    gid: int | None
    sim: float = 0.0


@dataclass(frozen=True)
class Alignment:
    raw: float
    q_start: int         # first / last aligned query index (inclusive)
    q_end: int
    r_start: int         # first / last aligned reference index in the window (inclusive)
    r_end: int
    ops: list[Op]        # only the aligned region; gids are window offsets until the caller rebases


def sub_score(sim: float) -> float:
    return (sim - 0.5) * 2.0


def local_align(q: list[str], r: list[str], gap: float = -0.7) -> Alignment | None:
    nq, nr = len(q), len(r)
    if not nq or not nr:
        return None
    H = [[0.0] * (nr + 1) for _ in range(nq + 1)]
    T = [[0] * (nr + 1) for _ in range(nq + 1)]   # 1 diag, 2 up (extra query word), 3 left (missing)
    best, bi, bj = 0.0, 0, 0
    for i in range(1, nq + 1):
        for j in range(1, nr + 1):
            d = H[i - 1][j - 1] + sub_score(similarity(q[i - 1], r[j - 1]))
            u = H[i - 1][j] + gap
            l = H[i][j - 1] + gap
            h = max(0.0, d, u, l)
            H[i][j] = h
            if h > 0:
                T[i][j] = 1 if h == d else (2 if h == u else 3)
            if h > best:
                best, bi, bj = h, i, j
    if best <= 0:
        return None
    ops: list[Op] = []
    i, j = bi, bj
    while i > 0 and j > 0 and H[i][j] > 0:
        t = T[i][j]
        if t == 1:
            s = similarity(q[i - 1], r[j - 1])
            ops.append(Op("match" if s >= MATCH_SIM else "substitution", i - 1, j - 1, s))
            i, j = i - 1, j - 1
        elif t == 2:
            ops.append(Op("extra", i - 1, None))
            i -= 1
        else:
            ops.append(Op("missing", None, j - 1))
            j -= 1
    ops.reverse()
    q_idx = [o.query_index for o in ops if o.query_index is not None]
    r_idx = [o.gid for o in ops if o.gid is not None]
    return Alignment(best, min(q_idx), max(q_idx), min(r_idx), max(r_idx), ops)
