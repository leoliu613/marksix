#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Mark Six inter-draw adjacency on a 5x10 board with optional 3x3 wrap (torus) geometry.

Board mapping:
- Numbers 1..49 arranged on 5 columns (1..10, 11..20, 21..30, 31..40, 41..50) and 10 rows bottom->top.
- Number n â†’ (row=(n-1)%10+1, col=(n-1)//10+1)
- Valid numbers are 1..49 (exclude 50 at (row=10,col=5)).

Adjacency:
- Chebyshev (king move) distance. Radius r âˆˆ {1,2}.
- wrap=False: ordinary board clipped at edges.
- wrap=True: torus (ç’°é�¢) distance using min(|Î”|, size-|Î”|) for rows (10) & columns (5).
- exclude_center: remove seed numbers themselves from union (æŽ’é™¤ã€Œæ­£ä¸­ã€�).

Statistics:
For each consecutive pair of draws (t-1 â†’ t):
  1. Build union neighborhood U from previous draw's 7 numbers.
  2. Count hits = numbers of current draw contained in U.
  3. Hypergeometric baseline: drawing n=7 from N=49 with K=|U|.
     Expectation E = n*K/N, Var = n*(K/N)*(1-K/N)*((N-n)/(N-1)).
  4. Aggregate totals â†’ z-score (è§€å¯Ÿå€¼èˆ‡ç�¨ç«‹éš¨æ©Ÿç›¸æ¯”æ˜¯å�¦é¡¯è‘—).

Outputs:
- Six configurations:
  * no-wrap r<=2 include center
  * no-wrap r<=2 exclude center
  * no-wrap r<=1 include center
  * wrap(3x3) r<=2 include center
  * wrap(3x3) r<=2 exclude center
  * wrap(3x3) r<=1 include center
- Histogram of hits per pair.
- Theoretical average single-seed neighborhood size.

Example:
Print detailed stats for the pair 2025-10-23 â†’ 2025-10-25 (r<=2, both wrap and no-wrap).

Usage:
  python3 scripts/analyze_grid_adjacency.py lottery_result.json
"""
import json, math, sys
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Set, Dict

ROWS = 10
COLS = 5
POP_SIZE = 49  # valid numbers 1..49

def load_draws(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for row in data:
        date = row.get("date", "")
        nums = [n for n in row.get("numbers", []) if 1 <= n <= 49]
        if date == "YYYY-MM-DD" or not nums:
            continue
        out.append({"date": date, "numbers": sorted(set(nums))})
    out.sort(key=lambda r: datetime.strptime(r["date"], "%Y-%m-%d"))
    return out

def pos(n: int) -> Tuple[int, int]:
    c = (n - 1) // 10 + 1
    r = (n - 1) % 10 + 1
    return (r, c)

def num(r: int, c: int) -> int:
    return (c - 1) * 10 + r  # may produce 50

def chebyshev_torus(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    dr = abs(a[0]-b[0]); dc = abs(a[1]-b[1])
    dr = min(dr, ROWS - dr)
    dc = min(dc, COLS - dc)
    return max(dr, dc)

def neighborhood(seed: int, radius: int, wrap: bool) -> Set[int]:
    r0, c0 = pos(seed)
    cells = set()
    if wrap:
        for n in range(1, 50):  # exclude 50
            if chebyshev_torus((r0, c0), pos(n)) <= radius:
                cells.add(n)
    else:
        rmin = max(1, r0 - radius)
        rmax = min(ROWS, r0 + radius)
        cmin = max(1, c0 - radius)
        cmax = min(COLS, c0 + radius)
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                n = num(r, c)
                if 1 <= n <= 49:
                    cells.add(n)
    return cells

def union_neighborhood(seeds: List[int], radius: int, wrap: bool, exclude_center: bool) -> Set[int]:
    U = set()
    for s in seeds:
        U |= neighborhood(s, radius, wrap)
    if exclude_center:
        U -= set(seeds)
    return U

def pair_stats(prev_nums: List[int], curr_nums: List[int], radius: int, wrap: bool, exclude_center: bool):
    U = union_neighborhood(prev_nums, radius, wrap, exclude_center)
    K = len(U)
    hits = sum(1 for n in curr_nums if n in U)
    N = POP_SIZE
    n = len(curr_nums)
    p = K / N
    exp = n * p
    var = n * p * (1 - p) * ((N - n) / (N - 1))
    return {"U_size": K, "hits": hits, "exp": exp, "var": var}

def summarize(pairs: List[Dict]) -> Dict:
    total_hits = sum(p["hits"] for p in pairs)
    total_exp  = sum(p["exp"]  for p in pairs)
    total_var  = sum(p["var"]  for p in pairs)
    z = (total_hits - total_exp) / math.sqrt(total_var) if total_var > 0 else float("nan")
    hist = Counter(p["hits"] for p in pairs)
    avg_U = sum(p["U_size"] for p in pairs) / len(pairs)
    return {
        "pairs": len(pairs),
        "total_hits": total_hits,
        "total_exp": total_exp,
        "total_var": total_var,
        "z_score": z,
        "avg_hits_per_pair": total_hits / len(pairs),
        "avg_exp_per_pair": total_exp / len(pairs),
        "avg_U_size": avg_U,
        "avg_U_cover_pct": 100 * avg_U / POP_SIZE,
        "hist_hits": dict(sorted(hist.items())),
    }

def theoretical_single_seed_size(radius: int, wrap: bool) -> float:
    sizes = []
    for s in range(1, 50):
        sizes.append(len(neighborhood(s, radius, wrap)))
    return sum(sizes) / len(sizes)

def analyze(draws: List[Dict], radius: int, wrap: bool, exclude_center: bool) -> Dict:
    pairs = []
    for i in range(1, len(draws)):
        prev = draws[i-1]["numbers"]
        curr = draws[i]["numbers"]
        pairs.append(pair_stats(prev, curr, radius, wrap, exclude_center))
    S = summarize(pairs)
    S["theory_single_seed"] = theoretical_single_seed_size(radius, wrap)
    return S

def main(path: str = "lottery_result.json"):
    draws = load_draws(path)
    print(f"Loaded draws (excluding placeholder): {len(draws)}")
    if len(draws) < 2:
        return
    print(f"Range: {draws[0]['date']} â†’ {draws[-1]['date']}")
    configs = [
        ("no-wrap r<=2 include center", False, 2, False),
        ("no-wrap r<=2 exclude center", False, 2, True),
        ("no-wrap r<=1 include center", False, 1, False),
        ("wrap(3x3) r<=2 include center", True, 2, False),
        ("wrap(3x3) r<=2 exclude center", True, 2, True),
        ("wrap(3x3) r<=1 include center", True, 1, False),
    ]
    for name, wrap, r, excl in configs:
        S = analyze(draws, r, wrap, excl)
        print(f"\n=== {name} ===")
        print(f"pairs={S['pairs']}, avg_U_size={S['avg_U_size']:.2f} ({S['avg_U_cover_pct']:.1f}%)")
        print(f"avg_hits={S['avg_hits_per_pair']:.3f}, avg_exp={S['avg_exp_per_pair']:.3f}, z={S['z_score']:.3f}")
        print(f"theory_single_seed_size (avg)={S['theory_single_seed']:.2f}")
        print(f"hist_hits={S['hist_hits']}")

    # Example 2025-10-23 â†’ 2025-10-25
    for i in range(1, len(draws)):
        if draws[i-1]["date"] == "2025-10-23" and draws[i]["date"] == "2025-10-25":
            prev = draws[i-1]["numbers"]; curr = draws[i]["numbers"]
            for wrap in (False, True):
                ex = pair_stats(prev, curr, radius=2, wrap=wrap, exclude_center=False)
                mode = "wrap" if wrap else "no-wrap"
                print(f"\nExample 2025-10-23 â†’ 2025-10-25 (r<=2, {mode}, include center)")
                print(f"prev={prev}\ncurr={curr}")
                print(f"|U|={ex['U_size']} ({100*ex['U_size']/POP_SIZE:.1f}%), hits={ex['hits']}, E={ex['exp']:.2f}")
            break

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lottery_result.json")
