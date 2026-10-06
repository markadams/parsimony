#!/usr/bin/env python3
"""Per-kernel summary of an ncu raw CSV (ncu --import X --csv --page raw --print-units base).

Kernels are named by their NVTX range (ncu --print-nvtx-rename kernel), since every
Grid kernel is LambdaApply.  Times are replay times: use them for ratios within one
kernel, never as wall time (take that from the benchmark's own timers).
"""
import csv, re, sys

A100_DRAM = 1555e9   # B/s, A100-SXM4-40GB

def col(h, pat):
    for i, c in enumerate(h):
        if re.fullmatch(pat, c):
            return i
    return None

def main(path):
    rows = list(csv.reader(open(path)))
    h, data = rows[0], rows[2:]          # rows[1] holds units
    get = lambda r, pat: (lambda i: r[i] if i is not None else '')(col(h, pat))
    num = lambda r, pat: float(get(r, pat).replace(',', '') or 'nan')
    stalls = [c for c in h if re.fullmatch(r'smsp__average_warps_issue_stalled_(.*)_per_issue_active\.ratio', c)]
    for r in data:
        t = num(r, r'gpu__time_duration\.sum')
        rd, wr = num(r, r'dram__bytes_read\.sum'), num(r, r'dram__bytes_write\.sum')
        l2 = num(r, r'lts__t_sectors\.sum') * 32
        st = sorted(((num(r, c), re.sub(r'smsp__average_warps_issue_stalled_|_per_issue_active\.ratio', '', c)) for c in stalls), reverse=True)[:4]
        print(f"--- {get(r, 'Kernel Name')}   [{get(r, r'thread Domain:Push/Pop_Range.*')}]")
        print(f"  grid {get(r, r'launch__grid_size')} x block {get(r, r'launch__block_size')}, "
              f"{get(r, r'launch__registers_per_thread')} regs, "
              f"occupancy theor {num(r, r'sm__maximum_warps_per_active_cycle_pct'):.0f}% achieved {num(r, r'sm__warps_active\.avg\.pct_of_peak_sustained_active'):.0f}%")
        print(f"  time {t/1e3:.1f} us (replay)  DRAM read {rd/1e6:.1f} MB write {wr/1e6:.1f} MB  "
              f"-> {(rd+wr)/t:.0f} GB/s = {100*(rd+wr)/t*1e9/A100_DRAM:.0f}% of peak;  L2 {l2/1e6:.1f} MB, hit {num(r, r'lts__t_sector_hit_rate\.pct'):.0f}%")
        print(f"  SM throughput {num(r, r'sm__throughput\.avg\.pct_of_peak_sustained_elapsed'):.0f}%  "
              f"memory throughput {num(r, r'gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed'):.0f}%  "
              f"local spill requests {get(r, r'derived__local_spilling_requests')}")
        print("  top stalls (warps per issue): " + ", ".join(f"{n} {v:.2f}" for v, n in st))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ncu_raw.csv')
