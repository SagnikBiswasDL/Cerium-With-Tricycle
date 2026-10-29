"""GPU kernel time per primitive class from an `nsys stats -r cuda_gpu_kern_sum -f csv` file.

    python kernel_breakdown.py KERN_SUM.csv LAUNCHES

LAUNCHES is the number of CUDA-graph launches in the profiled process
(warmup + timed runs); times are reported per launch. Classes come from the
kernel names Cerium generates:

    sud      _kernel_sud_*   key-switch ModDown (scale down by the special primes)
    ntt      _kernel_ntt_*   forward NTT
    intt     _kernel_int_*   inverse NTT
    bconv    _kernel_bco<>   RNS base conversion (ModUp and ModDown)
    fused    _kernel_<n>     fused element-wise kernels: tensor products,
                             key inner products, plaintext products,
                             automorphisms, additions
    other    anything else

The timing is measured (GPU kernel durations with --cuda-graph-trace=node), but
the classes are primitives, not homomorphic operations: a rotation and a
relinearization both run through the same NTT / bconv / ModDown kernels.
"""

import csv
import json
import re
import sys

CLASSES = [("sud", r"_kernel_sud_"), ("ntt", r"_kernel_ntt_"), ("intt", r"_kernel_int_"),
           ("bconv", r"_kernel_bco<"), ("fused", r"_kernel_\d+\(")]


def breakdown(path, launches):
    per_class = {name: 0.0 for name, _ in CLASSES}
    per_class["other"] = 0.0
    kernels = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            ns = float(row["Total Time (ns)"])
            kernels += int(row["Instances"])
            for name, pattern in CLASSES:
                if re.search(pattern, row["Name"]):
                    per_class[name] += ns
                    break
            else:
                per_class["other"] += ns
    total = sum(per_class.values())
    return {
        "launches": launches,
        "kernel_time_per_launch_ms": total / launches / 1e6,
        "kernels_per_launch": kernels / launches,
        "per_launch_ms": {k: v / launches / 1e6 for k, v in per_class.items()},
        "share": {k: v / total for k, v in per_class.items()} if total else {},
    }


if __name__ == "__main__":
    print(json.dumps(breakdown(sys.argv[1], int(sys.argv[2])), indent=1))
