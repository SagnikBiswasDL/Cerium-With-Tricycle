"""Homomorphic operation counts for a Cerium program, at two levels.

Source level (count_source): wraps the CeriumFunction graph constructors that
the DSL calls while a program is built, and counts each homomorphic operation
once per ciphertext it acts on. Whether an operand is a ciphertext is tracked
from the inputs (CiphertextInput vs PlaintextInput). These are the operations
written in the program, before the compiler lowers or fuses anything.

    ct_ct_mult      ciphertext x ciphertext products (tensor product; relin is separate)
    ct_pt_mult      ciphertext x plaintext products (incl. each BSGS plaintext)
    rotation        slot rotations by a nonzero amount (incl. hoisted / BSGS / rot-acc)
    conjugation     complex conjugations
    relinearize     relinearizations
    rescale         single rescales; double_rescale counts DoubleRescale ops
    to_ephemeral    key switches to the sparse ephemeral key (bootstrapping)
    bootstrap       BootstrapModRaise ops (one per bootstrapped ciphertext)
    add, sub, neg   additions / subtractions / negations involving a ciphertext
    key_switch      relinearize + rotation + conjugation + to_ephemeral

Compiled level (count_compiled): reads what the compiler produced for the
GPU. The *_cerium_graph_simple.dot file is the lowered kernel-group graph
(written when CERIUM_PRINT_GRAPHS=1); each line in a group is one
instruction on one RNS polynomial. ll.functions lists the functions the
runtime launches after fusion, with their type.
"""

import collections
import contextlib
import re

COUNTED = ("ct_ct_mult", "ct_pt_mult", "rotation", "conjugation", "relinearize", "rescale",
           "double_rescale", "modswitch", "to_ephemeral", "bootstrap", "add", "sub", "neg")


class _Tracker:
    def __init__(self):
        self.counts = collections.Counter()
        self.kind = {}   # id(term) -> ("ct" | "pt", number of ciphertexts)
        self._alive = []  # keeps terms alive so ids are not reused

    def mark(self, term, kind, n=1):
        self._alive.append(term)
        self.kind[id(term)] = (kind, n)
        return term

    def info(self, term):
        return self.kind.get(id(term), ("pt", 1))

    def unary(self, op, term, weight=1):
        kind, n = self.info(term)
        if kind == "ct":
            self.counts[op] += weight * n
        return kind, n


@contextlib.contextmanager
def count_source():
    """Context manager; build a CeriumProgram inside it, then read .counts."""
    from cerium.dsl import CeriumFunction

    t = _Tracker()
    originals = {}

    def patch(name, wrapper):
        original = getattr(CeriumFunction, name, None)
        if original is None:  # referenced by the DSL but not in this build's bindings
            return
        originals[name] = original

        def method(self, *args):
            return wrapper(original, self, *args)
        setattr(CeriumFunction, name, method)

    def result(term, kind, n):
        return t.mark(term, kind, n)

    patch("_make_ciphertext_input", lambda f, s, *a: result(f(s, *a), "ct", 1))
    patch("_make_plaintext_input", lambda f, s, *a: result(f(s, *a), "pt", 1))
    patch("_make_repeated_plaintext_input", lambda f, s, *a: result(f(s, *a), "pt", 1))

    def binary(op):
        def wrapper(f, s, x, y):
            (kx, nx), (ky, ny) = t.info(x), t.info(y)
            n = max(nx, ny)
            if op == "mul":
                if kx == ky == "ct":
                    t.counts["ct_ct_mult"] += n
                elif "ct" in (kx, ky):
                    t.counts["ct_pt_mult"] += n
            elif "ct" in (kx, ky):
                t.counts[op] += n
            return result(f(s, x, y), "ct" if "ct" in (kx, ky) else "pt", n)
        return wrapper

    patch("_make_multiply", binary("mul"))
    patch("_make_add", binary("add"))
    patch("_make_subtract", binary("sub"))

    def unary(op, weight=1):
        def wrapper(f, s, x, *rest):
            kind, n = t.unary(op, x, weight) if op else t.info(x)
            return result(f(s, x, *rest), kind, n)
        return wrapper

    def rotation(f, s, x, amount):
        kind, n = t.info(x)
        if kind == "ct" and amount != 0:
            t.counts["rotation"] += n
        return result(f(s, x, amount), kind, n)

    for name in ("_make_left_rotation", "_make_right_rotation", "_make_rotate2", "_make_rotate3"):
        patch(name, rotation)
    patch("_make_negate", unary("neg"))
    patch("_make_rescale", unary("rescale"))
    patch("_make_double_rescale", unary("double_rescale"))
    patch("_make_modswitch", unary("modswitch"))
    patch("_make_conjugate", unary("conjugation"))
    patch("_make_conjugate2", unary("conjugation"))
    patch("_make_ephemeral", unary("to_ephemeral"))
    patch("_make_bootstrap_modraise", unary("bootstrap"))

    def relinearize(f, s, x, *rest):
        kind, n = t.unary("relinearize", x)
        if rest and rest[0]:
            t.counts["rescale"] += n * rest[0]
        return result(f(s, x, *rest), kind, n)

    patch("_make_relinearize", relinearize)
    patch("_make_relinearize2", unary("relinearize"))
    patch("_make_relinearize3", unary("relinearize"))

    def hoisted(f, s, x, rotations):
        kind, n = t.info(x)
        if kind == "ct":
            t.counts["rotation"] += n * sum(1 for r in rotations if r != 0)
        return [result(term, kind, n) for term in f(s, x, rotations)]

    def rotate_accumulate(f, s, x, rotations):
        kind, n = t.info(x)
        if kind == "ct":
            t.counts["rotation"] += n * sum(1 for r in rotations if r != 0)
            t.counts["add"] += n * (len(rotations) - 1)
        return result(f(s, x, rotations), kind, n)

    def rotate_accumulate_many(f, s, xs, rotations):
        if any(t.info(x)[0] == "ct" for x in xs):
            t.counts["rotation"] += sum(1 for r in rotations if r != 0)
            t.counts["add"] += len(xs) - 1
        return result(f(s, xs, rotations), "ct", 1)

    def bsgs(f, s, x, plaintexts, baby, giant, rescale_levels):
        kind, n = t.info(x)
        if kind == "ct":
            t.counts["ct_pt_mult"] += n * len(plaintexts)
            t.counts["rotation"] += n * (sum(1 for r in baby if r != 0) + sum(1 for r in giant if r != 0))
            t.counts["add"] += n * (len(plaintexts) - 1)
            if rescale_levels == 1:
                t.counts["rescale"] += n
            elif rescale_levels == 2:
                t.counts["double_rescale"] += n
        return result(f(s, x, plaintexts, baby, giant, rescale_levels), kind, n)

    def mul_rot_acc(f, s, x, plaintexts, rotations):
        kind, n = t.info(x)
        if kind == "ct":
            t.counts["ct_pt_mult"] += n * len(plaintexts)
            t.counts["rotation"] += n * sum(1 for r in rotations if r != 0)
            t.counts["add"] += n * (len(plaintexts) - 1)
        return result(f(s, x, plaintexts, rotations), kind, n)

    def make_vector(f, s, terms):
        kind = "ct" if any(t.info(x)[0] == "ct" for x in terms) else "pt"
        return result(f(s, terms), kind, len(terms))

    def extract_vector(f, s, x):
        kind, _ = t.info(x)
        return [result(term, kind, 1) for term in f(s, x)]

    patch("_make_hoisted_rotate", hoisted)
    patch("_make_rotate_accumulate", rotate_accumulate)
    patch("_make_rotate_accumulate_many", rotate_accumulate_many)
    patch("_make_bsgs_multiply_accumulate", bsgs)
    patch("_make_multiply_rotate_accumulate", mul_rot_acc)
    patch("_make_rotate_multiply_accumulate", mul_rot_acc)
    patch("_make_vector", make_vector)
    patch("_make_extract_vector", extract_vector)
    try:
        yield t
    finally:
        for name, original in originals.items():
            setattr(CeriumFunction, name, original)
        c = t.counts
        c["key_switch"] = c["relinearize"] + c["rotation"] + c["conjugation"] + c["to_ephemeral"]
        for name in COUNTED:
            c.setdefault(name, 0)


_GROUP = re.compile(r'label="([a-z]+)\s*\n(.*?)"\]', re.S)


def count_compiled(dot_path, functions_path=None):
    """Instruction counts per lowered primitive, and launched function counts per type.

    Primitive names in the kernel-group graph: mul (element-wise product,
    including tensor products and key-switch inner products with evaluation
    keys), pmu (fused multiply-add by constants), add, int / ntt (inverse /
    forward NTT), bco (RNS base conversion), sud (key-switch mod-down: scale
    down by the special primes), rot (automorphism), inp (inputs).
    """
    text = open(dot_path).read()
    lines = collections.Counter()
    groups = collections.Counter()
    for kind, body in _GROUP.findall(text):
        groups[kind] += 1
        lines[kind] += sum(1 for line in body.splitlines() if line.strip())
    out = {"graph_file": str(dot_path), "groups": dict(groups), "instructions": dict(lines),
           # A key switch ends in one mod-down per output polynomial (2 per ciphertext);
           # a rotation applies the automorphism to both polynomials.
           "key_switches": lines["sud"] // 2,
           "automorphisms": lines["rot"] // 2}
    if functions_path:
        functions = open(functions_path).read()
        types = re.findall(r"^\s*\.type:\s*(\S+)", functions, re.M)
        out["launched_functions"] = dict(collections.Counter(types))
        out["launched_functions_total"] = len(types)
        # .base_conv: <bN: source prime indices : target prime indices>
        targets = set()
        for conversion in re.findall(r"<b\d+:([^>]*)>", functions):
            parts = conversion.split(":")
            if len(parts) == 2:
                targets.update(int(v) for v in parts[1].replace(",", " ").split())
        out["base_conversion_target_primes"] = sorted(targets)
    return out
