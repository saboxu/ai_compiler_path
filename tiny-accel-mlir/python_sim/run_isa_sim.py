#!/usr/bin/env python3
"""Tiny interpreter for the textual ISA dumped by --tinyaccel-emit-isa.

Example (after running tinyaccel-opt ... --tinyaccel-emit-isa)::

    FUNC dot_like
      r0 = MAC arg0, arg1, arg2
    ENDFUNC

    python3 python_sim/run_isa_sim.py --a 2 --b 3 --c 4
    # -> 10.0
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field


@dataclass
class FuncISA:
    name: str
    lines: list[str] = field(default_factory=list)


def parse_isa(text: str) -> list[FuncISA]:
    funcs: list[FuncISA] = []
    cur: FuncISA | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("FUNC "):
            cur = FuncISA(name=line.split(None, 1)[1])
            funcs.append(cur)
        elif line == "ENDFUNC":
            cur = None
        elif cur is not None and line and not line.startswith(";"):
            cur.lines.append(line)
    return funcs


def eval_func(fn: FuncISA, args: list[float]) -> float:
    regs: dict[str, float] = {f"arg{i}": float(v) for i, v in enumerate(args)}
    last = 0.0
    assign = re.compile(
        r"^(r\d+)\s*=\s*(CONST|ADD|MUL|MAC|LOAD_ARG)\s+(.+)$"
    )
    for line in fn.lines:
        m = assign.match(line)
        if not m:
            raise ValueError(f"bad isa line: {line}")
        dst, op, rest = m.group(1), m.group(2), m.group(3)
        if op == "CONST":
            last = float(rest)
        elif op == "LOAD_ARG":
            last = regs[f"arg{int(rest)}"]
        elif op == "ADD":
            a, b = [x.strip() for x in rest.split(",")]
            last = regs[a] + regs[b]
        elif op == "MUL":
            a, b = [x.strip() for x in rest.split(",")]
            last = regs[a] * regs[b]
        elif op == "MAC":
            a, b, c = [x.strip() for x in rest.split(",")]
            last = regs[a] * regs[b] + regs[c]
        else:
            raise ValueError(op)
        regs[dst] = last
    return last


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--isa-file", type=str, default="", help="ISA dump file")
    p.add_argument("--a", type=float, default=2.0)
    p.add_argument("--b", type=float, default=3.0)
    p.add_argument("--c", type=float, default=4.0)
    args = p.parse_args()

    if args.isa_file:
        text = open(args.isa_file, encoding="utf-8").read()
    else:
        # Default: fused form of a*b+c
        text = (
            "FUNC dot_like\n"
            "  r0 = MAC arg0, arg1, arg2\n"
            "ENDFUNC\n"
        )
    funcs = parse_isa(text)
    assert funcs, "no FUNC in ISA"
    out = eval_func(funcs[0], [args.a, args.b, args.c])
    print(f"{funcs[0].name}({args.a},{args.b},{args.c}) = {out}")


if __name__ == "__main__":
    main()
