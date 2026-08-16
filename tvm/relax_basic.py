"""Minimal Relax IR example: (x + y) * (x - y)."""

from tvm.script import ir as I
from tvm.script import relax as R


@I.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor((1, 3), "float32"),
        y: R.Tensor((1, 3), "float32"),
    ) -> R.Tensor((1, 3), "float32"):
        add = R.add(x, y)  # x + y
        sub = R.subtract(x, y)  # x - y
        z = R.multiply(add, sub)  # (x + y) * (x - y)
        return z


if __name__ == "__main__":
    print("Relax IR：")
    print(Module)
