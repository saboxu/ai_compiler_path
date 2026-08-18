//===----------------------------------------------------------------------===//
// Educational stub — does NOT compile standalone.
// Integrate into llvm-project/lib/Target/TinyAccel/ when building a real backend.
//===----------------------------------------------------------------------===//

#pragma once

// NOTE: Headers below exist inside an LLVM source/build tree, not in this demo.
#if 0 // illustrative only

#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"

namespace llvm {

class TinyAccelTargetMachine : public LLVMTargetMachine {
public:
  TinyAccelTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                        StringRef FS, const TargetOptions &Options,
                        std::optional<Reloc::Model> RM,
                        std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                        bool JIT);
};

// Pseudo-API close to the teaching snippet. Real registration uses
// TargetRegistry + LLVMInitializeTinyAccelTarget().
extern "C" void LLVMInitializeTinyAccelTarget() {
  // RegisterTarget ...
  // RegisterTargetMachine ...
}

} // namespace llvm

#endif

// Hardware capability cheat-sheet for the teaching accelerator:
//   - FP16: yes (planned)
//   - Scalar FP32 regs: 32
//   - Core ISA: LOAD_ARG, CONST, ADD, MUL, MAC
// Expand gradually; keep ISel patterns tiny at first.
