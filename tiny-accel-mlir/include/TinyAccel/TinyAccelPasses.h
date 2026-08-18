//===----------------------------------------------------------------------===//
// TinyAccelPasses.h
//===----------------------------------------------------------------------===//

#ifndef TINYACCEL_TINYACCELPASSES_H
#define TINYACCEL_TINYACCELPASSES_H

namespace tinyaccel {
void registerConvertArithToTinyAccelPass();
void registerFuseMulAddPass();
void registerEmitSimpleISAPass();
void registerAllPasses();
} // namespace tinyaccel

#endif // TINYACCEL_TINYACCELPASSES_H
