/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/AddBlockIdForControlOps.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/InterCoreTransferAndSync.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/MarkMainLoop.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/PreserveControlAttrsCanonicalize.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/RefineArgsBlockId.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/SeparateCVScope.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflowPass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

#include "ascend/include/DynamicCVPipeline/DebugPrint.h"

#include <iostream>

static constexpr const char *DEBUG_TYPE = "SplitDataflow";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

// Run the pass
void SplitDataflowPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  OpPassManager pm(module.getOperationName());
  LDBG("Enter pass.");

  pm.addPass(createDebugPrintPass("Before AddBlockIdForControlOps Pass"));
  // Step 1: Add block_id for control flow operations
  pm.addPass(createAddBlockIdForControlOpsPass());

  pm.addPass(createDebugPrintPass("Before DataDependencyAnalysis Pass"));
  // Step 2: Analyze data dependencies between Vector and Cube blocks
  pm.addPass(createDataDependencyAnalysisPass());

  pm.addPass(createDebugPrintPass("Before InterCoreTransferAndSync Pass"));
  // Step 3: Run InterCoreTransferAndSync
  pm.addPass(createInterCoreTransferAndSyncPass());

  pm.addPass(createDebugPrintPass("Before MarkMainLoop Pass"));
  // Step 4: Mark the main computation loop
  pm.addPass(createMarkMainLoopPass());

  pm.addPass(createDebugPrintPass("Before SeparateCVScope Pass"));
  // Step 5: Run SeparateCVScope
  pm.addPass(createSeparateCVScopePass());

  pm.addPass(createDebugPrintPass("Before PreserveVectorControlAttrCanonical Pass"));
  // Step 6: Canonicalize to preserve control flow attributes
  pm.addPass(createPreserveControlAttrsCanonicalizePass());

  pm.addPass(createDebugPrintPass("Before RefineArgsBlockId Pass"));
  // Step 7: Refine block id for iteration variables in main loops
  pm.addPass(createRefineArgsBlockIdPass());
  pm.addPass(createDebugPrintPass("Before ReorderOpsByBlock Pass"));
  pm.addPass(createReorderOpsByBlockIdPass());
  pm.addPass(createDebugPrintPass("After ReorderOpsByBlock Pass"));

  if (failed(runPipeline(pm, module))) {
    if (!CVPipeline::hasFallbackAttr(module)) {
      std::cout << "[VDV DEBUG] SplitDataFlow failed" << std::endl;
      module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
    return;
  }

  LDBG("Process successfully");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createSplitDataflowPass() {
  return std::make_unique<SplitDataflowPass>();
}
} // namespace triton
} // namespace mlir
