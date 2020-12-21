# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import sys
sys.path.append('..')

import tvm
from tvm import relay
from tvm import rpc
from tvm.relay import transform
from tvm.contrib import graph_runtime
from tvm.relay.op.contrib import arm_compute_lib
from tvm.relay.op.contrib.register import get_pattern_table
from tvm.contrib import utils
from tvm.autotvm.measure import request_remote

from common.infrastructure import Device

# Device.target = "llvm -mattr=+neon"
Device.target = "llvm"

def skip_runtime_test():
    """Skip test if it requires the runtime and it's not present."""
    # BNNS codegen not present.
    if not tvm.get_global_func("relay.ext.bnns", True):
        print("Skip because BNNS codegen is not available.")
        return True
    return False


def skip_codegen_test():
    """Skip test if it requires the BNNS codegen and it's not present."""
    if not tvm.get_global_func("relay.ext.bnns", True):
        print("Skip because BNNS codegen is not available.")
        return True


def build_module(mod, target, params=None, enable_bnns=True, tvm_ops=0, acl_partitions=1):
    """Build module with option to build for BNNS."""
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        if enable_bnns:
            target_annotation_pass = tvm.transform.Sequential(
                [
                    transform.InferType(),
                    transform.FoldConstant(),
                    transform.FoldScaleAxis(),
                    transform.MergeComposite(get_pattern_table('bnns')),
                    transform.AnnotateTarget('bnns'),
                    transform.MergeCompilerRegions(),
                    transform.PartitionGraph(),
                ]
            )
            mod = target_annotation_pass(mod)
        relay.backend.compile_engine.get().clear()
        return relay.build(mod, target=target, params=params)


def update_lib(lib, device, cross_compile):
    """Export the library to the remote/local device."""
    lib_name = "mod.so"
    temp = utils.tempdir()
    lib_path = temp.relpath(lib_name)
    if cross_compile:
        lib.export_library(lib_path, cc=cross_compile)
    else:
        lib.export_library(lib_path)
    device.upload(lib_path)
    lib = device.load_module(lib_name)
    return lib


