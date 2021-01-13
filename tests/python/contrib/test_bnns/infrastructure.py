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

import json

import tvm
from tvm import relay
from tvm import rpc
from tvm.relay import transform
from tvm.contrib import graph_runtime
from tvm.relay.op.contrib.register import get_pattern_table
from tvm.relay.op.contrib.bnns import partition_for_bnns
from tvm.contrib import utils
from tvm.autotvm.measure import request_remote

from common.infrastructure import Device

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
    with tvm.transform.PassContext(opt_level=3):
        if enable_bnns:
            mod = partition_for_bnns(mod)
        relay.backend.compile_engine.get().clear()
        return relay.build(mod, target=target, target_host=target, params=params)


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


def extract_bnns_modules(module):
    """Get the BNNS module(s) from llvm module."""
    return list(
        filter(lambda mod: mod.type_key == "bnns_json", module.get_lib().imported_modules)
    )


def verify_codegen(
    module,
    known_good_codegen,
    num_bnns_modules,
    tvm_ops=0,
    target=Device.target,
):
    """Check BNNS codegen against a known good output."""
    module = build_module(module, target, tvm_ops=tvm_ops)
    bnns_modules = extract_bnns_modules(module)

    assert len(bnns_modules) == num_bnns_modules, (
        f"The number of BNNS modules produced ({len(bnns_modules)}) does not "
        f"match the expected value ({num_bnns_modules})."
    )

    for mod in bnns_modules:
        source = mod.get_source("json")
        codegen = json.loads(source)["nodes"]
        # remove input and const names as these cannot be predetermined
        for node in range(len(codegen)):
            if codegen[node]["op"] == "input" or codegen[node]["op"] == "const":
                codegen[node]["name"] = ""
        codegen_str = json.dumps(codegen, sort_keys=True, indent=2)
        known_good_codegen_str = json.dumps(known_good_codegen, sort_keys=True, indent=2)

        assert codegen_str == known_good_codegen_str, (
            f"The JSON produced by codegen does not match the expected result. \n"
            f"Actual={codegen_str} \n"
            f"Expected={known_good_codegen_str}"
        )
