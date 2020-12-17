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

from itertools import zip_longest, combinations
import json
import os
import warnings

import numpy as np

import tvm
from tvm import relay
from tvm import rpc
from tvm.contrib import graph_runtime
from tvm.relay.op.contrib import arm_compute_lib
from tvm.contrib import utils
from tvm.autotvm.measure import request_remote

from common.infrastructure import Device, get_cpu_op_count

Device.target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"

def skip_runtime_test():
    """Skip test if it requires the runtime and it's not present."""
    # ACL codegen not present.
    if not tvm.get_global_func("relay.ext.arm_compute_lib", True):
        print("Skip because Arm Compute Library codegen is not available.")
        return True

    # Remote device is in use or ACL runtime not present
    # Note: Ensure that the device config has been loaded before this check
    if (
        not Device.connection_type != "local"
        and not arm_compute_lib.is_arm_compute_runtime_enabled()
    ):
        print("Skip because runtime isn't present or a remote device isn't being used.")
        return True


def skip_codegen_test():
    """Skip test if it requires the ACL codegen and it's not present."""
    if not tvm.get_global_func("relay.ext.arm_compute_lib", True):
        print("Skip because Arm Compute Library codegen is not available.")
        return True


def build_module(mod, target, params=None, enable_acl=True, tvm_ops=0, acl_partitions=1):
    """Build module with option to build for ACL."""
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        if enable_acl:
            mod = arm_compute_lib.partition_for_arm_compute_lib(mod, params)
            tvm_op_count = get_cpu_op_count(mod)
            assert tvm_op_count == tvm_ops, "Got {} TVM operators, expected {}".format(
                tvm_op_count, tvm_ops
            )
            partition_count = 0
            for global_var in mod.get_global_vars():
                if "arm_compute_lib" in global_var.name_hint:
                    partition_count += 1

            assert (
                acl_partitions == partition_count
            ), "Got {} Arm Compute Library partitions, expected {}".format(
                partition_count, acl_partitions
            )
        relay.backend.compile_engine.get().clear()
        return relay.build(mod, target=target, params=params)


def extract_acl_modules(module):
    """Get the ACL module(s) from llvm module."""
    return list(
        filter(lambda mod: mod.type_key == "arm_compute_lib", module.get_lib().imported_modules)
    )


def verify_codegen(
    module,
    known_good_codegen,
    num_acl_modules,
    tvm_ops=0,
    target="llvm -mtriple=aarch64-linux-gnu -mattr=+neon",
):
    """Check acl codegen against a known good output."""
    module = build_module(module, target, tvm_ops=tvm_ops, acl_partitions=num_acl_modules)
    acl_modules = extract_acl_modules(module)

    assert len(acl_modules) == num_acl_modules, (
        f"The number of Arm Compute Library modules produced ({len(acl_modules)}) does not "
        f"match the expected value ({num_acl_modules})."
    )

    for mod in acl_modules:
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
