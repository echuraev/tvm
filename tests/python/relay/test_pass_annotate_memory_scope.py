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
import numpy as np
import tvm
from tvm import target
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload


GPU_DEVICE = tvm.device("opencl")
HOST_TARGET = tvm.target.Target("llvm")
GPU_TARGET = tvm.target.Target("opencl --device=adreno").with_host(HOST_TARGET)
GPU = tvm.target.VirtualDevice(GPU_DEVICE, GPU_TARGET)


def run_opt_passes(expr, opt_passes):
    for opt_pass in opt_passes:
        assert isinstance(opt_pass, tvm.transform.Pass)
    CTXT = tvm.transform.PassContext(config={"relay.fallback_device_type": GPU.device_type_int})
    config = tvm.target.make_compilation_config(CTXT, GPU_TARGET)

    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    # PlanDevices should succeed.
    mod = relay.transform.PlanDevices(config)(mod)

    for opt_pass in opt_passes:
        mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_annotation_pass():
    dtype = "float32"
    input_shape = (1, 3, 224, 224)
    filter_shape = (64, 3, 7, 7)
    filter_shape_block = (16, 3, 7, 7, 4)

    def before():
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        D = relay.nn.conv2d(
            A,
            B,
            padding=[3, 3, 3, 3],
            strides=[2, 2],
            channels=64,
            kernel_size=[7, 7],
            out_dtype=dtype,
        )

        D = relay.nn.max_pool2d(D, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1])
        return relay.Function([A, B], D)

    #def expected():
    #    A = relay.var("data", shape=input_shape, dtype=dtype)
    #    B = relay.var("weight", shape=filter_shape, dtype=dtype)

    #    # layout_transform
    #    x = relay.var("p0", shape=filter_shape, dtype=dtype)
    #    y = relay.layout_transform(x, "OIHW", "OIWH4o")
    #    y = relay.Function([x], y)
    #    D = relay.Call(y, [B])

    #    # conv2d
    #    x = relay.var("p0", shape=input_shape, dtype=dtype)
    #    w = relay.var("p1", shape=filter_shape_block, dtype=dtype)
    #    y = relay.nn.conv2d(
    #        x,
    #        w,
    #        padding=[3, 3, 3, 3],
    #        strides=[2, 2],
    #        channels=64,
    #        kernel_size=[7, 7],
    #        data_layout="NCHW",
    #        kernel_layout="OIHW4o",
    #        out_layout="NCHW4c",
    #        out_dtype=dtype,
    #    )
    #    y = relay.Function([x, w], y)
    #    D = relay.Call(y, [A, D])


    #    vd1 = tvm.target.VirtualDevice(GPU_DEVICE, GPU_TARGET, "global")
    #    vd2 = tvm.target.VirtualDevice(GPU_DEVICE, GPU_TARGET, "global.texture")
    #    D = relay.annotation.on_device(D, vd1)
    #    D = relay.op.device_copy(D, vd1, vd2)
    #    D = relay.annotation.on_device(D, vd2)

    #    # max_pool2d
    #    x = relay.var("p0", shape=(1, 16, 112, 112, 4), dtype=dtype)
    #    y = relay.nn.max_pool2d(x, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1])
    #    y = relay.Function([x], y)
    #    D = relay.Call(y, [D])

    #    # layout_transform
    #    x = relay.var("p0", shape=(1, 16, 56, 56, 4), dtype=dtype)
    #    y = relay.layout_transform(x, "NCHW4c", "NCHW")
    #    y = relay.Function([x], y)
    #    D = relay.Call(y, [D])

    #    return relay.Function([A, B], D)

    # the fold constant should work on any context.
    with tvm.transform.PassContext(opt_level=3):
        with tvm.target.Target("opencl --device=adreno"):
            zz = run_opt_passes(before(), [
                transform.AlterOpLayout(),
                transform.FuseOps(),
                transform.AnnotateMemoryScope(),
                transform.InferType(),
            ])
    print(zz)
    #zexpected = tvm.IRModule.from_expr(expected())
    #zexpected = relay.transform.InferType()(zexpected)["main"]
    #print("********************")
    #print(zexpected)
    #ip = tvm.ir.base.get_first_structural_mismatch(zz, zexpected)
    #print(ip)
    ##tvm.ir.assert_structural_equal(zz, zexpected)

