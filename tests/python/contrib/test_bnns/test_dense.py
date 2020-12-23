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
"""Arm Compute Library integration dense tests."""

import numpy as np

import tvm
from tvm import relay
from tvm import testing
from common.infrastructure import (
    build_and_run,
    verify,
    generate_trials,
)
from .infrastructure import (
    Device,
    skip_runtime_test,
    skip_codegen_test,
    build_module,
)


def _get_model(shape, weight_shape, units, dtype, var_names, has_bias=False):
    """Return a model and any parameters it may have"""
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.dense(a, weights, units=units, out_dtype=dtype)
    params = {"w": w}
    if has_bias:
        b = tvm.nd.array(np.random.randint(-128, 127, weight_shape[0]).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.nn.bias_add(out, biasc)
        params["b"] = b
    return out, params


def test_dense():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    dtype = ["float32"]
    shape = [
        ((1, 128), (16, 128), 16),
        ((32, 32), (32, 32), 32),
        ((1, 64), (1, 64), 1),
        ((11, 2), (2, 2), 2),
        ((2, 2), (1, 2), 1),
    ]
    composite = [False, True]
    trials = generate_trials([dtype, shape, composite], 3)

    for dtype, (shape, weight_shape, units), composite in trials:
        outputs = []
        inputs = {"a": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype))}
        func, params = _get_model(
            shape, weight_shape, units, dtype, var_names=iter(inputs), has_bias=composite
        )
        for bnns in [False, True]:
            outputs.append(
                build_and_run(
                    func,
                    inputs,
                    1,
                    params,
                    device,
                    build_module,
                    enable_framework=bnns,
                )[0]
            )

        config = {
            "shape": shape,
            "weight_shape": weight_shape,
            "units": units,
            "dtype": dtype,
            "composite operators (bias)": composite,
        }
        verify(outputs, atol=0.001, rtol=0.01, config=config)


if __name__ == "__main__":
    test_dense()
