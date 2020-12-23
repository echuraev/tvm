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
# pylint: disable=invalid-name, unused-argument
"""BNNS library supported operators.
Is a part of Accelerate framework on macOS/iOS platforms. Apple provide several APIs
to handle tensor processing. Particularly:
 * BNNS (basic neural )
 * vDSP (1D and 2D tensor processing)
 * BLAS (gemm provide)

# There are two ways to registering a function for an op to indicate if it is
# supported by DNNL.

# - The first and simplest way is to use the helper so that
# users only need to provide the operator name and a boolean value to indicate if
# it is supported. For example:
#
#     .. code-block:: python
#
#       add = _register_external_op_helper("add")
#       add = _register_external_op_helper("add", True)
#       add = _register_external_op_helper("add", False)
#
# - The other way is to implement the function by themselves to
# check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import tvm.ir
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

# Old style BNNS API are used for iOS < 14 and macOS < 11
# TODO [xpeskov]: OS version should be extracted from target
use_old_bnns_api = True

def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by BNNS.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by BNNS.
    """

    @tvm.ir.register_op_attr(op_name, "target.bnns")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


# _register_external_op_helper("nn.batch_norm")
# _register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.dense")
# _register_external_op_helper("nn.relu")
# _register_external_op_helper("add")
# _register_external_op_helper("subtract")
# _register_external_op_helper("multiply")


# TODO [apeskov]: enlarge list of supported types
#                 plus clarify meaning of "" value
def dtype_is_supported(dtype):
    return dtype == "float32" or dtype == ""

@tvm.ir.register_op_attr("nn.conv2d", "target.bnns")
def conv2d(expr):
    """Check if the conv2d can be executed in BNNS."""
    attrs, args = expr.attrs, expr.args
    if use_old_bnns_api:
        if attrs.groups != 1:
            return False
        if attrs.dilation[0] != 1 or attrs.dilation[1] != 1:
            return False
        data_typ = args[0].checked_type
        if len(data_typ.shape) != 4 or data_typ.dtype != "float32":
            return False
        kernel_typ = args[1].checked_type
        if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "float32":
            return False
        # Asymetric pad case is not supported
        if attrs.padding[0] != attrs.padding[2] or attrs.padding[1] != attrs.padding[3]:
            return False

    if attrs.data_layout != "NCHW":
        return False
    if not dtype_is_supported(attrs.out_dtype):
        return False
    return True


def make_conv_relu_pattern(with_bias=True):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    return is_op("nn.relu")(conv_out)


def check_conv(extract):
    """Check conv pattern is supported by BNNS."""
    call = extract
    while call.op.name != "nn.conv2d":
        call = call.args[0]
    return conv2d(call)


@register_pattern_table("bnns")
def pattern_table():
    conv2d_bias_relu_pat = ("bnns.conv2d_bias_relu", make_conv_relu_pattern(with_bias=True), check_conv)
    conv2d_relu_pat = ("bnns.conv2d_relu", make_conv_relu_pattern(with_bias=False), check_conv)
    bnns_patterns = [conv2d_bias_relu_pat, conv2d_relu_pat]
    return bnns_patterns
