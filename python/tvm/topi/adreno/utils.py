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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return
"""util functions to be reused in different compute/schedule on Qualcomm Adreno GPU"""

from re import S
import tvm
from tvm import te
from tvm.topi.utils import simplify
from tvm.topi import nn
from ..utils import get_const_tuple
import numpy


def getDiv(value, start):
    div = 1
    for d in range(start,0,-1):
        if (value % d) == 0:
            div = d
            break
    return div


def split_to_chunks(trip_count, block):
    tail = trip_count % 4
    chunks = trip_count // 4
    if tail == 0:
        tail = 4
    else:
        chunks += 1
    return chunks, block, tail


def pack_input(Input, layout, batch, in_channel_chunks, in_channel_block, in_channel_tail, in_height, in_width):
    pad_value = tvm.tir.const(0, Input.dtype)
    def _reorder_data_nchw(*indices):
        condition = []
        condition.append(indices[1] == in_channel_chunks - 1)
        condition.append(indices[4] >= in_channel_tail)
        condition = tvm.tir.all(*condition)
        return tvm.tir.if_then_else(
                condition,
                pad_value,
                Input[indices[0],indices[1] * in_channel_block + indices[4], indices[2], indices[3]])

    def _reorder_data_nhwc(*indices):
        condition = []
        condition.append(indices[3] == in_channel_chunks - 1)
        condition.append(indices[4] >= in_channel_tail)
        condition = tvm.tir.all(*condition)
        return tvm.tir.if_then_else(
                condition,
                pad_value,
                Input[indices[0],indices[1], indices[2], indices[3] * in_channel_block + indices[4]])

    # compute:
    if layout == "NCHW":
        reordered_data = te.compute(
            [batch, in_channel_chunks, in_height, in_width, in_channel_block],
            _reorder_data_nchw,
            name="input_pack",
            tag="input_pack",
        )
    elif layout == "NHWC":
        reordered_data = te.compute(
            [batch, in_height, in_width, in_channel_chunks, in_channel_block],
            _reorder_data_nhwc,
            name="input_pack",
            tag="input_pack",
        )
    else:
        assert False, "Adreno util function pack_input does not accept unknown layout"
    return reordered_data


def pack_filter(Filter,
                layout,
                out_channel_chunks, out_channel_block, out_channel_tail,
                in_filter_channels,
                in_data_channel_chunks, in_data_channel_block, in_data_channel_tail,
                kernel_h, kernel_w):
    pad_value = tvm.tir.const(0, Filter.dtype)
    def _reorder_weights_depthwise_oihw(*indices):
        conditionA = []
        conditionA.append(indices[0] == out_channel_chunks - 1)
        conditionA.append(indices[4] >= out_channel_tail)
        conditionAT = tvm.tir.all(*conditionA)

        return tvm.tir.if_then_else(
                conditionAT,
                pad_value,
                Filter[indices[0] * out_channel_block + indices[4], indices[1], indices[2], indices[3]])

    def _reorder_weights_depthwise_hwoi(*indices):
        conditionA = []
        conditionA.append(indices[2] == out_channel_chunks - 1)
        conditionA.append(indices[4] >= out_channel_tail)
        conditionAT = tvm.tir.all(*conditionA)

        return tvm.tir.if_then_else(
                conditionAT,
                pad_value,
                Filter[indices[0], indices[1], indices[2] * out_channel_block + indices[4], indices[3]])

    def _reorder_weights_oihw(*indices):
        conditionA = []
        conditionA.append(indices[0] == out_channel_chunks - 1)
        conditionA.append(indices[4] >= out_channel_tail)
        conditionAT = tvm.tir.all(*conditionA)

        conditionO = []
        conditionO.append(conditionAT)
        conditionO.append(indices[1] >= in_data_channel_chunks * in_data_channel_block + in_data_channel_tail)
        conditionOT = tvm.tir.any(*conditionO)
        return tvm.tir.if_then_else(
                conditionOT,
                pad_value,
                Filter[indices[0] * out_channel_block + indices[4], indices[1], indices[2], indices[3]])

    def _reorder_weights_hwio(*indices):
        conditionA = []
        conditionA.append(indices[3] == out_channel_chunks - 1)
        conditionA.append(indices[4] >= out_channel_tail)
        conditionAT = tvm.tir.all(*conditionA)

        conditionO = []
        conditionO.append(conditionAT)
        conditionO.append(indices[2] >= in_data_channel_chunks * in_data_channel_block + in_data_channel_tail)
        conditionOT = tvm.tir.any(*conditionO)
        return tvm.tir.if_then_else(
                conditionOT,
                pad_value,
                Filter[indices[0], indices[1], indices[2], indices[3] * out_channel_block + indices[4]])

    if in_filter_channels == 1:
        if layout == "OIHW":
            reordered_filter = te.compute(
                [out_channel_chunks, in_filter_channels, kernel_h, kernel_w, out_channel_block],
                _reorder_weights_depthwise_oihw,
                name="filter_pack",
                tag="filter_pack",
            )
        elif layout == "HWOI":
            reordered_filter = te.compute(
                [kernel_h, kernel_w, out_channel_chunks, in_filter_channels, out_channel_block],
                _reorder_weights_depthwise_hwoi,
                name="filter_pack",
                tag="filter_pack",
            )
        else:
            assert False, "Adreno util function def pack_filter does not accept unknown layout"
    else:
        if layout == "OIHW":
            reordered_filter = te.compute(
                [out_channel_chunks, in_filter_channels, kernel_h, kernel_w, out_channel_block],
                _reorder_weights_oihw,
                name="filter_pack",
                tag="filter_pack",
            )
        elif layout == "HWIO":
            reordered_filter = te.compute(
                [kernel_h, kernel_w, in_filter_channels, out_channel_chunks, out_channel_block],
                _reorder_weights_hwio,
                name="filter_pack",
                tag="filter_pack",
            )
        else:
            assert False, "Adreno util function def pack_filter does not accept unknown layout"
    return reordered_filter


def expand_spatial_dimensions(in_height, in_width,
                              kernel_h, kernel_w,
                              dilation_h, dilation_w,
                              padding,
                              stride_h, stride_w):
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    out_height_orig = out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width_orig = out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # can output shape be divded by 2 or even 4?
    # if it cannot be divided, need to extend for further help with split
    # theortically there should be addition padding for inputs, but it will be optimized by
    # cache_read InferBound. We must proceed pad here exactly to produce tensor which is
    # required for calculation of original out size, not more! In other case intermediate
    # tensor might be allcoated with less sizes while compute will try to fill the expanded
    # one - data discrepancy as a result
    # And in case of textures it is not a problem if we provide texture of less size because
    # 1. It is not important which values would be for extra calc - these calculations are
    #    required only for better utilizatin of GPU fit to working groups
    # 2. When we request pixel out opf bound, texture will handle this correctly. As mentioned
    #    above, the value itself is not important
    if out_height % 2 != 0:
        out_height += 1
    if out_width % 2 != 0:
        out_width += 1

    if out_height % 4 != 0:
        out_height += 2
    if out_width % 4 != 0:
        out_width += 2
    return out_height_orig, out_height, out_width_orig, out_width 


def add_pad(data,
            layout,
            in_height, in_width,
            out_height_orig, out_width_orig,
            kernel_h, kernel_w,
            dilation_h, dilation_w,
            padding,
            stride_h, stride_w):
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    # compute graph
    if layout == "NCHW":
        y_axis = 2
        x_axis = 3
    elif layout == "NHWC":
        y_axis = 1
        x_axis = 2
    else:
        assert False, "not supported layout in adrenno util add_pad"
    pad_before = [0, 0, 0, 0, 0]
    pad_after = [0, 0, 0, 0, 0]
    pad_before[y_axis] = pad_top
    pad_before[x_axis] = pad_left
    pad_after[y_axis] = pad_down
    pad_after[x_axis] = pad_right

    # calculation of real used input size:
    input_latest_w = (out_width_orig - 1) * stride_w + (kernel_w - 1) * dilation_w + 1
    input_latest_h = (out_height_orig - 1) * stride_h + (kernel_h - 1) * dilation_h + 1
    if input_latest_w < in_width + pad_before[x_axis] + pad_after[x_axis]:
        pad_after[x_axis] -= in_width + pad_before[x_axis] + pad_after[x_axis] - input_latest_w
    if input_latest_h < in_height + pad_before[y_axis] + pad_after[y_axis]:
        pad_after[y_axis] -= in_height + pad_before[y_axis] + pad_after[y_axis] - input_latest_h
    return nn.pad(data, pad_before, pad_after, name="pad_temp")


def bind_data_copy(stage, axis_to_vectorize = None):
    shape = get_const_tuple(stage.op.output(0).shape)
    if axis_to_vectorize != None and len(shape) == 4 and shape[axis_to_vectorize] % 4 == 0:
          ax0, ax1, ax2, ax3 = stage.op.axis
          if axis_to_vectorize == 1:
            oax1, iax1 = stage.split(ax1, factor=4)
            stage.reorder(ax0, oax1, ax2, ax3, iax1)
            stage.vectorize(iax1)
            fused = stage.fuse(ax0, oax1, ax2, ax3)
          elif axis_to_vectorize == 3:
            oax3, iax3 = stage.split(ax3, factor=4)
            stage.reorder(ax0, ax1, ax2, oax3, iax3)
            stage.vectorize(iax3)
            fused = stage.fuse(ax0, ax1, ax2, oax3)

          ftc = numpy.prod(shape) / 4
          div = getDiv(ftc, 128)
          block, thread = stage.split(fused, factor=div)

          stage.bind(block, te.thread_axis("blockIdx.z"))
          stage.bind(thread, te.thread_axis("threadIdx.z"))
    else:
        axes = stage.op.axis
        fused = stage.fuse(*axes[:-1])
        if shape[-1] <= 32:
            ftc = numpy.prod(shape[:-1])
            div = getDiv(ftc, 64)
            block, thread = stage.split(fused, factor=div)
            stage.bind(block, te.thread_axis("blockIdx.x"))
            stage.bind(thread, te.thread_axis("threadIdx.x"))
            if shape[-1] == 4:
                stage.vectorize(axes[-1])
        else:
            stage.bind(fused, te.thread_axis("blockIdx.x"))
            stage.bind(*axes[-1:], te.thread_axis("threadIdx.x"))

def get_texture_storage(shape):
    limit = 16384
    if shape[0] * shape[1] * shape[2] < limit and shape[3] < limit:
        return "global.texture"
    elif shape[0] * shape[1] < limit and shape[2] * shape[3] < limit:
        return "global.texture-nhwc"
    else:
        return "global.texture-weight"
