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
"""conv2d nchw schedule on Qualcomm Adreno GPU"""
import tvm
from tvm import te
from tvm import autotvm

from ..utils import get_const_tuple, traverse_inline
from .utils import (
    split_to_chunks,
    pack_input,
    pack_filter,
    expand_spatial_dimensions,
    add_pad,
    bind_data_copy,
    get_default_conv2d_config,
    get_texture_storage,
)


@autotvm.register_topi_schedule("conv2d_nchwc.image2d")
def schedule_conv2d_nchwc(cfg, outs):
    """Create the schedule for conv2d_nchw"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "adreno_conv2d_latest_op":
            schedule_conv2d_NCHWc_KCRSk(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nchwc.image2d")
def conv2d_nchwc(cfg, Input, Filter, stride, padding, dilation, out_dtype):
    """
    Convolution operator in NCHWc layout.
    Algo:
      1. Convert into blocked format if we have 4d original tensor.
         In case of AutoTVM we override the convert by just tensors since such conversion
         will be absent for real blocked convolution, no sense to include into tuning
      2. Expand spatial dimensions to have width and height be dividable by factor 4
         This leads to slightly bigger amount of compute but allow utilize GPU much better
      3. Add paddings. This happens even if we do not need pad originaly. This is useful
         due to work arounding of the gaps of texture annotation between Primary Functions
         and limited support of textures in schedules. Later on this pad will be executed
         separately and will produce texture
      4. 5d Convolution compute with accumulating into out_dtype
      5. Cast to the origin output data type
      6. For case of 4d convolution: convert of output from 5d to 4d
    """

    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    out_channel, in_filter_channels, kernel_h, kernel_w = Filter.shape

    out_height_orig, out_height, out_width_orig, out_width = expand_spatial_dimensions(
        in_height, in_width, kernel_h, kernel_w, dilation_h, dilation_w, padding, stride_h, stride_w
    )

    temp = add_pad(
        Input,
        "NCHW",
        out_height_orig,
        out_width_orig,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        padding,
        stride_h,
        stride_w,
    )

    rcc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    conv = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, fc, yy, xx: te.sum(
            (
                temp[nn, rcc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w]
                * Filter[fc, rcc, ry, rx]
            ).astype(out_dtype),
            axis=[rcc, ry, rx],
        ),
        tag="conv2d_nchwc",
    )
    return te.compute(
        (batch, out_channel, out_height_orig, out_width_orig),
        lambda n, ffc, y, x: conv[n, ffc, y, x,].astype(out_dtype),
        tag="adreno_conv2d_latest_op",
    )


def schedule_conv2d_NCHWc_KCRSk(cfg, s, output):
    """
    schedule optimized for batch size = 1

    Algo:
    1. Split output axis to three parts: global work size, vthread, local worksize.
       The limitations for tuning includes heuristics from some tuned networks to limit
       search space and not pay much time for useles configurations.
    2. In case of 4d convolution schedule copying of the input (and filter) into
      5d tensors
    4. pad should be scheduled separately to create independent opencl kernel. If pad is
       inlined into convolution, this gives 1.5x performance drop
    5. We are using cache_read for intermediate tensors to produce texture and guarantee
       the best performance on the next stage.
       The weights are managed through static texture planning mechanism and guarantied come
       in texture memory scope.
       Thus way we are calling cache_read only for data tensor
    6. For 5d convolution we schedule the latest op with binding 5d axis and vectorize
       for textures
       For 4d tensor we are doing the same for the latest blocked stage, i.e. conversion
       of data type
    7. In case of 4d conv we need to schedule postops as well
    """
    latest = s.outputs[0].output(0)
    conv = output.op.input_tensors[0]
    latest_blocked = latest

    pad_data, kernel = s[conv].op.input_tensors
    input = s[pad_data].op.input_tensors[0]

    print("-" * 10)
    print("1. pad_data.shape: ", pad_data.shape)
    transform_data = lambda n, c, h, w: [n, c //4, h, w, c%4]
    pad_data_t = s[input].transform_layout(transform_data)
    pad_data_t = s[pad_data].transform_layout(transform_data)
    print("2. pad_data.shape: ", pad_data_t)
    print("-" * 10)
    print("-" * 10)
    print("1. kernel.shape: ", kernel.shape)
    transform_weights = lambda o, i, h, w: [o // 4, i, h, w, o%4]
    kernel_t = s[kernel].transform_layout(transform_weights)
    print("2. kernel.shape: ", kernel_t)
    print("-" * 10)

    print("<" * 10)
    print(s[kernel].layout_transforms)
    print(">" * 10)
    ##### space definition begin #####
    c_n, c_fc, c_y, c_x, c_fb = s[conv].transform_layout(transform_data)
    rcc, ry, rx = s[conv].op.reduce_axis

    if conv.shape[1] % 2 == 0:
        min_threads_div = 2
    else:
        min_threads_div = 1
    cfg.define_split(
        "tile_fc",
        c_fc,
        num_outputs=3,
        filter=lambda entity: entity.size[1] <= 8
        and entity.size[2] >= min_threads_div
        and entity.size[2] < 256,
    )
    cfg.define_split(
        "tile_y",
        c_y,
        num_outputs=3,
        filter=lambda entity: entity.size[1] <= 8 and entity.size[2] <= 16,
    )
    cfg.define_split(
        "tile_x",
        c_x,
        num_outputs=3,
        filter=lambda entity: entity.size[1] <= 8 and entity.size[2] <= 16,
    )

    cfg.define_split("tile_rcc", rcc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    cfg.multi_filter(
        filter=lambda entity: (  # pylint: disable=chained-comparison
            entity["tile_fc"].size[1] * entity["tile_y"].size[1] * entity["tile_x"].size[1]
        )
        <= 24
        and 32
        <= (entity["tile_fc"].size[2] * entity["tile_y"].size[2] * entity["tile_x"].size[2])
        < 1024
    )
    if cfg.is_fallback:
        get_default_conv2d_config(cfg, conv.shape[1], conv.shape[2], conv.shape[3])
    ##### space definition end #####

    # There are several conditions that have to be handled:
    # 1. If we are in the tuning, we always add cache read for data to main conv kernel
    #    to get texture in tuning opencl kernel
    # 2. If we are repacking input in runtime, we should always explicit schedule this one more
    #    stage of data copy from 4d to 5d (referred as pack_data).
    # 3. If we have pad (independently if we have runtime repack or not) we should inline it in the
    #    cache_read("texture")
    if autotvm.GLOBAL_SCOPE.in_tuning:
        if "pad_temp" in pad_data.op.name:
            s[pad_data].compute_inline()

        AT = s.cache_read(pad_data, get_texture_storage(pad_data.shape), [conv])
        bind_data_copy(s[AT])
        WT = s.cache_read(kernel, get_texture_storage(kernel.shape), [conv])
        bind_data_copy(s[WT])
    elif "pad_temp" in pad_data.op.name:
        s[pad_data].compute_inline()
        # create cache stage
        ##AT = s.cache_read(pad_data, get_texture_storage(pad_data.shape), [conv])
        ##bind_data_copy(s[AT])

    s[conv].set_scope("local")
    if latest_blocked == latest and output != latest:
        s[output].compute_inline()

    # tile and bind spatial axes
    #n, fc, y, x = s[latest_blocked].op.axis
    n, fc, y, x, fb = s[latest_blocked].transform_layout(transform_data)

    kernel_scope, n = s[latest_blocked].split(n, nparts=1)

    bf, vf, tf = cfg["tile_fc"].apply(s, latest_blocked, fc)
    by, vy, ty = cfg["tile_y"].apply(s, latest_blocked, y)
    bx, vx, tx = cfg["tile_x"].apply(s, latest_blocked, x)

    bf = s[latest_blocked].fuse(n, bf)
    s[latest_blocked].bind(bf, te.thread_axis("blockIdx.z"))
    s[latest_blocked].bind(by, te.thread_axis("blockIdx.y"))
    s[latest_blocked].bind(bx, te.thread_axis("blockIdx.x"))
    s[latest_blocked].bind(vf, te.thread_axis("vthread"))
    s[latest_blocked].bind(vy, te.thread_axis("vthread"))
    s[latest_blocked].bind(vx, te.thread_axis("vthread"))
    s[latest_blocked].bind(tf, te.thread_axis("threadIdx.z"))
    s[latest_blocked].bind(ty, te.thread_axis("threadIdx.y"))
    s[latest_blocked].bind(tx, te.thread_axis("threadIdx.x"))
    s[latest_blocked].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fb)
    s[latest_blocked].vectorize(fb)

    s[conv].compute_at(s[latest_blocked], tx)

    # tile reduction axes
    n, fc, y, x, fb = c_n, c_fc, c_y, c_x, c_fb

    rcc, ry, rx = s[conv].op.reduce_axis
    rco, rci = cfg["tile_rcc"].apply(s, conv, rcc)
    ryo, ryi = cfg["tile_ry"].apply(s, conv, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, conv, rx)

    s[conv].reorder(rco, ryo, rxo, rci, ryi, rxi, n, fc, y, x, fb)
    s[conv].vectorize(fb)

    # unroll
    s[latest_blocked].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[latest_blocked].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    if latest_blocked != latest:
        s[latest].compute_root()
        bind_data_copy(s[latest], 1)
        if latest != output:
            s[output].compute_inline()

    N, OCC, OH, OW = get_const_tuple(latest_blocked.shape)
    _, IC, KH, KW = get_const_tuple(kernel.shape)
    ICKHKW = IC * KH * KW

    if isinstance(N, int):
        cfg.add_flop(2 * N * OH * OW * OCC * ICKHKW)
