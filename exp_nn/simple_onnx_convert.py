import onnx
import tvm
import tvm.relay as relay
from tvm.relay import transform
from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime
from tvm.relay.op.contrib.register import get_pattern_table
from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
from tvm.contrib import xcode

import numpy as np

import os
import re
import argparse

ap = argparse.ArgumentParser(description='Some simple script to convert ONNX model.')
ap.add_argument('-i',  '--input', dest='model', required=True,
                help="input ONNX model file")
ap.add_argument('-sh', '--shape', dest='shape', required=False, default='224x224',
                help="input shape (default 224x224)")
ap.add_argument('-t',  '--target', dest='target', required=False, default='macos',
                help="input model file [ios|android|host]")
ap.add_argument('-acl', dest='acl', default=False, action='store_true',
                help="use arm compute lib")
ap.add_argument('-dnnl', dest='dnnl', default=False, action='store_true',
                help="use Intel oneDNN lib")
ap.add_argument('-bnns', dest='bnns', default=False, action='store_true',
                help="use Intel oneDNN lib")
ap.add_argument('-tar', dest='tar', default=False, action='store_true',
                help="use tar export format")
ap.add_argument('-so', dest='tar', default=False, action='store_false',
                help="use share lib export format")
ap.add_argument('-v', '--verify', dest='verify', default=False, action='store_true',
                help="perform cat verification (mobilenet like)")
ap.add_argument('-p', '--print', dest='print', default=False, action='store_true',
                help="perform cat verification (mobilenet like)")
ap.add_argument('-d', '--debug', dest='debug', default=False, action='store_true',
                help="debug")
ap.add_argument('-d', '--debug', dest='debug', default=False, action='store_true',
                help="debug")

args = ap.parse_args()


def verify_with_cat(compiled_module):
    ctx = tvm.cpu(0)
    module = graph_runtime.GraphModule(compiled_module['default'](ctx))

    file_npz = np.load('imagenet_cat.npz')
    in_blob = [file_npz[name] for name in file_npz.files]
    module.set_input(0, in_blob[0])
    module.run()
    out_data = module.get_output(0).asnumpy()

    cat_label = 281
    everyone_is_cat = True
    for b in range(out_data.shape[0]):
        top1 = np.argmax(out_data[b])
        print("MB({0}) top-1 #{1}: {2:.2f} ".format(b, top1, out_data[b, top1]))
        everyone_is_cat &= top1 == cat_label

    print("GOOD scoring!!! +++ " if everyone_is_cat else "Bad scoring...  --- ")


def debug_with_cat(compiled_module, tmp_dir):
    ctx = tvm.cpu(0)
    module = debug_runtime.GraphModuleDebug(
        compiled_module["debug_create"]("default", ctx),
        [ctx],
        compiled_module.get_json(),
        dump_root=tmp_dir,
    )

    file_npz = np.load('imagenet_cat.npz')
    in_blob = [file_npz[name] for name in file_npz.files]
    module.set_input(0, in_blob[0])
    module.run()
    out_data = module.get_output(0).asnumpy()

    cat_label = 281
    everyone_is_cat = True
    for b in range(out_data.shape[0]):
        top1 = np.argmax(out_data[b])
        print("MB({0}) top-1 #{1}: {2:.2f} ".format(b, top1, out_data[b, top1]))
        everyone_is_cat &= top1 == cat_label

    print("GOOD scoring!!! +++ " if everyone_is_cat else "Bad scoring...  --- ")


def make_dirs_for(compiled_module_name):
    abs_path = os.path.abspath(compiled_module_name)
    dir_path = os.path.dirname(abs_path)
    os.makedirs(dir_path, exist_ok=True)


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    s = re.sub('[^-a-zA-Z0-9_.() ]+', '', s)
    if args.acl:
        s += '_acl'
    if args.dnnl:
        s += '_dnnl'
    if args.bnns:
        s += '_bnns'
    return s


def parse_shape(shape_string):
    shape = [int(i) for i in shape_string.split('x')]
    if len(shape) == 2:  # Assume that provided shape is only about spatial dimensions of picture
        shape = [1, 3, shape[0], shape[1]]
    return shape


def get_input_name(model):
    inputs = [node.name for node in model.graph.input]
    initializer = [node.name for node in model.graph.initializer]

    inputs = list(set(inputs) - set(initializer))
    if len(inputs) != 1:
        raise ValueError('ONNX model should contain only one input')
    return inputs[0]


def get_target(args):
    return 'llvm -mtriple=arm64-apple-darwin -mattr=+neon' if args.target == 'ios' else 'llvm -mcpu=core-avx2'


def mod_partitioning(mod):
    def annotate_with_target(_mod, target_name):
        target_annotation_pass = tvm.transform.Sequential(
            [
                transform.InferType(),
                transform.FoldConstant(),
                transform.SimplifyInference(),
                transform.FoldScaleAxis(),
                transform.MergeComposite(get_pattern_table(target_name)),
                transform.AnnotateTarget(target_name),
                transform.MergeCompilerRegions(),
                transform.PartitionGraph(),
            ]
        )
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            _mod = target_annotation_pass(_mod)
        return _mod

    if args.dnnl:
        mod = annotate_with_target(mod, 'dnnl')
    if args.bnns:
        mod = annotate_with_target(mod, 'bnns')
    if args.acl:
        mod = partition_for_arm_compute_lib(mod)
    return mod


def load_mod(mod_path):
    onnx_model = onnx.load(args.model)
    input_name = get_input_name(onnx_model)
    return relay.frontend.from_onnx(onnx_model, {input_name: parse_shape(args.shape)}, freeze_params=True)


def store_compiled_mod(compiled_mod):
    model_name = os.path.basename(args.model)
    model_name = os.path.splitext(model_name)[0]

    module_name = get_valid_filename(model_name)

    module_name += '.tar' if args.tar else '.dylib'
    compiled_module_name = args.target + '/' + module_name
    make_dirs_for(compiled_module_name)

    os_specific_args = {
        'fcompile': xcode.create_dylib,
        'arch': 'arm64',
        'sdk':  'iphoneos'
    } if args.target == "ios" else {}

    compiled_mod.export_library(compiled_module_name, **os_specific_args)
    print('Compiled module : ' + compiled_module_name)


def run_converter():
    mod, params = load_mod(args.model)
    mod = mod_partitioning(mod)

    lib = relay.build(mod, target=get_target(args), params=params)

    if args.verify:
        verify_with_cat(lib)

    if args.debug:
        debug_with_cat(lib, '/tmp/tvmdump0')

    if args.print:
        print(mod)

    store_compiled_mod(lib)


run_converter()
