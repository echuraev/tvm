import tvm
from tvm import rpc
from tvm import relay
from tvm.contrib import ndk
from tvm.runtime.vm import VirtualMachine
from tvm.contrib import graph_executor

import os
import numpy as np
import onnx

from tvm.target import Target


USE_VM = True
RUN_ON_HOST = True

rpc_key = "android"
target_c = "opencl -device=adreno"
if RUN_ON_HOST:
    target_h = "llvm -mcpu=tigerlake"
    target_h = "llvm"
else:
    target_h = "llvm -mtriple=arm64-linux-android"


def get_model():
    dtype = "float32"
    input_shape = (relay.Any(), 3, 224, 224)
    shape_dict = {
        "data": (1, 3, 224, 224),
        "weight": (64, 3, 7, 7),
    }
    filter_shape = (64, 3, 7, 7)
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

    mod = relay.Function([A, B], D)
    np.random.seed(0)
    #filter_data = np.zeros(filter_shape).astype(dtype)
    params1 = {
        #"weight": tvm.nd.array(filter_data),
    }
    module = tvm.IRModule({})
    module["main"] = mod
    return module, params1, shape_dict


def get_ssd_model():
    dtype = "float32"
    input_shape = (1, 3, 1200, 1200)
    shape_dict = {
        "data": (1, 3, 1200, 1200)
    }
    filter_shape = (64, 3, 7, 7)
    filter_shape2 = (64, 64, 3, 3)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    E = relay.var("weight2", shape=filter_shape2, dtype=dtype)

    D = relay.nn.conv2d(
        A,
        B,
        padding=[3, 3, 3, 3],
        strides=[2, 2],
        channels=64,
        kernel_size=[7, 7],
        out_dtype=dtype,
    )
    D = relay.nn.relu(D)

    MP = relay.nn.max_pool2d(D, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1])

    D = relay.nn.conv2d(
        MP,
        E,
        padding=[1, 1, 1, 1],
        channels=64,
        kernel_size=[3, 3],
        out_dtype=dtype,
    )
    #D = relay.nn.relu(D)

    #D = relay.nn.conv2d(
    #    D,
    #    E,
    #    padding=[1, 1, 1, 1],
    #    channels=64,
    #    kernel_size=[3, 3],
    #    out_dtype=dtype,
    #)
    #D = relay.add(D, MP)
    #D = relay.nn.relu(D)

    mod = relay.Function([A, B, E], D)
    np.random.seed(0)
    filter_data = np.zeros(filter_shape).astype(dtype)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "weight2": tvm.nd.array(filter_data2),
    }
    module = tvm.IRModule({})
    module["main"] = mod
    return module, params1, shape_dict


def download_model(model_url, model_file):
    print("Download model...")
    if not os.path.exists(model_file):
        import urllib.request
        urllib.request.urlretrieve(model_url, model_file)
    return model_file


def get_resnet_model():
    model_file = download_model("https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx", "resnet50-v2-7.onnx")
    print("Import model...")
    onnx_model = onnx.load(model_file)
    shape_dict = {
        "data": (1, 3, 224, 224)
    }
    model, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
    print("=" * 10)
    print(model)
    print("=" * 10)
    return model, params, shape_dict


def get_resnet_ssd_model():
    model_file = download_model("https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd/model/ssd-12.onnx", "resnet-ssd.onnx")
    print("Import model...")
    onnx_model = onnx.load(model_file)
    shape_dict = {
        "image": (1, 3, 1200, 1200)
    }
    model, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
    print("=" * 10)
    print(model)
    print("=" * 10)
    return model, params, shape_dict


def get_onnx_yolo_model():
    model_file = download_model("https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov3/model/yolov3-12.onnx", "onnx_yolov3.onnx")
    print("Import model...")
    onnx_model = onnx.load(model_file)
    shape_dict = {
        "input_1": (1, 3, 416, 416),
        "image_shape": (1, 2),
    }
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    print("=" * 10)
    print(model)
    print("=" * 10)
    return model, params, shape_dict


def get_onnx_faster_rcnn_model():
    model_file = download_model("https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx", "FasterRCNN.onnx")
    print("Import model...")
    onnx_model = onnx.load(model_file)
    shape_dict = {
        "image": (3, 800, 800),
    }
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    print("=" * 10)
    print(model)
    print("=" * 10)
    return model, params, shape_dict


def compile_model_for_vm(name, model, params, target):
    lib_name = f"{name}.vm.so"
    with tvm.transform.PassContext(opt_level=3):
        vmc = relay.vm.compile(model, target=target_c, target_host=target_h, params=params)
        if RUN_ON_HOST:
            vmc.mod.export_library(lib_name)
        else:
            vmc.mod.export_library(lib_name, ndk.create_shared)
        text_file = open(f"{name}.vm.json", "w")
        text_file.write(vmc.bytecode)
        text_file.close()

    return vmc, lib_name


def compile_model_for_ge(name, model, params, target):
    lib_name = f"{name}.graph.so"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(model, target=target, params=params)
        if RUN_ON_HOST:
            lib.export_library(lib_name)
        else:
            lib.export_library(lib_name, ndk.create_shared)
    return lib, lib_name


def run_model_with_vm(input_dict, lib_name):
    if RUN_ON_HOST:
        remote = rpc.LocalSession()
        remote.upload(lib_name)
    else:
        rpc_tracker_host = os.getenv("TVM_TRACKER_HOST", "0.0.0.0")
        rpc_tracker_port = int(os.getenv("TVM_TRACKER_PORT", "9190"))
        tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
        remote = tracker.request(
            rpc_key, priority=0
        )
        remote.upload(lib_name)
    rlib = remote.load_module(lib_name)
    dev = remote.cl()
    vm = VirtualMachine(rlib, dev, "naive")
    data = {}
    for k, v in input_dict.items():
        data[k] = tvm.nd.array(v, dev)
    vm.set_input("main", **data)
    #return vm.run()
    vm.invoke_stateful("main")
    outputs = vm.get_outputs()
    return outputs


def run_model_with_ge(input_dict, lib_name):
    if RUN_ON_HOST:
        remote = rpc.LocalSession()
        remote.upload(lib_name)
    else:
        rpc_tracker_host = os.getenv("TVM_TRACKER_HOST", "0.0.0.0")
        rpc_tracker_port = int(os.getenv("TVM_TRACKER_PORT", "9190"))
        tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
        remote = tracker.request(
            rpc_key, priority=0
        )
        remote.upload(lib_name)
    rlib = remote.load_module(lib_name)
    dev = remote.cl()
    module = graph_executor.GraphModule(rlib["default"](dev))
    module.set_input(**input_dict)
    module.run()
    return module.get_output(0)


if __name__ == '__main__':
    target = Target(target_c, host=target_h)
    #name = "resnet_vm_model"
    #model, params, shape_dict = get_resnet_model()
    #name = "my_conv2d"
    #model, params, shape_dict = get_model()
    #name = "resnet_ssd_vm_model"
    #model, params, shape_dict = get_resnet_ssd_model()
    #name = "my_ssd_vm_model"
    #model, params, shape_dict = get_ssd_model()
    #name = "onnx_yolo_vm_model"
    #model, params, shape_dict = get_onnx_yolo_model()
    name = "onnx_faster_rcnn_model"
    model, params, shape_dict = get_onnx_faster_rcnn_model()
    input_dict = {}
    for k, v in shape_dict.items():
        img = np.random.rand(*v).astype("float32")
        input_dict[k] = img
    if USE_VM:
        _, lib_name = compile_model_for_vm(name, model, params, target)
        tvm_res = run_model_with_vm(input_dict, lib_name)
    else:
        _, lib_name = compile_model_for_ge(name, model, params, target)
        tvm_res = run_model_with_ge(input_dict, lib_name)
