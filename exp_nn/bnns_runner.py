import tvm
from tvm import rpc, relay
from tvm.contrib import utils, xcode, coreml_runtime, graph_runtime
from tvm.contrib.debugger import debug_runtime
from tvm.contrib.download import download_testdata
from tvm.relay.op.contrib.bnns import partition_for_bnns

import os
import onnx
import numpy as np

MODEL_URL="https://github.com/onnx/models/raw/master/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx"
INPUT_SHAPE = [1, 256]
proxy_host = os.environ["TVM_IOS_RPC_PROXY_HOST"]
destination = os.environ["TVM_IOS_RPC_DESTINATION"]
proxy_port = 9090
key = "iphone"
arch = "arm64"
sdk = "iphoneos"
target_iphone = "llvm -mtriple=%s-apple-darwin" % arch
run_on_host = False
if run_on_host:
    target = "llvm -mtriple=x86_64-apple-darwin20.1.0"
else:
    target = "llvm -mtriple=%s-apple-darwin" % arch


def get_name_from_url(url):
    return url[url.rfind('/') + 1:].strip()


def get_input_name(model):
    inputs = [node.name for node in model.graph.input]
    initializer = [node.name for node in model.graph.initializer]

    inputs = list(set(inputs) - set(initializer))
    return inputs


def generate_bert_input(shape):
    return np.random.randint(256, size=(shape[0], shape[1])).astype("int64")


def get_model(model_url):
    model_name = get_name_from_url(model_url)
    model_path = download_testdata(model_url, model_name, module="models")
    onnx_model = onnx.load(model_path)
    input_dict = {}
    input_name = get_input_name(onnx_model)
    for input in input_name:
        input_dict[input] = INPUT_SHAPE
    mod, params = relay.frontend.from_onnx(onnx_model, input_dict, freeze_params=True)
    return mod, params, input_dict


def process(model_url):
    temp = utils.tempdir()
    model, params, input_dict = get_model(model_url)
    gen_inputs = {}
    print("Got model")

    def generate_inputs():
        if len(gen_inputs) > 0:
            return gen_inputs
        for name, shape in input_dict.items():
            gen_inputs[name] = generate_bert_input(shape)
        return gen_inputs

    def set_inputs(m, ctx, pregen_inputs=None):
        for name, shape in input_dict.items():
            if pregen_inputs is None:
                m.set_input(name, tvm.nd.array(generate_bert_input(shape), ctx))
            else:
                m.set_input(name, tvm.nd.array(pregen_inputs[name], ctx))

    def run(mod, target, with_bnns):
        with tvm.transform.PassContext(opt_level=3):
            if with_bnns:
                mod = partition_for_bnns(mod)
            graph_module = relay.build(mod, target=target, target_host=target, params=params)
        print("before rpc connection")
        if run_on_host:
            lib_name = "deploy.tar"
            path_dso = temp.relpath(lib_name)
            graph_module.export_library(path_dso)
            remote = rpc.LocalSession()
            remote.upload(path_dso)
        else:
            lib_name = "deploy.dylib"
            path_dso = temp.relpath(lib_name)
            graph_module.export_library(path_dso, xcode.create_dylib, arch=arch, sdk=sdk)
            xcode.codesign(path_dso)
            # Start RPC test server that contains the compiled library.
            xcode.popen_test_rpc(proxy_host, proxy_port, key, destination=destination, libs=[path_dso])
            remote = rpc.connect(proxy_host, proxy_port, key=key)
        print("rpc connected")

        func = remote.load_module(lib_name)
        ctx = remote.cpu(0)
        print("Create module")
        m = graph_runtime.GraphModule(func["default"](ctx))
        set_inputs(m, ctx, generate_inputs())
        print("Run module")
        m.run()
        print("After Run module")
        tvm_output = m.get_output(0)
        top1 = np.argmax(tvm_output.asnumpy()[0])
        print("TVM prediction top-1:", top1)

        # evaluate
        ftimer = m.module.time_evaluator("run", ctx, number=3, repeat=10)
        prof_res = np.array(ftimer().results) * 1000
        return top1, np.mean(prof_res)

    res, t = run(model, target, False)
    res_bnns, t_bnns = run(model, target, True)
    if res != res_bnns:
        raise Exception("Results are not the same (llvm: {}, bnns: {})".format(res, res_bnns))
    print("=" * 100)
    print("llvm time: {:.3f} ms".format(t))
    print("bnns time: {:.3f} ms".format(t_bnns))
    print("=" * 100)


if __name__ == '__main__':
    process(MODEL_URL)

