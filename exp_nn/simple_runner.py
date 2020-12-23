import tvm
from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime

import numpy as np
import argparse

ap = argparse.ArgumentParser(description='Some simple script to convert ONNX model.')
ap.add_argument('-m',  '--model', dest='model', required=True,
                help="input module file (tgz or dylib)")
ap.add_argument('-i',  '--input', dest='input', required=False,
                help="input npz file")
ap.add_argument('-d', '--debug', dest='debug', default=False, action='store_true',
                help="use debug runtime")
args = ap.parse_args()


def get_inputs():
    if args.input:
        file_npz = np.load(args.input)
        in_blob = [file_npz[name] for name in file_npz.files]
        assert(len(in_blob) == 1)
    else:
        in_blob = [np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")]

    return in_blob


ctx = tvm.cpu(0)
loaded_lib = tvm.runtime.load_module(args.model)

if args.debug:
    gmod = loaded_lib["debug_create"]("default", ctx)
    set_input = gmod["set_input"]
    get_output = gmod["get_output"]
    run = gmod["run"]
    # run = gmod["run_individual"]

    input_data = get_inputs()[0]
    set_input(0, tvm.nd.array(input_data))
    run()
    out_data = get_output(0).asnumpy()
else:
    module = graph_runtime.GraphModule(loaded_lib["default"](ctx))
    input_data = get_inputs()[0]
    module.set_input(0, input_data)
    module.run()
    out_data = module.get_output(0).asnumpy()

cat_label = 281
everyone_is_cat = True
for b in range(out_data.shape[0]):
    top1 = np.argmax(out_data[b])
    print("MB({0}) top-1 #{1}: {2:.2f} ".format(b, top1, out_data[b, top1]))
    everyone_is_cat &= (top1 == cat_label)

if everyone_is_cat:
    print("Correct scoring!!! +++ ")
else:
    print("Bad scoring...     --- ")




