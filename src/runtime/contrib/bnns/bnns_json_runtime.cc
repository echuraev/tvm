/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file
 * \brief A simple JSON runtime for BNNS.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#include "Accelerate/Accelerate.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

/** C++ wrapper on top of original BNNS C api */
namespace BNNS {
  using Dim = int64_t;
  using Shape = std::vector<int64_t>;
  using Dtype = BNNSDataType;

  class Tensor {
   public:
    void* get_data_hdl() { return data_handler; }
    const void* get_data_hdl() const { return data_handler; };

   private:
    Shape real_shape;
    Dtype data_type;
    void* data_handler;
    BNNSImageStackDescriptor bnns_desc;
  };

  class Primitive {
   public:
    Primitive(BNNSFilter f) : bnns_filter(f) {}
    ~Primitive() {
      if (bnns_filter) {
        BNNSFilterDestroy(bnns_filter);
        bnns_filter = nullptr;
      }
    }

    void execute(const Tensor &src1, Tensor &dst1) {
      BNNSFilterApply(bnns_filter, src1.get_data_hdl(), dst1.get_data_hdl());
    }

   private:
    BNNSFilter bnns_filter = nullptr;
  };
}

class BNNSJSONRuntime : public JSONRuntimeBase {

 public:
  BNNSJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "bnns_json"; }

  void Init(const Array<NDArray>& consts) override {
    BuildEngine();

//    ICHECK_EQ(consts.size(), const_idx_.size())
//        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
  }

  void Run() override {
    // Fill in the input buffers.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      const DLTensor &dlt = *data_entry_[eid];

      size_t buffer_size = GetDataSize(dlt);
      write_to_bnns_memory(dlt.data, buffer_size, entry_out_mem_[eid]);
    }

    // Invoke the engine through intepreting the stream.
    for (int i = 0; i < primitives_.size(); ++i) {
      auto src = tensors_.at(prim_args_.at(i).first);
      auto dst = tensors_.at(prim_args_.at(i).second);
      primitives_.at(i).execute(src, dst);
    }

    // Read output buffers.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      const DLTensor &dlt = *data_entry_[eid];

      size_t buffer_size = GetDataSize(dlt);
      read_from_dnnl_memory(dlt.data, buffer_size, entry_out_mem_[eid]);
    }
  }

 private:
  // Build up the engine based on the input graph.
  void BuildEngine() {
    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
//        } else if ("dnnl.conv2d_relu" == op_name) {
//          Conv2d(nid, true, false);
//        } else if ("dnnl.conv2d_bias_relu" == op_name) {
//          Conv2d(nid, true, true);
//        } else if ("nn.dense" == op_name) {
//          Dense(nid);
//        } else if ("nn.batch_norm" == op_name) {
//          BatchNorm(nid);
//        } else if ("nn.relu" == op_name) {
//          Relu(nid);
//        } else if ("add" == op_name) {
//          Add(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

//  // Bind a JSON graph node entry to a DNNL memory.
//  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory::desc mem_desc,
//                              size_t offset = 0) {
//    auto eid = EntryID(entry);
//    if (entry_out_mem_.count(eid) == 0) {
//      return BindDNNLMemory(entry, dnnl::memory(mem_desc, engine_), offset);
//    }
//    return entry_out_mem_[eid].first;
//  }
//
//  // Bind a JSON graph node entry to a given DNNL memory.
//  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory mem,
//                              size_t offset = 0) {
//    auto eid = EntryID(entry);
//    // Since the DNNL memory has been created before calling this function, we assume the entry
//    // has not yet been bound to the other DNNL memory; otherwise it may have memory leak.
//    ICHECK_EQ(entry_out_mem_.count(eid), 0);
//
//    // TODO(@comanic): Support other data types (i.e., int8).
//    auto data_node = nodes_[entry.id_];
//    auto dltype = data_node.GetOpDataType()[entry.index_];
//    ICHECK_EQ(dltype.bits, 32);
//
//    entry_out_mem_[eid] = {mem, offset};
//    return entry_out_mem_[eid].first;
//  }
//
  void Conv2d(const size_t& nid, const bool has_relu = false, const bool has_bias = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    BNNS::Shape input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    BNNS::Shape weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    BNNS::Dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    BNNS::Dim N = input_shape[0],               // batch size
        IC = input_shape[1],                    // input channels
        IH = input_shape[2],                    // input height
        IW = input_shape[2],                    // input width
        OC = weight_shape[0],                   // output channels
        KH = weight_shape[2],                   // weight height
        KW = weight_shape[3],                   // weight width
        PH_L = std::stoi(str_padding[1]),       // height padding: left
        PH_R = std::stoi(str_padding[3]),       // height padding: right
        PW_L = std::stoi(str_padding[0]),       // width padding: left
        PW_R = std::stoi(str_padding[2]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[0]),         // weight-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    BNNS::Shape src_dims = {N, IC, IH, IW};
    BNNS::Shape weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {   // TODO [apeskov]: Group param is not supported for ios < 14
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    BNNS::Shape bias_dims = {OC};
    BNNS::Shape dst_dims = {N, OC, OH, OW};
    BNNS::Shape strides_dims = {SH, SW};
    BNNS::Shape padding_dims_l = {PH_L, PW_L};
    BNNS::Shape padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
//    auto conv_src_md = dnnl::memory::desc(src_dims, dt::f32, tag::any);
//    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::f32, tag::any);
//    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::any);
//    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::nchw);

    // Covn2d description.
//    auto filer = BNNSConvolutionLayerParameters

  }

  // Read from BNNS memory and write to the handle.
  inline void read_from_dnnl_memory(void* handle, size_t size, BNNS::Tensor& tensor) {
    uint8_t* src = static_cast<uint8_t*>(tensor.get_data_hdl());
    std::copy(src, src + size, static_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to BNNS tensor.
  inline void write_to_bnns_memory(void* handle, size_t size, BNNS::Tensor& tensor) {
    uint8_t* dst = static_cast<uint8_t*>(tensor.get_data_hdl());
    std::copy(reinterpret_cast<uint8_t*>(handle), reinterpret_cast<uint8_t*>(handle) + size,
              dst);
  }
//
//  // Generate DNNL memory description and infer the data layout by the given shape.
//  inline dnnl::memory::desc GenDNNLMemDescByShape(const dnnl::memory::dims& shape, dt dtype) {
//    dnnl::memory::desc data_md;
//    switch (shape.size()) {
//      case 1:
//        data_md = dnnl::memory::desc({shape, dtype, tag::a});
//        break;
//      case 2:
//        data_md = dnnl::memory::desc({shape, dtype, tag::ab});
//        break;
//      case 3:
//        data_md = dnnl::memory::desc({shape, dtype, tag::abc});
//        break;
//      case 4:
//        data_md = dnnl::memory::desc({shape, dtype, tag::abcd});
//        break;
//      case 5:
//        data_md = dnnl::memory::desc({shape, dtype, tag::abcde});
//        break;
//      default:
//        LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
//        break;
//    }
//    return data_md;
//  }
//
  BNNSFilterParameters execution_param;

  std::vector<BNNS::Tensor> tensors_;
  std::vector<BNNS::Primitive> primitives_;
  std::vector<std::pair<uint32_t, uint32_t>> prim_args_;

//  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, BNNS::Tensor> entry_out_mem_;
};

runtime::Module BNNSJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<BNNSJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.BNNSJSONRuntimeCreate").set_body_typed(BNNSJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_bnns_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<BNNSJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
