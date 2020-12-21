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
#include <numeric>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#include "Accelerate/Accelerate.h"

#define USE_OLD_BNNS_API 1

template<typename T1, typename T2>
bool one_of(T1 arg1, T2 arg2) {
  return arg1 == arg2;
}

template<typename T1, typename T2, typename ...T>
bool one_of(T1 arg1, T2 arg2, T... args) {
  return arg1 == arg2 || one_of(arg1, args...);
}

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

/** C++ wrapper on top of original BNNS C api */
namespace BNNS {
  using Dim = size_t;
  using Shape = std::vector<Dim>;
  using Dtype = BNNSDataType;

  void* default_alloc(size_t size) {
    // TODO: make it aligned with cash line
    return malloc(size);
  }

  void default_free(void* ptr) {
    free(ptr);
  }

  class Tensor {
   public:
    Tensor(Shape shape, Dtype dtype, void* hdl):real_shape(shape) {
#if USE_OLD_BNNS_API
      ICHECK(one_of(shape.size(), 3, 4));
      const auto dim_shift = (shape.size() == 4) ? 1 : 0;
      const size_t N = dim_shift ? shape[0] : 1;
      const size_t C = shape[dim_shift + 0];
      const size_t H = shape[dim_shift + 1];
      const size_t W = shape[dim_shift + 2];
      bnns_desc = {
          W,     /* width */
          H,     /* height */
          C,     /* channels */
          W,     /* row_stride */
          H*W,   /* image_stride */
          dtype, /* data_type */
          1.,    /* data_scale */
          0.     /* data_bias */
      };
#else
      ICHECK(shape.size() < BNNS_MAX_TENSOR_DIMENSION);
      /*
       *    BNNSNDArrayFlags flags;
            BNNSDataLayout layout;

            size_t size[BNNS_MAX_TENSOR_DIMENSION];
            size_t stride[BNNS_MAX_TENSOR_DIMENSION];

            void * _Nullable data;
            BNNSDataType data_type;

            void * _Nullable table_data;
            BNNSDataType table_data_type;

            float data_scale;
            float data_bias;
       */

      BNNSNDArrayFlags default_nd_array_flag = BNNSNDArrayFlagBackpropSet;
      bnns_nd_desc = {
        default_nd_array_flag,
        BNNSDataLayout4DLastMajor,       // TODO [apeskov]: should support all ND layouts
        {},      // shape
        {},      // strides
        hdl,     // data handler
        dtype,   // data type
        nullptr, // table of values (clustering case)
        BNNSDataTypeFloat32,
        1.f,
        0.f
      };

      std::copy(shape.begin(), shape.end(), std::begin(bnns_nd_desc.size));
#endif
      if (hdl) {
        data_handler = hdl;
        is_external_data = true;
      } else {
        const size_t elem_size = (dtype & 0xFFFF) / 4;
        const size_t elem_count = std::accumulate(real_shape.begin(), real_shape.end(),
            1, std::multiplies<int>());

        data_handler = default_alloc(elem_count * elem_size);
        is_external_data = false;
      }
    }

    ~Tensor() {
      if (data_handler && !is_external_data) {
        default_free(data_handler);
        data_handler = nullptr;
      }
    }

    Dtype get_data_type() const { return bnns_desc.data_type; }
    size_t get_elem_size() const { return bnns_desc.data_type & 0xffff; }

    void* get_data_hdl() { return data_handler; }
    const void* get_data_hdl() const { return data_handler; };
    void set_data_hdl(void *hdl) {
      if (data_handler && !is_external_data) {
        default_free(data_handler);
        data_handler = nullptr;
      }

      data_handler = hdl;
      is_external_data = true;
    }

    const BNNSImageStackDescriptor& get_desc() const { return bnns_desc; };

    BNNSLayerData get_bnns_layer_data() const {
      return {
          data_handler,         /* data */
          bnns_desc.data_type,  /* data_type */
          bnns_desc.data_scale, /* data_scale */
          bnns_desc.data_bias,  /* data_bias */
          nullptr               /* data_table */
      };
    }

   private:
    Shape real_shape;
    void* data_handler;
    bool is_external_data = false;
    BNNSImageStackDescriptor bnns_desc;
    BNNSNDArrayDescriptor bnns_nd_desc;
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
      auto res = BNNSFilterApply(bnns_filter, src1.get_data_hdl(), dst1.get_data_hdl());
      ICHECK_EQ(res, 0) << "BNNS runtime. Primitive was not executed properly";
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
    SetupConstants(consts);
    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
  }

  void Run() override {
    // Wrap external handler into BNNS tensor representation
    auto bind_ext_hdl_to_tensor = [this] (uint32_t eid) {
      std::cout << "Blabla !!!!" << std::endl;
      const auto &ext_dlt = *data_entry_[eid];
      auto &bnns_tensor = *entry_out_mem_[eid];
      bnns_tensor.set_data_hdl(ext_dlt.data);
    };

    // Bind all input/output external data object into internal abstractions
    for (const auto &eid : input_var_eid_) {
      bind_ext_hdl_to_tensor(eid);
    }
    for (const auto &out_entity : outputs_) {
      bind_ext_hdl_to_tensor(EntryID(out_entity));
    }

    // Invoke primitives in topological order
    for (int i = 0; i < primitives_.size(); ++i) {
      auto src = entry_out_mem_.at(prim_args_.at(i).first);
      auto dst = entry_out_mem_.at(prim_args_.at(i).second);
      primitives_.at(i).execute(*src, *dst);
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
        } else if ("bnns.conv2d_relu" == op_name) {
          Conv2d(nid, true, false);
        } else if ("bnns.conv2d_bias_relu" == op_name) {
          Conv2d(nid, true, true);
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

  // Bind a JSON graph node entry to a BNNS tensor.
  std::shared_ptr<BNNS::Tensor> BindBNNSTensor(const JSONGraphNodeEntry& entry, void *hdl = nullptr) {
    auto eid = EntryID(entry);
    if (entry_out_mem_.count(eid) == 0) {
      auto data_node = nodes_[entry.id_];
      auto dlshape = data_node.GetOpShape()[entry.index_];
      auto dltype = data_node.GetOpDataType()[entry.index_];

      entry_out_mem_[eid] = std::make_shared<BNNS::Tensor>(
          BNNS::Shape{dlshape.begin(), dlshape.end()},
          convertToBNNS(dltype), hdl);
    }
    return entry_out_mem_[eid];
  }

  void Conv2d(const size_t& nid, const bool has_relu = false, const bool has_bias = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    auto dst_entry = JSONGraphNodeEntry(nid, 0);

    auto dl_input_shape = nodes_[src_entry.id_].GetOpShape()[src_entry.index_];
    auto dl_weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    BNNS::Shape input_shape {dl_input_shape.begin(), dl_input_shape.end()};
    BNNS::Shape weight_shape {dl_weight_shape.begin(), dl_weight_shape.end()};
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
        PH_L = std::stoi(str_padding[0]),       // height padding: left
        PH_R = std::stoi(str_padding[2]),       // height padding: right
        PW_L = std::stoi(str_padding[1]),       // width padding: left
        PW_R = std::stoi(str_padding[3]),       // width padding: right
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

    auto weight_ext_data_hdl = data_entry_[EntryID(weight_entry)]->data;

    // Memory descriptions.
    auto src_md = BindBNNSTensor(src_entry);
    auto weights_md = BindBNNSTensor(weight_entry, weight_ext_data_hdl);
    auto dst_md = BindBNNSTensor(dst_entry);
    // TODO [apeskov]: check correctness of tensor shapes

    BNNSLayerData bias_data = {};

    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      auto bias_ext_data_hdl = data_entry_[EntryID(bias_entry)]->data;
      bias_data = BindBNNSTensor(bias_entry, weight_ext_data_hdl)->get_bnns_layer_data();
    } else {
      bias_data = {
          nullptr,  /* data */
          BNNSDataTypeFloat32, /* data_type */
          1.,       /* data_scale */
          0.,       /* data_bias */
          nullptr   /* data_table */
      };
    }

    BNNSActivation activation = { has_relu ?
        BNNSActivationFunctionRectifiedLinear : BNNSActivationFunctionIdentity };

    BNNSConvolutionLayerParameters conv_param = {
        SW ,  /* x_stride */
        SH ,  /* y_stride */
        PW_L, /* x_padding */
        PH_L, /* y_padding */
        KW,   /* k_width */
        KH,   /* k_height */
        IC,   /* in_channels */
        OC,   /* out_channels */
        weights_md->get_bnns_layer_data(), /* weights */
        bias_data,  /* bias */
        activation, /* activation */
    };

    auto filter = BNNSFilterCreateConvolutionLayer(&src_md->get_desc(), &dst_md->get_desc(),
              &conv_param, &common_filter_param);
    ICHECK(filter) << "BNNS primitive was not created. Unsupported attributes configuration";

    primitives_.emplace_back(filter);
    prim_args_.push_back({EntryID(src_entry), EntryID(dst_entry)});
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

  BNNS::Dtype convertToBNNS(const DLDataType &dl_dtype) {
    if (dl_dtype.code == DLDataTypeCode::kDLFloat) {
      if (dl_dtype.bits == 32) return BNNSDataTypeFloat32;
      if (dl_dtype.bits == 16) return BNNSDataTypeFloat16;
    }
    if (dl_dtype.code == DLDataTypeCode::kDLInt) {
      if (dl_dtype.bits == 32) return BNNSDataTypeInt32;
      if (dl_dtype.bits == 16) return BNNSDataTypeInt16;
      if (dl_dtype.bits == 8) return BNNSDataTypeInt8;
    }
    if (dl_dtype.code == DLDataTypeCode::kDLUInt) {
      if (dl_dtype.bits == 32) return BNNSDataTypeUInt32;
      if (dl_dtype.bits == 16) return BNNSDataTypeUInt16;
      if (dl_dtype.bits == 8) return BNNSDataTypeUInt8;
    }
    LOG(FATAL) << "Unsupported data type for BNNS runtime";
    return BNNS::Dtype(0);
  }

  BNNSFilterParameters common_filter_param;

  std::vector<BNNS::Primitive> primitives_;
  std::vector<std::pair<uint32_t, uint32_t>> prim_args_;

  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::shared_ptr<BNNS::Tensor>> entry_out_mem_;
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
