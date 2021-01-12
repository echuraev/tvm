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
 * \brief Implementation of BNNS codegen APIs.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <numeric>
#include <sstream>

#include "../../utils.h"

#include "../../../../runtime/contrib/json/json_node.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class BNNSJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  BNNSJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    Expr expr = GetRef<Expr>(cn);
    std::string name;
    const CallNode* call = cn;
    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      auto comp = fn->GetAttr<String>(attr::kComposite);
      ICHECK(comp.defined()) << "BNNS JSON runtime only supports composite functions.";
      name = comp.value();

      auto body = fn->body.as<CallNode>();
      if (name == "bnns.conv2d_bias_relu") {
        auto add_op_type = IsOp(body->args[0].as<CallNode>(), "add") ? "add" : "nn.bias_add";
        call = GetRootCall(body, 2, {"nn.conv2d", add_op_type, "nn.relu"});
      } else if (name == "bnns.conv2d_bias") {
        auto add_op_type = IsOp(body, "add") ? "add" : "nn.bias_add";
        call = GetRootCall(body, 1, {"nn.conv2d", add_op_type});
      } else if (name == "bnns.conv2d_relu") {
        call = GetRootCall(body, 1, {"nn.conv2d", "nn.relu"});
        ICHECK(call->op.as<OpNode>()) << "Not op node";
      } else if (name == "bnns.dense_bias") {
        call = GetRootCall(fn->body.as<CallNode>(), 1, {"nn.dense", "add"});
      } else if (name == "bnns.dense_bias_gelu") {
        call = FindExpRootCall(fn->body.as<CallNode>(), 10, "nn.dense");
      } else {
        LOG(FATAL) << "Unrecognized BNNS pattern: " << name;
      }
    } else {
      LOG(FATAL) << "BNNS JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }

    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    SetCallNodeAttribute(node, call);
    return AddNode(node, GetRef<Expr>(cn));
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module BNNSCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  BNNSJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();

  const auto* pf = runtime::Registry::Get("runtime.BNNSJSONRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.bnns").set_body_typed(BNNSCompiler);

/*!
 * \brief A helper to expand the params by adding the ones used by BNNS runtime
 * for a given expression. Same as default ConstantUpdater but skip essential
 * BNNS composed function nodes.
 */
struct BNNSConstantUpdater : public ConstantUpdater {
 public:
  BNNSConstantUpdater(const std::string& symbol,
                      std::unordered_map<std::string, runtime::NDArray>* params,
                      std::vector<std::string> &skip_mask)
      : ConstantUpdater(symbol, params), skip_mask_(skip_mask) {}

  /**!
   * Like an original implementation but avoid visiting of body nodes
   * for BNNS specific composite primitives.
   */
  void VisitExpr_(const FunctionNode* op) final override {
    this->VisitSpan(op->span);
    for (auto param : op->params) {
      this->VisitExpr(param);
    }

    if (!isBNNSSpecificComposite(op)) {
      this->VisitExpr(op->body);
    }
  }

 private:
  bool isBNNSSpecificComposite(const FunctionNode* op) {
    auto comp = op->GetAttr<String>(attr::kComposite);
    if (!comp)
      return false;

    auto comp_name = comp.value();

    bool is_match = false;
    for (const auto &mask : skip_mask_) {
      if (std::string(comp_name).substr(0, mask.size()) == mask) {
        is_match = true;
        break;
      }
    }
    return is_match;
  }

  std::vector<std::string> skip_mask_;
};

Map<String, runtime::NDArray> BNNSConstantUpdaterFunc(Expr expr, std::string symbol) {
  std::vector<std::string> filter_by_mask = {"bnns."};

  // Visit all suitable constant nodes
  std::unordered_map<std::string, runtime::NDArray> res;
  BNNSConstantUpdater const_updater(symbol, &res, filter_by_mask);
  const_updater(expr);

  // Convert to tvm::Map
  Map<String, runtime::NDArray> ret;
  for (const auto& kvp: res)
    ret.Set(kvp.first, kvp.second);
  return ret;
}

TVM_REGISTER_GLOBAL("relay.ext.bnns.constant_updater")
    .set_body_typed(BNNSConstantUpdaterFunc);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
