import { j as defineComponent, r as ref, z as createElementBlock, F as createVNode, G as withCtx, H as unref, a1 as Fragment, y as openBlock, P as createTextVNode, a2 as NButton, K as NCard, b9 as renderList, A as createBaseVNode, M as NInput, Q as NSelect, a5 as toDisplayString } from "./index.js";
import { a as NTabs, N as NTabPane } from "./Tabs.js";
import { N as NModal } from "./Modal.js";
const _hoisted_1 = /* @__PURE__ */ createBaseVNode("div", null, "Model 2", -1);
const _hoisted_2 = /* @__PURE__ */ createBaseVNode("div", null, "Model 2", -1);
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("div", null, "Model 2", -1);
const _hoisted_4 = { style: { "display": "inline-flex", "width": "100%", "margin-top": "12px" } };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TestView",
  setup(__props) {
    const models = ["Model 1", "Model 2", "Model 3", "Model 4"];
    const selectedModel = ref("");
    const show = ref(true);
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(unref(NButton), {
          onClick: _cache[0] || (_cache[0] = ($event) => show.value = true)
        }, {
          default: withCtx(() => [
            createTextVNode("Show Modal")
          ]),
          _: 1
        }),
        createVNode(unref(NModal), {
          show: show.value,
          "onUpdate:show": _cache[1] || (_cache[1] = ($event) => show.value = $event),
          closable: "",
          "mask-closable": ""
        }, {
          default: withCtx(() => [
            createVNode(unref(NCard), {
              title: "Models",
              style: { "width": "70vw" }
            }, {
              default: withCtx(() => [
                createVNode(unref(NTabs), {
                  type: "segment",
                  style: { "height": "70vh" }
                }, {
                  default: withCtx(() => [
                    createVNode(unref(NTabPane), {
                      name: "PyTorch",
                      style: { "height": "100%" }
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NCard), {
                          title: "Models",
                          style: { "height": "100%" }
                        }, {
                          default: withCtx(() => [
                            (openBlock(), createElementBlock(Fragment, null, renderList(models, (model) => {
                              return createBaseVNode("div", {
                                style: { "display": "inline-flex", "width": "100%", "align-items": "center", "justify-content": "space-between", "border-bottom": "1px solid rgb(66, 66, 71)" },
                                key: model
                              }, [
                                createBaseVNode("p", null, toDisplayString(model), 1),
                                createVNode(unref(NButton), {
                                  type: "success",
                                  ghost: "",
                                  onClick: ($event) => selectedModel.value = model
                                }, {
                                  default: withCtx(() => [
                                    createTextVNode("Select")
                                  ]),
                                  _: 2
                                }, 1032, ["onClick"])
                              ]);
                            }), 64))
                          ]),
                          _: 1
                        })
                      ]),
                      _: 1
                    }),
                    createVNode(unref(NTabPane), { name: "AITemplate" }, {
                      default: withCtx(() => [
                        _hoisted_1
                      ]),
                      _: 1
                    }),
                    createVNode(unref(NTabPane), { name: "TensorRT" }, {
                      default: withCtx(() => [
                        _hoisted_2
                      ]),
                      _: 1
                    }),
                    createVNode(unref(NTabPane), { name: "Extra" }, {
                      default: withCtx(() => [
                        _hoisted_3
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                }),
                createBaseVNode("div", _hoisted_4, [
                  createVNode(unref(NInput), {
                    placeholder: "Model Name",
                    disabled: "",
                    style: { "justify-self": "start", "margin-right": "12px" },
                    value: selectedModel.value
                  }, null, 8, ["value"]),
                  createVNode(unref(NSelect), {
                    placeholder: "GPU",
                    style: { "margin-right": "12px" }
                  }),
                  createVNode(unref(NButton), {
                    style: { "justify-self": "end" },
                    type: "success",
                    ghost: ""
                  }, {
                    default: withCtx(() => [
                      createTextVNode("Load")
                    ]),
                    _: 1
                  })
                ])
              ]),
              _: 1
            })
          ]),
          _: 1
        }, 8, ["show"])
      ], 64);
    };
  }
});
export {
  _sfc_main as default
};
