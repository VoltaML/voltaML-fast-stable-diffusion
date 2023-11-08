import { _ as _export_sfc, d as defineComponent, u as useState, e as openBlock, p as createBlock, w as withCtx, h as unref, g as createVNode, C as NTabPane, D as NTabs } from "./index.js";
const _sfc_main$2 = {};
function _sfc_render$1(_ctx, _cache) {
  return "Autofill manager";
}
const AutofillManager = /* @__PURE__ */ _export_sfc(_sfc_main$2, [["render", _sfc_render$1]]);
const _sfc_main$1 = {};
function _sfc_render(_ctx, _cache) {
  return "Dependency manager";
}
const DependencyManager = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["render", _sfc_render]]);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ExtraView",
  setup(__props) {
    const state = useState();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NTabs), {
        type: "segment",
        value: unref(state).state.extra.tab,
        "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(state).state.extra.tab = $event)
      }, {
        default: withCtx(() => [
          createVNode(unref(NTabPane), {
            tab: "Dependencies",
            name: "dependencies"
          }, {
            default: withCtx(() => [
              createVNode(unref(DependencyManager))
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), {
            tab: "Autofill",
            name: "autofill"
          }, {
            default: withCtx(() => [
              createVNode(unref(AutofillManager))
            ]),
            _: 1
          })
        ]),
        _: 1
      }, 8, ["value"]);
    };
  }
});
export {
  _sfc_main as default
};
