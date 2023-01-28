import { N as NResult } from "./Result.js";
import { d as defineComponent, q as createBlock, e as unref, o as openBlock } from "./index.js";
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ExtraView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NResult), {
        status: "info",
        title: "Extra",
        description: "Work in progress"
      });
    };
  }
});
export {
  _sfc_main as default
};
