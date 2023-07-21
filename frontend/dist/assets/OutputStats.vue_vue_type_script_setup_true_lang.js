import { N as NDescriptionsItem, a as NDescriptions } from "./DescriptionsItem.js";
import { d as defineComponent, e as openBlock, v as createBlock, w as withCtx, g as createVNode, h as unref, m as createTextVNode, t as toDisplayString, i as NCard, x as createCommentVNode } from "./index.js";
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "OutputStats",
  props: {
    genData: {
      type: Object,
      required: true
    }
  },
  setup(__props) {
    return (_ctx, _cache) => {
      return __props.genData.time_taken || __props.genData.seed ? (openBlock(), createBlock(unref(NCard), {
        key: 0,
        title: "Stats"
      }, {
        default: withCtx(() => [
          createVNode(unref(NDescriptions), null, {
            default: withCtx(() => [
              createVNode(unref(NDescriptionsItem), { label: "Total Time" }, {
                default: withCtx(() => [
                  createTextVNode(toDisplayString(__props.genData.time_taken) + "s ", 1)
                ]),
                _: 1
              }),
              createVNode(unref(NDescriptionsItem), { label: "Seed" }, {
                default: withCtx(() => [
                  createTextVNode(toDisplayString(__props.genData.seed), 1)
                ]),
                _: 1
              })
            ]),
            _: 1
          })
        ]),
        _: 1
      })) : createCommentVNode("", true);
    };
  }
});
export {
  _sfc_main as _
};
