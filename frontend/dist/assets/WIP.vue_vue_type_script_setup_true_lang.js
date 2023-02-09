import { d as defineComponent, o as openBlock, j as createElementBlock, a as createBaseVNode, c as createBlock, w as withCtx, b as createVNode, u as unref, E as NIcon } from "./index.js";
import { N as NResult } from "./Result.js";
const _hoisted_1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M393.87 190a32.1 32.1 0 0 1-45.25 0l-26.57-26.57a32.09 32.09 0 0 1 0-45.26L382.19 58a1 1 0 0 0-.3-1.64c-38.82-16.64-89.15-8.16-121.11 23.57c-30.58 30.35-32.32 76-21.12 115.84a31.93 31.93 0 0 1-9.06 32.08L64 380a48.17 48.17 0 1 0 68 68l153.86-167a31.93 31.93 0 0 1 31.6-9.13c39.54 10.59 84.54 8.6 114.72-21.19c32.49-32 39.5-88.56 23.75-120.93a1 1 0 0 0-1.6-.26z",
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-miterlimit": "10",
    "stroke-width": "32"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3 = /* @__PURE__ */ createBaseVNode(
  "circle",
  {
    cx: "96",
    cy: "416",
    r: "16",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4 = [_hoisted_2, _hoisted_3];
const BuildOutline = defineComponent({
  name: "BuildOutline",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1, _hoisted_4);
  }
});
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "WIP",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NResult), {
        title: "Work in progress",
        description: "This page is still under development.",
        style: { "height": "70vh", "display": "flex", "align-items": "center", "justify-content": "center" }
      }, {
        icon: withCtx(() => [
          createVNode(unref(NIcon), { size: "250" }, {
            default: withCtx(() => [
              createVNode(unref(BuildOutline))
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
export {
  _sfc_main as _
};
