import { d as defineComponent, e as openBlock, f as createElementBlock, n as createBaseVNode } from "./index.js";
const _hoisted_1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2 = /* @__PURE__ */ createBaseVNode(
  "rect",
  {
    x: "32",
    y: "48",
    width: "448",
    height: "80",
    rx: "32",
    ry: "32",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M74.45 160a8 8 0 0 0-8 8.83l26.31 252.56a1.5 1.5 0 0 0 0 .22A48 48 0 0 0 140.45 464h231.09a48 48 0 0 0 47.67-42.39v-.21l26.27-252.57a8 8 0 0 0-8-8.83zm248.86 180.69a16 16 0 1 1-22.63 22.62L256 318.63l-44.69 44.68a16 16 0 0 1-22.63-22.62L233.37 296l-44.69-44.69a16 16 0 0 1 22.63-22.62L256 273.37l44.68-44.68a16 16 0 0 1 22.63 22.62L278.62 296z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4 = [_hoisted_2, _hoisted_3];
const TrashBin = defineComponent({
  name: "TrashBin",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1, _hoisted_4);
  }
});
export {
  TrashBin as T
};
