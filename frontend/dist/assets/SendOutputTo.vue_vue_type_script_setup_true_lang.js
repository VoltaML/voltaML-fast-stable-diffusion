import { d as defineComponent, o as openBlock, a as createElementBlock, i as createBaseVNode, bG as useRouter, u as useSettings, v as useState, r as ref, b9 as reactive, c as computed, b as createVNode, w as withCtx, e as unref, N as NCard, Q as NScrollbar, F as Fragment, f as renderList, h as createTextVNode, t as toDisplayString, be as NDivider, g as NButton, m as NModal, k as createBlock, A as NGi, H as NGrid, G as createCommentVNode } from "./index.js";
import { a as NSwitch } from "./Switch.js";
const _hoisted_1$3 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$3 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "stroke-width": "32",
    d: "M368 368L144 144"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$3 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "stroke-width": "32",
    d: "M368 144L144 368"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$3 = [_hoisted_2$3, _hoisted_3$3];
const CloseOutline = defineComponent({
  name: "CloseOutline",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$3, _hoisted_4$3);
  }
});
const _hoisted_1$2 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$2 = /* @__PURE__ */ createBaseVNode(
  "rect",
  {
    x: "128",
    y: "128",
    width: "336",
    height: "336",
    rx: "57",
    ry: "57",
    fill: "none",
    stroke: "currentColor",
    "stroke-linejoin": "round",
    "stroke-width": "32"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$2 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M383.5 128l.5-24a56.16 56.16 0 0 0-56-56H112a64.19 64.19 0 0 0-64 64v216a56.16 56.16 0 0 0 56 56h24",
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "stroke-width": "32"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$2 = [_hoisted_2$2, _hoisted_3$2];
const CopyOutline = defineComponent({
  name: "CopyOutline",
  render: function render2(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$2, _hoisted_4$2);
  }
});
const _hoisted_1$1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M376 160H272v153.37l52.69-52.68a16 16 0 0 1 22.62 22.62l-80 80a16 16 0 0 1-22.62 0l-80-80a16 16 0 0 1 22.62-22.62L240 313.37V160H136a56.06 56.06 0 0 0-56 56v208a56.06 56.06 0 0 0 56 56h240a56.06 56.06 0 0 0 56-56V216a56.06 56.06 0 0 0-56-56z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$1 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M272 48a16 16 0 0 0-32 0v112h32z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$1 = [_hoisted_2$1, _hoisted_3$1];
const Download = defineComponent({
  name: "Download",
  render: function render3(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$1, _hoisted_4$1);
  }
});
const _hoisted_1 = { style: { "margin": "0 24px" } };
const _hoisted_2 = { style: { "display": "flex", "flex-direction": "row", "justify-content": "space-between" } };
const _hoisted_3 = { style: { "display": "flex", "flex-direction": "row", "justify-content": "flex-end" } };
const _hoisted_4 = { key: 0 };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "SendOutputTo",
  props: {
    output: {
      type: String,
      required: true
    },
    card: {
      type: Boolean,
      default: true
    },
    data: {
      type: Object,
      required: false,
      default: () => ({})
    }
  },
  setup(__props) {
    const props = __props;
    const router = useRouter();
    const conf = useSettings();
    const state = useState();
    const showModal = ref(false);
    const maybeTarget = ref(null);
    const targets = {
      img2img: "img2img",
      controlnet: "img2img",
      inpainting: "img2img",
      upscale: "extra",
      tagger: "tagger"
    };
    function handleClick(target) {
      if (props.data) {
        maybeTarget.value = target;
        showModal.value = true;
      } else {
        toTarget(target);
      }
    }
    function modalCopyClick() {
      showModal.value = false;
      if (maybeTarget.value) {
        const tmp = maybeTarget.value;
        maybeTarget.value = null;
        toTarget(tmp);
      }
    }
    const valuesToCopy = reactive(
      Object.fromEntries(Object.keys(props.data).map((key) => [key, false]))
    );
    const valuesToCopyFiltered = computed(() => {
      return Object.fromEntries(
        Object.keys(valuesToCopy).filter((key) => {
          if (maybeTarget.value) {
            return Object.keys(conf.data.settings[maybeTarget.value]).includes(
              key
            );
          }
        }).map((key) => [key, valuesToCopy[key]])
      );
    });
    async function toTarget(target) {
      const targetPage = targets[target];
      conf.data.settings[target].image = props.output;
      state.state[targetPage].tab = target;
      Object.keys(props.data).forEach((key) => {
        if (valuesToCopy[key]) {
          if (Object.keys(conf.data.settings[target]).includes(key)) {
            conf.data.settings[target][key] = props.data[key];
          }
        }
      });
      await router.push("/" + targetPage);
    }
    function capitalizeAndReplace(target) {
      return target.split("_").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
    }
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(unref(NModal), { show: showModal.value }, {
          default: withCtx(() => [
            createVNode(unref(NCard), {
              style: { "max-width": "700px" },
              title: "Copy additional properties"
            }, {
              default: withCtx(() => [
                createVNode(unref(NScrollbar), { style: { "max-height": "70vh", "margin-bottom": "8px" } }, {
                  default: withCtx(() => [
                    createBaseVNode("div", _hoisted_1, [
                      (openBlock(true), createElementBlock(Fragment, null, renderList(Object.keys(valuesToCopyFiltered.value), (item) => {
                        return openBlock(), createElementBlock("div", null, [
                          createBaseVNode("div", _hoisted_2, [
                            createTextVNode(toDisplayString(capitalizeAndReplace(item)) + " ", 1),
                            createVNode(unref(NSwitch), {
                              value: valuesToCopy[item],
                              "onUpdate:value": (v) => valuesToCopy[item] = v
                            }, null, 8, ["value", "onUpdate:value"])
                          ]),
                          createVNode(unref(NDivider), { style: { "margin": "12px 0" } })
                        ]);
                      }), 256))
                    ])
                  ]),
                  _: 1
                }),
                createBaseVNode("div", _hoisted_3, [
                  createVNode(unref(NButton), {
                    type: "default",
                    onClick: _cache[0] || (_cache[0] = () => showModal.value = false),
                    style: { "margin-right": "4px", "flex-grow": "1" }
                  }, {
                    icon: withCtx(() => [
                      createVNode(unref(CloseOutline))
                    ]),
                    default: withCtx(() => [
                      createTextVNode(" Cancel ")
                    ]),
                    _: 1
                  }),
                  createVNode(unref(NButton), {
                    type: "primary",
                    onClick: modalCopyClick,
                    style: { "flex-grow": "1" }
                  }, {
                    icon: withCtx(() => [
                      createVNode(unref(CopyOutline))
                    ]),
                    default: withCtx(() => [
                      createTextVNode(" Copy ")
                    ]),
                    _: 1
                  })
                ])
              ]),
              _: 1
            })
          ]),
          _: 1
        }, 8, ["show"]),
        __props.output ? (openBlock(), createElementBlock("div", _hoisted_4, [
          __props.output && __props.card ? (openBlock(), createBlock(unref(NCard), {
            key: 0,
            style: { "margin": "12px 0" },
            title: "Send To"
          }, {
            default: withCtx(() => [
              createVNode(unref(NGrid), {
                cols: "4",
                "x-gap": "4",
                "y-gap": "4"
              }, {
                default: withCtx(() => [
                  (openBlock(true), createElementBlock(Fragment, null, renderList(Object.keys(targets), (target) => {
                    return openBlock(), createBlock(unref(NGi), null, {
                      default: withCtx(() => [
                        createVNode(unref(NButton), {
                          type: "default",
                          onClick: () => handleClick(target),
                          style: { "width": "100%" },
                          ghost: ""
                        }, {
                          default: withCtx(() => [
                            createTextVNode(toDisplayString(capitalizeAndReplace(target)), 1)
                          ]),
                          _: 2
                        }, 1032, ["onClick"])
                      ]),
                      _: 2
                    }, 1024);
                  }), 256))
                ]),
                _: 1
              })
            ]),
            _: 1
          })) : (openBlock(), createBlock(unref(NGrid), {
            key: 1,
            cols: "3",
            "x-gap": "4",
            "y-gap": "4"
          }, {
            default: withCtx(() => [
              (openBlock(true), createElementBlock(Fragment, null, renderList(Object.keys(targets), (target) => {
                return openBlock(), createBlock(unref(NGi), null, {
                  default: withCtx(() => [
                    createVNode(unref(NButton), {
                      type: "default",
                      onClick: () => handleClick(target),
                      style: { "width": "100%" },
                      ghost: ""
                    }, {
                      default: withCtx(() => [
                        createTextVNode("-> " + toDisplayString(capitalizeAndReplace(target)), 1)
                      ]),
                      _: 2
                    }, 1032, ["onClick"])
                  ]),
                  _: 2
                }, 1024);
              }), 256))
            ]),
            _: 1
          }))
        ])) : createCommentVNode("", true)
      ], 64);
    };
  }
});
export {
  Download as D,
  _sfc_main as _
};
