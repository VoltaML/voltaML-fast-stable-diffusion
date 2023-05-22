import { d as defineComponent, e as openBlock, f as createElementBlock, n as createBaseVNode, u as useState, a as useSettings, E as ref, bi as onMounted, o as onUnmounted, s as serverUrl, x as createBlock, w as withCtx, g as createVNode, h as unref, N as NGi, F as NButton, G as NIcon, m as createTextVNode, z as NGrid, by as NAlert, y as createCommentVNode, i as NCard } from "./index.js";
const _hoisted_1$1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M133 440a35.37 35.37 0 0 1-17.5-4.67c-12-6.8-19.46-20-19.46-34.33V111c0-14.37 7.46-27.53 19.46-34.33a35.13 35.13 0 0 1 35.77.45l247.85 148.36a36 36 0 0 1 0 61l-247.89 148.4A35.5 35.5 0 0 1 133 440z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$1 = [_hoisted_2$1];
const Play = defineComponent({
  name: "Play",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$1, _hoisted_3$1);
  }
});
const _hoisted_1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M402 76.94C362.61 37.63 310.78 16 256 16h-.37A208 208 0 0 0 48 224v100.67A79.62 79.62 0 0 0 98.29 399l23.71 9.42a15.92 15.92 0 0 1 9.75 11.72l10 50.13A32.09 32.09 0 0 0 173.12 496H184a8 8 0 0 0 8-8v-39.55c0-8.61 6.62-16 15.23-16.43A16 16 0 0 1 224 448v40a8 8 0 0 0 8 8a8 8 0 0 0 8-8v-39.55c0-8.61 6.62-16 15.23-16.43A16 16 0 0 1 272 448v40a8 8 0 0 0 8 8a8 8 0 0 0 8-8v-39.55c0-8.61 6.62-16 15.23-16.43A16 16 0 0 1 320 448v40a8 8 0 0 0 8 8h10.88a32.09 32.09 0 0 0 31.38-25.72l10-50.14a16 16 0 0 1 9.74-11.72l23.71-9.42A79.62 79.62 0 0 0 464 324.67v-99c0-56-22-108.81-62-148.73zM171.66 335.88a56 56 0 1 1 52.22-52.22a56 56 0 0 1-52.22 52.22zM281 397.25a16.37 16.37 0 0 1-9.3 2.75h-31.4a16.37 16.37 0 0 1-9.28-2.75a16 16 0 0 1-6.6-16.9l15.91-47.6C243 326 247.25 321 254 320.13c8.26-1 14 2.87 17.61 12.22l16 48a16 16 0 0 1-6.61 16.9zm66.68-61.37a56 56 0 1 1 52.22-52.22a56 56 0 0 1-52.24 52.22z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3 = [_hoisted_2];
const Skull = defineComponent({
  name: "Skull",
  render: function render2(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1, _hoisted_3);
  }
});
const _2img = "";
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "GenerateSection",
  props: {
    generate: {
      type: Function,
      required: true
    },
    doNotDisableGenerate: {
      type: Boolean,
      default: false
    }
  },
  setup(__props) {
    const props = __props;
    const global = useState();
    const conf = useSettings();
    const generateButton = ref(null);
    onMounted(() => {
      window.addEventListener("keydown", handleKeyDown);
    });
    onUnmounted(() => {
      window.removeEventListener("keydown", handleKeyDown);
    });
    function handleKeyDown(e) {
      if (e.key === "Enter" && e.ctrlKey) {
        e.preventDefault();
        if (global.state.generating) {
          return;
        }
        const fn = props.generate;
        fn(e);
      }
      if (e.key === "Escape") {
        e.preventDefault();
        interrupt();
      }
    }
    function interrupt() {
      fetch(`${serverUrl}/api/general/interrupt`, {
        method: "POST"
      }).then((res) => {
        if (res.status === 200) {
          global.state.generating = false;
        }
      });
    }
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), { style: { "margin-bottom": "12px" } }, {
        default: withCtx(() => {
          var _a, _b;
          return [
            createVNode(unref(NGrid), {
              cols: "2",
              "x-gap": "24"
            }, {
              default: withCtx(() => [
                createVNode(unref(NGi), null, {
                  default: withCtx(() => {
                    var _a2, _b2;
                    return [
                      createVNode(unref(NButton), {
                        type: "success",
                        ref_key: "generateButton",
                        ref: generateButton,
                        onClick: props.generate,
                        disabled: !props.doNotDisableGenerate && (unref(global).state.generating || ((_a2 = unref(conf).data.settings.model) == null ? void 0 : _a2.name) === "" || ((_b2 = unref(conf).data.settings.model) == null ? void 0 : _b2.name) === void 0),
                        loading: unref(global).state.generating,
                        style: { "width": "100%" },
                        ghost: ""
                      }, {
                        icon: withCtx(() => [
                          createVNode(unref(NIcon), null, {
                            default: withCtx(() => [
                              createVNode(unref(Play))
                            ]),
                            _: 1
                          })
                        ]),
                        default: withCtx(() => [
                          createTextVNode("Generate ")
                        ]),
                        _: 1
                      }, 8, ["onClick", "disabled", "loading"])
                    ];
                  }),
                  _: 1
                }),
                createVNode(unref(NGi), null, {
                  default: withCtx(() => [
                    createVNode(unref(NButton), {
                      type: "error",
                      onClick: interrupt,
                      style: { "width": "100%" },
                      ghost: "",
                      disabled: !unref(global).state.generating
                    }, {
                      icon: withCtx(() => [
                        createVNode(unref(NIcon), null, {
                          default: withCtx(() => [
                            createVNode(unref(Skull))
                          ]),
                          _: 1
                        })
                      ]),
                      default: withCtx(() => [
                        createTextVNode("Interrupt ")
                      ]),
                      _: 1
                    }, 8, ["disabled"])
                  ]),
                  _: 1
                })
              ]),
              _: 1
            }),
            ((_a = unref(conf).data.settings.model) == null ? void 0 : _a.name) === "" || ((_b = unref(conf).data.settings.model) == null ? void 0 : _b.name) === void 0 ? (openBlock(), createBlock(unref(NAlert), {
              key: 0,
              style: { "margin-top": "12px" },
              type: "warning",
              title: "No model loaded",
              bordered: false
            })) : createCommentVNode("", true)
          ];
        }),
        _: 1
      });
    };
  }
});
export {
  _sfc_main as _
};
