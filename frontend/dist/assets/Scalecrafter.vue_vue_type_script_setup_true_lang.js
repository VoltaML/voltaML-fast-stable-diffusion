import { d as defineComponent, u as useSettings, i as computed, o as openBlock, g as createElementBlock, b as createBaseVNode, e as createVNode, f as unref, w as withCtx, k as createTextVNode, bE as NAlert, N as NCard, m as NSelect, h as createCommentVNode, F as Fragment, c as createBlock, C as NTabPane, D as NTabs, l as NTooltip, j as NSpace } from "./index.js";
import { N as NSwitch } from "./Switch.js";
import { N as NInputNumber } from "./InputNumber.js";
import { N as NSlider } from "./Slider.js";
import { b as _sfc_main$3 } from "./Upscale.vue_vue_type_script_setup_true_lang.js";
const _hoisted_1$1 = { class: "flex-container" };
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_3$1 = { key: 0 };
const _hoisted_4$1 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "diffusers", -1);
const _hoisted_5$1 = { class: "flex-container space-between" };
const _hoisted_6$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Depth", -1);
const _hoisted_7$1 = { class: "flex-container" };
const _hoisted_8$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Stop at", -1);
const _hoisted_9$1 = { class: "flex-container space-between" };
const _hoisted_10 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Depth", -1);
const _hoisted_11 = { class: "flex-container" };
const _hoisted_12 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Stop at", -1);
const _hoisted_13 = { class: "flex-container" };
const _hoisted_14 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Scale", -1);
const _hoisted_15 = { class: "flex-container" };
const _hoisted_16 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Latent scaler", -1);
const _hoisted_17 = { class: "flex-container" };
const _hoisted_18 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Early out", -1);
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "DeepShrink",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const latentUpscalerOptions = [
      { label: "Nearest", value: "nearest" },
      { label: "Nearest exact", value: "nearest-exact" },
      { label: "Area", value: "area" },
      { label: "Bilinear", value: "bilinear" },
      { label: "Bicubic", value: "bicubic" },
      { label: "Bislerp", value: "bislerp" }
    ];
    const target = computed(() => {
      if (props.target === "settings") {
        return settings.data.settings;
      }
      return settings.defaultSettings;
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createBaseVNode("div", _hoisted_1$1, [
          _hoisted_2$1,
          createVNode(unref(NSwitch), {
            value: target.value[props.tab].deepshrink.enabled,
            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => target.value[props.tab].deepshrink.enabled = $event)
          }, null, 8, ["value"])
        ]),
        target.value[props.tab].deepshrink.enabled ? (openBlock(), createElementBlock("div", _hoisted_3$1, [
          createVNode(unref(NAlert), { type: "warning" }, {
            default: withCtx(() => [
              createTextVNode(" Only works on "),
              _hoisted_4$1,
              createTextVNode(" samplers ")
            ]),
            _: 1
          }),
          createVNode(unref(NCard), {
            bordered: false,
            title: "First layer"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_5$1, [
                _hoisted_6$1,
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].deepshrink.depth_1,
                  "onUpdate:value": _cache[1] || (_cache[1] = ($event) => target.value[props.tab].deepshrink.depth_1 = $event),
                  max: 4,
                  min: 1,
                  step: 1
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_7$1, [
                _hoisted_8$1,
                createVNode(unref(NSlider), {
                  value: target.value[props.tab].deepshrink.stop_at_1,
                  "onUpdate:value": _cache[2] || (_cache[2] = ($event) => target.value[props.tab].deepshrink.stop_at_1 = $event),
                  min: 0.05,
                  max: 1,
                  step: 0.05,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].deepshrink.stop_at_1,
                  "onUpdate:value": _cache[3] || (_cache[3] = ($event) => target.value[props.tab].deepshrink.stop_at_1 = $event),
                  max: 1,
                  min: 0.05,
                  step: 0.05
                }, null, 8, ["value"])
              ])
            ]),
            _: 1
          }),
          createVNode(unref(NCard), {
            bordered: false,
            title: "Second layer"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_9$1, [
                _hoisted_10,
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].deepshrink.depth_2,
                  "onUpdate:value": _cache[4] || (_cache[4] = ($event) => target.value[props.tab].deepshrink.depth_2 = $event),
                  max: 4,
                  min: 1,
                  step: 1
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_11, [
                _hoisted_12,
                createVNode(unref(NSlider), {
                  value: target.value[props.tab].deepshrink.stop_at_2,
                  "onUpdate:value": _cache[5] || (_cache[5] = ($event) => target.value[props.tab].deepshrink.stop_at_2 = $event),
                  min: 0.05,
                  max: 1,
                  step: 0.05
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].deepshrink.stop_at_2,
                  "onUpdate:value": _cache[6] || (_cache[6] = ($event) => target.value[props.tab].deepshrink.stop_at_2 = $event),
                  max: 1,
                  min: 0.05,
                  step: 0.05
                }, null, 8, ["value"])
              ])
            ]),
            _: 1
          }),
          createVNode(unref(NCard), {
            bordered: false,
            title: "Scale"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_13, [
                _hoisted_14,
                createVNode(unref(NSlider), {
                  value: target.value[props.tab].deepshrink.base_scale,
                  "onUpdate:value": _cache[7] || (_cache[7] = ($event) => target.value[props.tab].deepshrink.base_scale = $event),
                  min: 0.05,
                  max: 1,
                  step: 0.05
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].deepshrink.base_scale,
                  "onUpdate:value": _cache[8] || (_cache[8] = ($event) => target.value[props.tab].deepshrink.base_scale = $event),
                  max: 1,
                  min: 0.05,
                  step: 0.05
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_15, [
                _hoisted_16,
                createVNode(unref(NSelect), {
                  value: target.value[props.tab].deepshrink.scaler,
                  "onUpdate:value": _cache[9] || (_cache[9] = ($event) => target.value[props.tab].deepshrink.scaler = $event),
                  filterable: "",
                  options: latentUpscalerOptions
                }, null, 8, ["value"])
              ])
            ]),
            _: 1
          }),
          createVNode(unref(NCard), {
            bordered: false,
            title: "Other"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_17, [
                _hoisted_18,
                createVNode(unref(NSwitch), {
                  value: target.value[props.tab].deepshrink.early_out,
                  "onUpdate:value": _cache[10] || (_cache[10] = ($event) => target.value[props.tab].deepshrink.early_out = $event)
                }, null, 8, ["value"])
              ])
            ]),
            _: 1
          })
        ])) : createCommentVNode("", true)
      ], 64);
    };
  }
});
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "HighResFixTabs",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), {
        title: "High Resolution Fix",
        class: "generate-extra-card"
      }, {
        default: withCtx(() => [
          createVNode(unref(NTabs), {
            animated: "",
            type: "segment"
          }, {
            default: withCtx(() => [
              createVNode(unref(NTabPane), {
                tab: "Image to Image",
                name: "highresfix"
              }, {
                default: withCtx(() => [
                  createVNode(unref(_sfc_main$3), {
                    tab: props.tab,
                    target: props.target
                  }, null, 8, ["tab", "target"])
                ]),
                _: 1
              }),
              createVNode(unref(NTabPane), {
                tab: "Scalecrafter",
                name: "scalecrafter"
              }, {
                default: withCtx(() => [
                  createVNode(unref(_sfc_main), {
                    tab: props.tab,
                    target: props.target
                  }, null, 8, ["tab", "target"])
                ]),
                _: 1
              }),
              createVNode(unref(NTabPane), {
                tab: "DeepShrink",
                name: "deepshrink"
              }, {
                default: withCtx(() => [
                  createVNode(unref(_sfc_main$2), {
                    tab: props.tab,
                    target: props.target
                  }, null, 8, ["tab", "target"])
                ]),
                _: 1
              })
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
const _hoisted_1 = { class: "flex-container" };
const _hoisted_2 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "Automatic", -1);
const _hoisted_4 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "Karras", -1);
const _hoisted_5 = { class: "flex-container" };
const _hoisted_6 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Disperse", -1);
const _hoisted_7 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, " However, this comes at the cost of increased vram usage, generally in the range of 3-4x. ", -1);
const _hoisted_8 = { class: "flex-container" };
const _hoisted_9 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Unsafe resolutions", -1);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "Scalecrafter",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const target = computed(() => {
      if (props.target === "settings") {
        return settings.data.settings;
      }
      return settings.defaultSettings;
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createBaseVNode("div", _hoisted_1, [
          _hoisted_2,
          createVNode(unref(NSwitch), {
            value: target.value[props.tab].scalecrafter.enabled,
            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => target.value[props.tab].scalecrafter.enabled = $event)
          }, null, 8, ["value"])
        ]),
        target.value[props.tab].scalecrafter.enabled ? (openBlock(), createBlock(unref(NSpace), {
          key: 0,
          vertical: "",
          class: "left-container"
        }, {
          default: withCtx(() => [
            createVNode(unref(NAlert), { type: "warning" }, {
              default: withCtx(() => [
                createTextVNode(" Only works with "),
                _hoisted_3,
                createTextVNode(" and "),
                _hoisted_4,
                createTextVNode(" sigmas ")
              ]),
              _: 1
            }),
            createBaseVNode("div", _hoisted_5, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_6
                ]),
                default: withCtx(() => [
                  createTextVNode(" May generate more unique images. "),
                  _hoisted_7
                ]),
                _: 1
              }),
              createVNode(unref(NSwitch), {
                value: target.value[props.tab].scalecrafter.disperse,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => target.value[props.tab].scalecrafter.disperse = $event)
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_8, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_9
                ]),
                default: withCtx(() => [
                  createTextVNode(" Allow generating with unique resolutions that don't have configs ready for them, or clamp them (really, force them) to the closest resolution. ")
                ]),
                _: 1
              }),
              createVNode(unref(NSwitch), {
                value: target.value[props.tab].scalecrafter.unsafe_resolutions,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => target.value[props.tab].scalecrafter.unsafe_resolutions = $event)
              }, null, 8, ["value"])
            ])
          ]),
          _: 1
        })) : createCommentVNode("", true)
      ], 64);
    };
  }
});
export {
  _sfc_main$1 as _
};
