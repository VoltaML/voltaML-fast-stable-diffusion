var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
import { N as NDescriptionsItem, a as NDescriptions } from "./DescriptionsItem.js";
import { d as defineComponent, o as openBlock, j as createElementBlock, f as createBaseVNode, e as createBlock, w as withCtx, g as createVNode, h as unref, k as createTextVNode, C as toDisplayString, n as NCard, m as createCommentVNode, u as useSettings, l as NTooltip, i as NSelect, F as Fragment, a as useState, c as computed, G as spaceRegex, B as NIcon, H as promptHandleKeyUp, I as promptHandleKeyDown, J as NInput, _ as _export_sfc, K as watch, z as ref, t as serverUrl } from "./index.js";
import { N as NSlider } from "./Slider.js";
import { N as NInputNumber } from "./InputNumber.js";
import { N as NSwitch } from "./Switch.js";
import { N as NForm, a as NFormItem } from "./SamplerPicker.vue_vue_type_script_setup_true_lang.js";
const _hoisted_1$4 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$4 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M262.29 192.31a64 64 0 1 0 57.4 57.4a64.13 64.13 0 0 0-57.4-57.4zM416.39 256a154.34 154.34 0 0 1-1.53 20.79l45.21 35.46a10.81 10.81 0 0 1 2.45 13.75l-42.77 74a10.81 10.81 0 0 1-13.14 4.59l-44.9-18.08a16.11 16.11 0 0 0-15.17 1.75A164.48 164.48 0 0 1 325 400.8a15.94 15.94 0 0 0-8.82 12.14l-6.73 47.89a11.08 11.08 0 0 1-10.68 9.17h-85.54a11.11 11.11 0 0 1-10.69-8.87l-6.72-47.82a16.07 16.07 0 0 0-9-12.22a155.3 155.3 0 0 1-21.46-12.57a16 16 0 0 0-15.11-1.71l-44.89 18.07a10.81 10.81 0 0 1-13.14-4.58l-42.77-74a10.8 10.8 0 0 1 2.45-13.75l38.21-30a16.05 16.05 0 0 0 6-14.08c-.36-4.17-.58-8.33-.58-12.5s.21-8.27.58-12.35a16 16 0 0 0-6.07-13.94l-38.19-30A10.81 10.81 0 0 1 49.48 186l42.77-74a10.81 10.81 0 0 1 13.14-4.59l44.9 18.08a16.11 16.11 0 0 0 15.17-1.75A164.48 164.48 0 0 1 187 111.2a15.94 15.94 0 0 0 8.82-12.14l6.73-47.89A11.08 11.08 0 0 1 213.23 42h85.54a11.11 11.11 0 0 1 10.69 8.87l6.72 47.82a16.07 16.07 0 0 0 9 12.22a155.3 155.3 0 0 1 21.46 12.57a16 16 0 0 0 15.11 1.71l44.89-18.07a10.81 10.81 0 0 1 13.14 4.58l42.77 74a10.8 10.8 0 0 1-2.45 13.75l-38.21 30a16.05 16.05 0 0 0-6.05 14.08c.33 4.14.55 8.3.55 12.47z",
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
const _hoisted_3$3 = [_hoisted_2$4];
const SettingsOutline = defineComponent({
  name: "SettingsOutline",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$4, _hoisted_3$3);
  }
});
const _sfc_main$5 = /* @__PURE__ */ defineComponent({
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
const _hoisted_1$3 = {
  key: 0,
  class: "flex-container"
};
const _hoisted_2$3 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1);
const _hoisted_3$2 = {
  key: 1,
  class: "flex-container"
};
const _hoisted_4$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1);
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "BatchSizeInput",
  props: {
    batchSizeObject: {
      type: Object,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    return (_ctx, _cache) => {
      return unref(settings).data.settings.aitDim.batch_size ? (openBlock(), createElementBlock("div", _hoisted_1$3, [
        createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
          trigger: withCtx(() => [
            _hoisted_2$3
          ]),
          default: withCtx(() => [
            createTextVNode(" Number of images to generate in paralel. ")
          ]),
          _: 1
        }),
        createVNode(unref(NSlider), {
          value: props.batchSizeObject.batch_size,
          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => props.batchSizeObject.batch_size = $event),
          min: unref(settings).data.settings.aitDim.batch_size[0],
          max: unref(settings).data.settings.aitDim.batch_size[1],
          style: { "margin-right": "12px" }
        }, null, 8, ["value", "min", "max"]),
        createVNode(unref(NInputNumber), {
          value: props.batchSizeObject.batch_size,
          "onUpdate:value": _cache[1] || (_cache[1] = ($event) => props.batchSizeObject.batch_size = $event),
          size: "small",
          min: unref(settings).data.settings.aitDim.batch_size[0],
          max: unref(settings).data.settings.aitDim.batch_size[1],
          style: { "min-width": "96px", "width": "96px" }
        }, null, 8, ["value", "min", "max"])
      ])) : (openBlock(), createElementBlock("div", _hoisted_3$2, [
        createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
          trigger: withCtx(() => [
            _hoisted_4$1
          ]),
          default: withCtx(() => [
            createTextVNode(" Number of images to generate in paralel. ")
          ]),
          _: 1
        }),
        createVNode(unref(NSlider), {
          value: props.batchSizeObject.batch_size,
          "onUpdate:value": _cache[2] || (_cache[2] = ($event) => props.batchSizeObject.batch_size = $event),
          min: 1,
          max: 9,
          style: { "margin-right": "12px" }
        }, null, 8, ["value"]),
        createVNode(unref(NInputNumber), {
          value: props.batchSizeObject.batch_size,
          "onUpdate:value": _cache[3] || (_cache[3] = ($event) => props.batchSizeObject.batch_size = $event),
          size: "small",
          style: { "min-width": "96px", "width": "96px" }
        }, null, 8, ["value"])
      ]));
    };
  }
});
const _hoisted_1$2 = {
  key: 0,
  class: "flex-container"
};
const _hoisted_2$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1);
const _hoisted_3$1 = {
  key: 1,
  class: "flex-container"
};
const _hoisted_4 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1);
const _hoisted_5 = {
  key: 2,
  class: "flex-container"
};
const _hoisted_6 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1);
const _hoisted_7 = {
  key: 3,
  class: "flex-container"
};
const _hoisted_8 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1);
const _hoisted_9 = { key: 4 };
const _hoisted_10 = { class: "flex-container" };
const _hoisted_11 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Enable Deepshrink", -1);
const _hoisted_12 = { key: 0 };
const _hoisted_13 = { class: "flex-container" };
const _hoisted_14 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "First layer", -1);
const _hoisted_15 = /* @__PURE__ */ createBaseVNode("div", null, null, -1);
const _hoisted_16 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Stop at", -1);
const _hoisted_17 = /* @__PURE__ */ createBaseVNode("div", null, null, -1);
const _hoisted_18 = { class: "flex-container" };
const _hoisted_19 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Second layer", -1);
const _hoisted_20 = /* @__PURE__ */ createBaseVNode("div", null, null, -1);
const _hoisted_21 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Stop at", -1);
const _hoisted_22 = /* @__PURE__ */ createBaseVNode("div", null, null, -1);
const _hoisted_23 = { class: "flex-container" };
const _hoisted_24 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Scale", -1);
const _hoisted_25 = /* @__PURE__ */ createBaseVNode("div", null, null, -1);
const _hoisted_26 = { class: "flex-container" };
const _hoisted_27 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Latent scaler", -1);
const _hoisted_28 = { class: "flex-container" };
const _hoisted_29 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Early out", -1);
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "DimensionsInput",
  props: {
    dimensionsObject: {
      type: Object,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const latentUpscalerOptions = [
      { label: "Nearest", value: "nearest" },
      { label: "Nearest exact", value: "nearest-exact" },
      { label: "Area", value: "area" },
      { label: "Bilinear", value: "bilinear" },
      { label: "Bicubic", value: "bicubic" },
      { label: "Bislerp", value: "bislerp" }
    ];
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        unref(settings).data.settings.aitDim.width ? (openBlock(), createElementBlock("div", _hoisted_1$2, [
          _hoisted_2$2,
          createVNode(unref(NSlider), {
            value: props.dimensionsObject.width,
            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => props.dimensionsObject.width = $event),
            min: unref(settings).data.settings.aitDim.width[0],
            max: unref(settings).data.settings.aitDim.width[1],
            step: 64,
            style: { "margin-right": "12px" }
          }, null, 8, ["value", "min", "max"]),
          createVNode(unref(NInputNumber), {
            value: props.dimensionsObject.width,
            "onUpdate:value": _cache[1] || (_cache[1] = ($event) => props.dimensionsObject.width = $event),
            size: "small",
            style: { "min-width": "96px", "width": "96px" },
            min: unref(settings).data.settings.aitDim.width[0],
            max: unref(settings).data.settings.aitDim.width[1],
            step: 64
          }, null, 8, ["value", "min", "max"])
        ])) : (openBlock(), createElementBlock("div", _hoisted_3$1, [
          _hoisted_4,
          createVNode(unref(NSlider), {
            value: props.dimensionsObject.width,
            "onUpdate:value": _cache[2] || (_cache[2] = ($event) => props.dimensionsObject.width = $event),
            min: 128,
            max: 2048,
            step: 1,
            style: { "margin-right": "12px" }
          }, null, 8, ["value"]),
          createVNode(unref(NInputNumber), {
            value: props.dimensionsObject.width,
            "onUpdate:value": _cache[3] || (_cache[3] = ($event) => props.dimensionsObject.width = $event),
            size: "small",
            style: { "min-width": "96px", "width": "96px" },
            step: 1
          }, null, 8, ["value"])
        ])),
        unref(settings).data.settings.aitDim.height ? (openBlock(), createElementBlock("div", _hoisted_5, [
          _hoisted_6,
          createVNode(unref(NSlider), {
            value: props.dimensionsObject.height,
            "onUpdate:value": _cache[4] || (_cache[4] = ($event) => props.dimensionsObject.height = $event),
            min: unref(settings).data.settings.aitDim.height[0],
            max: unref(settings).data.settings.aitDim.height[1],
            step: 64,
            style: { "margin-right": "12px" }
          }, null, 8, ["value", "min", "max"]),
          createVNode(unref(NInputNumber), {
            value: props.dimensionsObject.height,
            "onUpdate:value": _cache[5] || (_cache[5] = ($event) => props.dimensionsObject.height = $event),
            size: "small",
            style: { "min-width": "96px", "width": "96px" },
            min: unref(settings).data.settings.aitDim.height[0],
            max: unref(settings).data.settings.aitDim.height[1],
            step: 64
          }, null, 8, ["value", "min", "max"])
        ])) : (openBlock(), createElementBlock("div", _hoisted_7, [
          _hoisted_8,
          createVNode(unref(NSlider), {
            value: props.dimensionsObject.height,
            "onUpdate:value": _cache[6] || (_cache[6] = ($event) => props.dimensionsObject.height = $event),
            min: 128,
            max: 2048,
            step: 1,
            style: { "margin-right": "12px" }
          }, null, 8, ["value"]),
          createVNode(unref(NInputNumber), {
            value: props.dimensionsObject.height,
            "onUpdate:value": _cache[7] || (_cache[7] = ($event) => props.dimensionsObject.height = $event),
            size: "small",
            style: { "min-width": "96px", "width": "96px" },
            step: 1
          }, null, 8, ["value"])
        ])),
        props.dimensionsObject.width * props.dimensionsObject.height >= 768 * 768 || unref(settings).defaultSettings.flags.deepshrink.enabled ? (openBlock(), createElementBlock("div", _hoisted_9, [
          createBaseVNode("div", _hoisted_10, [
            _hoisted_11,
            createVNode(unref(NSwitch), {
              value: unref(settings).defaultSettings.flags.deepshrink.enabled,
              "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(settings).defaultSettings.flags.deepshrink.enabled = $event)
            }, null, 8, ["value"])
          ]),
          unref(settings).defaultSettings.flags.deepshrink.enabled ? (openBlock(), createElementBlock("div", _hoisted_12, [
            createBaseVNode("div", _hoisted_13, [
              _hoisted_14,
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.flags.deepshrink.depth_1,
                "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(settings).defaultSettings.flags.deepshrink.depth_1 = $event),
                max: 4,
                min: 1,
                step: 1
              }, null, 8, ["value"]),
              _hoisted_15,
              _hoisted_16,
              createVNode(unref(NSlider), {
                value: unref(settings).defaultSettings.flags.deepshrink.stop_at_1,
                "onUpdate:value": _cache[10] || (_cache[10] = ($event) => unref(settings).defaultSettings.flags.deepshrink.stop_at_1 = $event),
                min: 0.05,
                max: 1,
                step: 0.05
              }, null, 8, ["value"]),
              _hoisted_17,
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.flags.deepshrink.stop_at_1,
                "onUpdate:value": _cache[11] || (_cache[11] = ($event) => unref(settings).defaultSettings.flags.deepshrink.stop_at_1 = $event),
                max: 1,
                min: 0.05,
                step: 0.05
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_18, [
              _hoisted_19,
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.flags.deepshrink.depth_2,
                "onUpdate:value": _cache[12] || (_cache[12] = ($event) => unref(settings).defaultSettings.flags.deepshrink.depth_2 = $event),
                max: 4,
                min: 1,
                step: 1
              }, null, 8, ["value"]),
              _hoisted_20,
              _hoisted_21,
              createVNode(unref(NSlider), {
                value: unref(settings).defaultSettings.flags.deepshrink.stop_at_2,
                "onUpdate:value": _cache[13] || (_cache[13] = ($event) => unref(settings).defaultSettings.flags.deepshrink.stop_at_2 = $event),
                min: 0.05,
                max: 1,
                step: 0.05
              }, null, 8, ["value"]),
              _hoisted_22,
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.flags.deepshrink.stop_at_2,
                "onUpdate:value": _cache[14] || (_cache[14] = ($event) => unref(settings).defaultSettings.flags.deepshrink.stop_at_2 = $event),
                max: 1,
                min: 0.05,
                step: 0.05
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_23, [
              _hoisted_24,
              createVNode(unref(NSlider), {
                value: unref(settings).defaultSettings.flags.deepshrink.base_scale,
                "onUpdate:value": _cache[15] || (_cache[15] = ($event) => unref(settings).defaultSettings.flags.deepshrink.base_scale = $event),
                min: 0.05,
                max: 1,
                step: 0.05
              }, null, 8, ["value"]),
              _hoisted_25,
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.flags.deepshrink.base_scale,
                "onUpdate:value": _cache[16] || (_cache[16] = ($event) => unref(settings).defaultSettings.flags.deepshrink.base_scale = $event),
                max: 1,
                min: 0.05,
                step: 0.05
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_26, [
              _hoisted_27,
              createVNode(unref(NSelect), {
                value: unref(settings).defaultSettings.flags.deepshrink.scaler,
                "onUpdate:value": _cache[17] || (_cache[17] = ($event) => unref(settings).defaultSettings.flags.deepshrink.scaler = $event),
                filterable: "",
                options: latentUpscalerOptions
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_28, [
              _hoisted_29,
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.flags.deepshrink.early_out,
                "onUpdate:value": _cache[18] || (_cache[18] = ($event) => unref(settings).defaultSettings.flags.deepshrink.early_out = $event)
              }, null, 8, ["value"])
            ])
          ])) : createCommentVNode("", true)
        ])) : createCommentVNode("", true)
      ], 64);
    };
  }
});
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "Prompt",
  props: {
    tab: {
      type: String,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const state = useState();
    const promptCount = computed(() => {
      return settings.data.settings[props.tab].prompt.split(spaceRegex).length - 1;
    });
    const negativePromptCount = computed(() => {
      return settings.data.settings[props.tab].negative_prompt.split(spaceRegex).length - 1;
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", null, [
        createVNode(unref(NInput), {
          value: unref(settings).data.settings[props.tab].prompt,
          "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).data.settings[props.tab].prompt = $event),
          type: "textarea",
          placeholder: "Prompt",
          class: "prompt",
          "show-count": "",
          onKeyup: _cache[4] || (_cache[4] = ($event) => unref(promptHandleKeyUp)(
            $event,
            unref(settings).data.settings[props.tab],
            "prompt",
            unref(state)
          )),
          onKeydown: unref(promptHandleKeyDown)
        }, {
          suffix: withCtx(() => [
            createVNode(unref(NTooltip), null, {
              trigger: withCtx(() => [
                createVNode(unref(NIcon), { style: { "margin-top": "10px" } }, {
                  default: withCtx(() => [
                    createVNode(unref(SettingsOutline))
                  ]),
                  _: 1
                })
              ]),
              default: withCtx(() => [
                createVNode(unref(NForm), { "show-feedback": false }, {
                  default: withCtx(() => [
                    createVNode(unref(NFormItem), {
                      label: "Prompt-to-Prompt preprocessing",
                      class: "form-item"
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NSwitch), {
                          value: unref(settings).data.settings.api.prompt_to_prompt,
                          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).data.settings.api.prompt_to_prompt = $event)
                        }, null, 8, ["value"])
                      ]),
                      _: 1
                    }),
                    createVNode(unref(NFormItem), {
                      label: "Prompt-to-Prompt model",
                      class: "form-item"
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NSelect), {
                          filterable: "",
                          "consistent-menu-width": false,
                          options: [
                            {
                              value: "lllyasviel/Fooocus-Expansion",
                              label: "lllyasviel/Fooocus-Expansion"
                            },
                            {
                              value: "daspartho/prompt-extend",
                              label: "daspartho/prompt-extend"
                            },
                            {
                              value: "succinctly/text2image-prompt-generator",
                              label: "succinctly/text2image-prompt-generator"
                            },
                            {
                              value: "Gustavosta/MagicPrompt-Stable-Diffusion",
                              label: "Gustavosta/MagicPrompt-Stable-Diffusion"
                            },
                            {
                              value: "Ar4ikov/gpt2-medium-650k-stable-diffusion-prompt-generator",
                              label: "Ar4ikov/gpt2-medium-650k-stable-diffusion-prompt-generator"
                            }
                          ],
                          value: unref(settings).data.settings.api.prompt_to_prompt_model,
                          "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).data.settings.api.prompt_to_prompt_model = $event)
                        }, null, 8, ["value"])
                      ]),
                      _: 1
                    }),
                    createVNode(unref(NFormItem), {
                      label: "Prompt-to-Prompt device",
                      class: "form-item"
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NSelect), {
                          options: [
                            {
                              value: "gpu",
                              label: "On-Device"
                            },
                            {
                              value: "cpu",
                              label: "CPU"
                            }
                          ],
                          value: unref(settings).data.settings.api.prompt_to_prompt_device,
                          "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).data.settings.api.prompt_to_prompt_device = $event)
                        }, null, 8, ["value"])
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                })
              ]),
              _: 1
            })
          ]),
          count: withCtx(() => [
            createTextVNode(toDisplayString(promptCount.value), 1)
          ]),
          _: 1
        }, 8, ["value", "onKeydown"]),
        createVNode(unref(NInput), {
          value: unref(settings).data.settings[props.tab].negative_prompt,
          "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(settings).data.settings[props.tab].negative_prompt = $event),
          type: "textarea",
          placeholder: "Negative prompt",
          "show-count": "",
          onKeyup: _cache[6] || (_cache[6] = ($event) => unref(promptHandleKeyUp)(
            $event,
            unref(settings).data.settings[props.tab],
            "negative_prompt",
            unref(state)
          )),
          onKeydown: unref(promptHandleKeyDown)
        }, {
          count: withCtx(() => [
            createTextVNode(toDisplayString(negativePromptCount.value), 1)
          ]),
          _: 1
        }, 8, ["value", "onKeydown"])
      ]);
    };
  }
});
const Prompt = /* @__PURE__ */ _export_sfc(_sfc_main$2, [["__scopeId", "data-v-780680bc"]]);
const _hoisted_1$1 = { class: "flex-container" };
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "CFG Scale", -1);
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 3-15 for most images.", -1);
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "CFGScaleInput",
  props: {
    tab: {
      type: String,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const cfgMax = computed(() => {
      var scale = 30;
      return scale + Math.max(
        settings.defaultSettings.api.apply_unsharp_mask ? 15 : 0,
        settings.defaultSettings.api.cfg_rescale_threshold == "off" ? 0 : 30
      );
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
        createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
          trigger: withCtx(() => [
            _hoisted_2$1
          ]),
          default: withCtx(() => [
            createTextVNode(" Guidance scale indicates how close should the model stay to the prompt. Higher values might be exactly what you want, but generated images might have some artifacts. Lower values give the model more freedom, and therefore might produce more coherent/less-artifacty images, but wouldn't follow the prompt as closely. "),
            _hoisted_3
          ]),
          _: 1
        }),
        createVNode(unref(NSlider), {
          value: unref(settings).data.settings[props.tab].cfg_scale,
          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).data.settings[props.tab].cfg_scale = $event),
          min: 1,
          max: cfgMax.value,
          step: 0.5,
          style: { "margin-right": "12px" }
        }, null, 8, ["value", "max"]),
        createVNode(unref(NInputNumber), {
          value: unref(settings).data.settings[props.tab].cfg_scale,
          "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).data.settings[props.tab].cfg_scale = $event),
          size: "small",
          style: { "min-width": "96px", "width": "96px" },
          min: 1,
          max: cfgMax.value,
          step: 0.5
        }, null, 8, ["value", "max"])
      ]);
    };
  }
});
const _hoisted_1 = {
  key: 0,
  class: "flex-container"
};
const _hoisted_2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Self Attention Scale", -1);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "SAGInput",
  props: {
    tab: {
      type: String,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    return (_ctx, _cache) => {
      var _a;
      return ((_a = unref(settings).data.settings.model) == null ? void 0 : _a.backend) === "PyTorch" ? (openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
          trigger: withCtx(() => [
            _hoisted_2
          ]),
          default: withCtx(() => [
            createTextVNode(" If self attention is >0, SAG will guide the model and improve the quality of the image at the cost of speed. Higher values will follow the guidance more closely, which can lead to better, more sharp and detailed outputs. ")
          ]),
          _: 1
        }),
        createVNode(unref(NSlider), {
          value: unref(settings).data.settings[props.tab].self_attention_scale,
          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).data.settings[props.tab].self_attention_scale = $event),
          min: 0,
          max: 1,
          step: 0.05,
          style: { "margin-right": "12px" }
        }, null, 8, ["value"]),
        createVNode(unref(NInputNumber), {
          value: unref(settings).data.settings[props.tab].self_attention_scale,
          "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).data.settings[props.tab].self_attention_scale = $event),
          size: "small",
          style: { "min-width": "96px", "width": "96px" },
          step: 0.05
        }, null, 8, ["value"])
      ])) : createCommentVNode("", true);
    };
  }
});
class BurnerClock {
  constructor(observed_value, settings, callback, timerOverrride = 0, sendInterrupt = true) {
    __publicField(this, "isChanging", ref(false));
    __publicField(this, "timer", null);
    __publicField(this, "timeoutDuration");
    this.observed_value = observed_value;
    this.settings = settings;
    this.callback = callback;
    this.timerOverrride = timerOverrride;
    this.sendInterrupt = sendInterrupt;
    this.timeoutDuration = this.timerOverrride !== 0 ? this.timerOverrride : this.settings.data.settings.frontend.on_change_timer;
    watch(this.observed_value, () => {
      this.handleChange();
    });
  }
  handleChange() {
    if (!this.isChanging.value) {
      this.startTimer();
    } else {
      this.resetTimer();
    }
  }
  startTimer() {
    if (this.timeoutDuration > 0) {
      this.isChanging.value = true;
      this.timer = setTimeout(() => {
        if (this.sendInterrupt) {
          fetch(`${serverUrl}/api/general/interrupt`, {
            method: "POST"
          }).then((res) => {
            if (res.status === 200) {
              this.callback();
              this.isChanging.value = false;
            }
          }).catch((err) => {
            this.isChanging.value = false;
            console.error(err);
          });
        } else {
          this.callback();
          this.isChanging.value = false;
        }
      }, this.timeoutDuration);
    }
  }
  resetTimer() {
    if (this.timer) {
      clearTimeout(this.timer);
    }
    this.timer = setTimeout(() => {
      fetch(`${serverUrl}/api/general/interrupt`, {
        method: "POST"
      }).then((res) => {
        if (res.status === 200) {
          this.callback();
          this.isChanging.value = false;
        }
      }).catch((err) => {
        this.isChanging.value = false;
        console.error(err);
      });
    }, this.timeoutDuration);
  }
  cleanup() {
    if (this.timer) {
      clearTimeout(this.timer);
    }
  }
}
export {
  BurnerClock as B,
  Prompt as P,
  _sfc_main$3 as _,
  _sfc_main$1 as a,
  _sfc_main as b,
  _sfc_main$4 as c,
  _sfc_main$5 as d
};
