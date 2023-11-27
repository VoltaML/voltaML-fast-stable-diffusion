import { d as defineComponent, u as useSettings, o as openBlock, e as createBlock, w as withCtx, g as createVNode, h as unref, J as NInput, i as NSelect, n as NCard, b8 as reactive, c as computed, by as convertToTextString, z as ref, t as serverUrl, K as watch, a as useState, f as createBaseVNode, j as createElementBlock, L as renderList, bb as NText, k as createTextVNode, C as toDisplayString, F as Fragment, m as createCommentVNode, D as NTabPane, E as NTabs, R as inject, bz as themeKey, A as NButton, l as NTooltip, p as useMessage, bA as useNotification, q as onUnmounted, bB as defaultSettings } from "./index.js";
import { a as NFormItem, _ as _sfc_main$h, N as NForm } from "./SamplerPicker.vue_vue_type_script_setup_true_lang.js";
import { N as NSwitch, a as NSlider } from "./Switch.js";
import { N as NInputNumber } from "./InputNumber.js";
import "./Settings.js";
const _sfc_main$g = /* @__PURE__ */ defineComponent({
  __name: "ControlNetSettings",
  setup(__props) {
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), null, {
        default: withCtx(() => [
          createVNode(unref(NForm), null, {
            default: withCtx(() => [
              createVNode(unref(NFormItem), {
                label: "Prompt",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInput), {
                    value: unref(settings).defaultSettings.controlnet.prompt,
                    "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.controlnet.prompt = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Negative Prompt",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInput), {
                    value: unref(settings).defaultSettings.controlnet.negative_prompt,
                    "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.controlnet.negative_prompt = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Batch Count",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.controlnet.batch_count,
                    "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.controlnet.batch_count = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Batch Size",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.controlnet.batch_size,
                    "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).defaultSettings.controlnet.batch_size = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "CFG Scale",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.controlnet.cfg_scale,
                    "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).defaultSettings.controlnet.cfg_scale = $event),
                    step: 0.1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Height",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.controlnet.height,
                    "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(settings).defaultSettings.controlnet.height = $event),
                    step: 8
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Width",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.controlnet.width,
                    "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(settings).defaultSettings.controlnet.width = $event),
                    step: 8
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "ControlNet",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSelect), {
                    options: unref(settings).controlnet_options,
                    value: unref(settings).defaultSettings.controlnet.controlnet,
                    "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(settings).defaultSettings.controlnet.controlnet = $event),
                    filterable: "",
                    tag: ""
                  }, null, 8, ["options", "value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Seed",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.controlnet.seed,
                    "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(settings).defaultSettings.controlnet.seed = $event),
                    min: -1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Is Preprocessed",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSwitch), {
                    value: unref(settings).defaultSettings.controlnet.is_preprocessed,
                    "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(settings).defaultSettings.controlnet.is_preprocessed = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Steps",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.controlnet.steps,
                    "onUpdate:value": _cache[10] || (_cache[10] = ($event) => unref(settings).defaultSettings.controlnet.steps = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "ControlNet Conditioning Scale",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.controlnet.controlnet_conditioning_scale,
                    "onUpdate:value": _cache[11] || (_cache[11] = ($event) => unref(settings).defaultSettings.controlnet.controlnet_conditioning_scale = $event),
                    step: 0.1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Detection Resolution",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.controlnet.detection_resolution,
                    "onUpdate:value": _cache[12] || (_cache[12] = ($event) => unref(settings).defaultSettings.controlnet.detection_resolution = $event),
                    step: 8
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(_sfc_main$h, {
                type: "controlnet",
                target: "defaultSettings"
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
const _sfc_main$f = /* @__PURE__ */ defineComponent({
  __name: "ImageBrowserSettings",
  setup(__props) {
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NForm), null, {
        default: withCtx(() => [
          createVNode(unref(NCard), null, {
            default: withCtx(() => [
              createVNode(unref(NFormItem), {
                label: "Number of columns",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.frontend.image_browser_columns,
                    "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.frontend.image_browser_columns = $event)
                  }, null, 8, ["value"])
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
const _sfc_main$e = /* @__PURE__ */ defineComponent({
  __name: "ImageToImageSettings",
  setup(__props) {
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), null, {
        default: withCtx(() => [
          createVNode(unref(NForm), null, {
            default: withCtx(() => [
              createVNode(unref(NFormItem), {
                label: "Prompt",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInput), {
                    value: unref(settings).defaultSettings.img2img.prompt,
                    "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.img2img.prompt = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Negative Prompt",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInput), {
                    value: unref(settings).defaultSettings.img2img.negative_prompt,
                    "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.img2img.negative_prompt = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Batch Count",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.img2img.batch_count,
                    "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.img2img.batch_count = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Batch Size",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.img2img.batch_size,
                    "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).defaultSettings.img2img.batch_size = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "CFG Scale",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.img2img.cfg_scale,
                    "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).defaultSettings.img2img.cfg_scale = $event),
                    step: 0.1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Height",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.img2img.height,
                    "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(settings).defaultSettings.img2img.height = $event),
                    step: 1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Width",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.img2img.width,
                    "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(settings).defaultSettings.img2img.width = $event),
                    step: 1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Seed",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.img2img.seed,
                    "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(settings).defaultSettings.img2img.seed = $event),
                    min: -1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Steps",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.img2img.steps,
                    "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(settings).defaultSettings.img2img.steps = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Denoising Strength",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.img2img.denoising_strength,
                    "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(settings).defaultSettings.img2img.denoising_strength = $event),
                    step: 0.1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(_sfc_main$h, {
                type: "img2img",
                target: "defaultSettings"
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
const _sfc_main$d = /* @__PURE__ */ defineComponent({
  __name: "InpaintingSettings",
  setup(__props) {
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), null, {
        default: withCtx(() => [
          createVNode(unref(NForm), null, {
            default: withCtx(() => [
              createVNode(unref(NFormItem), {
                label: "Prompt",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInput), {
                    value: unref(settings).defaultSettings.inpainting.prompt,
                    "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.inpainting.prompt = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Negative Prompt",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInput), {
                    value: unref(settings).defaultSettings.inpainting.negative_prompt,
                    "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.inpainting.negative_prompt = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Batch Count",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.inpainting.batch_count,
                    "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.inpainting.batch_count = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Batch Size",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.inpainting.batch_size,
                    "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).defaultSettings.inpainting.batch_size = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "CFG Scale",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.inpainting.cfg_scale,
                    "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).defaultSettings.inpainting.cfg_scale = $event),
                    step: 0.1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Height",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.inpainting.height,
                    "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(settings).defaultSettings.inpainting.height = $event),
                    step: 8
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Width",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.inpainting.width,
                    "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(settings).defaultSettings.inpainting.width = $event),
                    step: 8
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Seed",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.inpainting.seed,
                    "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(settings).defaultSettings.inpainting.seed = $event),
                    min: -1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Steps",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.inpainting.steps,
                    "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(settings).defaultSettings.inpainting.steps = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(_sfc_main$h, {
                type: "inpainting",
                target: "defaultSettings"
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
const _sfc_main$c = /* @__PURE__ */ defineComponent({
  __name: "TextToImageSettings",
  setup(__props) {
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), null, {
        default: withCtx(() => [
          createVNode(unref(NForm), null, {
            default: withCtx(() => [
              createVNode(unref(NFormItem), {
                label: "Prompt",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInput), {
                    value: unref(settings).defaultSettings.txt2img.prompt,
                    "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.txt2img.prompt = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Negative Prompt",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInput), {
                    value: unref(settings).defaultSettings.txt2img.negative_prompt,
                    "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.txt2img.negative_prompt = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Batch Count",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.txt2img.batch_count,
                    "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.txt2img.batch_count = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Batch Size",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.txt2img.batch_size,
                    "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).defaultSettings.txt2img.batch_size = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "CFG Scale",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.txt2img.cfg_scale,
                    "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).defaultSettings.txt2img.cfg_scale = $event),
                    step: 0.1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Height",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.txt2img.height,
                    "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(settings).defaultSettings.txt2img.height = $event),
                    step: 1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Width",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.txt2img.width,
                    "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(settings).defaultSettings.txt2img.width = $event),
                    step: 1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Seed",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.txt2img.seed,
                    "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(settings).defaultSettings.txt2img.seed = $event),
                    min: -1
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Steps",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.txt2img.steps,
                    "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(settings).defaultSettings.txt2img.steps = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(_sfc_main$h, {
                type: "txt2img",
                target: "defaultSettings"
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
const _sfc_main$b = /* @__PURE__ */ defineComponent({
  __name: "ThemeSettings",
  setup(__props) {
    const settings = useSettings();
    const extraThemes = reactive([]);
    const themeOptions = computed(() => {
      return extraThemes.map((theme) => {
        return { label: convertToTextString(theme), value: theme };
      });
    });
    const themesLoading = ref(true);
    fetch(`${serverUrl}/api/general/themes`).then(async (res) => {
      const data = await res.json();
      extraThemes.push(...data);
      themesLoading.value = false;
    }).catch((err) => {
      console.error(err);
      themesLoading.value = false;
    });
    watch(settings.defaultSettings.frontend, () => {
      settings.data.settings.frontend = settings.defaultSettings.frontend;
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), null, {
        default: withCtx(() => [
          createVNode(unref(NForm), null, {
            default: withCtx(() => [
              createVNode(unref(NFormItem), {
                label: "Theme",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSelect), {
                    options: themeOptions.value,
                    value: unref(settings).defaultSettings.frontend.theme,
                    "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.frontend.theme = $event),
                    loading: themesLoading.value,
                    filterable: ""
                  }, null, 8, ["options", "value", "loading"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Background Image Override",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInput), {
                    value: unref(settings).defaultSettings.frontend.background_image_override,
                    "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.frontend.background_image_override = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Enable Theme Editor",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSwitch), {
                    value: unref(settings).defaultSettings.frontend.enable_theme_editor,
                    "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.frontend.enable_theme_editor = $event)
                  }, null, 8, ["value"])
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
const _hoisted_1$3 = { style: { "width": "100%" } };
const _sfc_main$a = /* @__PURE__ */ defineComponent({
  __name: "AutoloadSettings",
  setup(__props) {
    const settings = useSettings();
    const global = useState();
    const textualInversions = computed(() => {
      return global.state.models.filter((model) => {
        return model.backend === "Textual Inversion";
      });
    });
    const textualInversionOptions = computed(() => {
      return textualInversions.value.map((model) => {
        return {
          value: model.path,
          label: model.name
        };
      });
    });
    const availableModels = computed(() => {
      return global.state.models.filter((model) => {
        return model.backend === "AITemplate" || model.backend === "PyTorch" || model.backend === "ONNX";
      });
    });
    const availableVaes = computed(() => {
      return global.state.models.filter((model) => {
        return model.backend === "VAE";
      });
    });
    const autoloadModelOptions = computed(() => {
      return availableModels.value.map((model) => {
        return {
          value: model.path,
          label: model.name
        };
      });
    });
    const autoloadVaeOptions = computed(() => {
      const arr = availableVaes.value.map((model) => {
        return {
          value: model.path,
          label: model.name
        };
      });
      arr.push({ value: "default", label: "Default" });
      return arr;
    });
    const autoloadVaeValue = (model) => {
      return computed({
        get: () => {
          return settings.defaultSettings.api.autoloaded_vae[model] ?? "default";
        },
        set: (value) => {
          if (!value || value === "default") {
            delete settings.defaultSettings.api.autoloaded_vae[model];
          } else {
            settings.defaultSettings.api.autoloaded_vae[model] = value;
          }
        }
      });
    };
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NForm), null, {
        default: withCtx(() => [
          createVNode(unref(NFormItem), {
            label: "Model",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                multiple: "",
                filterable: "",
                options: autoloadModelOptions.value,
                value: unref(settings).defaultSettings.api.autoloaded_models,
                "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.api.autoloaded_models = $event)
              }, null, 8, ["options", "value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Textual Inversions",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                multiple: "",
                filterable: "",
                options: textualInversionOptions.value,
                value: unref(settings).defaultSettings.api.autoloaded_textual_inversions,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.api.autoloaded_textual_inversions = $event)
              }, null, 8, ["options", "value"])
            ]),
            _: 1
          }),
          createVNode(unref(NCard), { title: "VAE" }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_1$3, [
                (openBlock(true), createElementBlock(Fragment, null, renderList(availableModels.value, (model) => {
                  return openBlock(), createElementBlock("div", {
                    key: model.name,
                    style: { "display": "flex", "flex-direction": "row", "margin-bottom": "4px" }
                  }, [
                    createVNode(unref(NText), { style: { "width": "50%" } }, {
                      default: withCtx(() => [
                        createTextVNode(toDisplayString(model.name), 1)
                      ]),
                      _: 2
                    }, 1024),
                    createVNode(unref(NSelect), {
                      filterable: "",
                      options: autoloadVaeOptions.value,
                      value: autoloadVaeValue(model.path).value,
                      "onUpdate:value": ($event) => autoloadVaeValue(model.path).value = $event
                    }, null, 8, ["options", "value", "onUpdate:value"])
                  ]);
                }), 128))
              ])
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
const _sfc_main$9 = /* @__PURE__ */ defineComponent({
  __name: "BotSettings",
  setup(__props) {
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), null, {
        default: withCtx(() => [
          createVNode(unref(NForm), null, {
            default: withCtx(() => [
              createVNode(unref(NFormItem), {
                label: "Default Scheduler",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSelect), {
                    options: unref(settings).scheduler_options,
                    value: unref(settings).defaultSettings.bot.default_scheduler,
                    "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.bot.default_scheduler = $event)
                  }, null, 8, ["options", "value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Use Default Negative Prompt",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSwitch), {
                    value: unref(settings).defaultSettings.bot.use_default_negative_prompt,
                    "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.bot.use_default_negative_prompt = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Verbose",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSwitch), {
                    value: unref(settings).defaultSettings.bot.verbose,
                    "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.bot.verbose = $event)
                  }, null, 8, ["value"])
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
const _sfc_main$8 = /* @__PURE__ */ defineComponent({
  __name: "FilesSettings",
  setup(__props) {
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NForm), null, {
        default: withCtx(() => [
          createVNode(unref(NFormItem), {
            label: "Template for saving outputs",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NInput), {
                value: unref(settings).defaultSettings.api.save_path_template,
                "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.api.save_path_template = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Disable generating grid image",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.disable_grid,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.api.disable_grid = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Image extension",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                value: unref(settings).defaultSettings.api.image_extension,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.api.image_extension = $event),
                options: [
                  {
                    label: "PNG",
                    value: "png"
                  },
                  {
                    label: "WebP",
                    value: "webp"
                  },
                  {
                    label: "JPEG",
                    value: "jpeg"
                  }
                ]
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          unref(settings).defaultSettings.api.image_extension != "png" ? (openBlock(), createBlock(unref(NFormItem), {
            key: 0,
            label: "Image quality (JPEG/WebP only)",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.api.image_quality,
                "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).defaultSettings.api.image_quality = $event),
                min: 0,
                max: 100,
                step: 1
              }, null, 8, ["value"])
            ]),
            _: 1
          })) : createCommentVNode("", true)
        ]),
        _: 1
      });
    };
  }
});
const _sfc_main$7 = /* @__PURE__ */ defineComponent({
  __name: "FlagsSettings",
  setup(__props) {
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), { title: "Hi-res fix" }, {
        default: withCtx(() => [
          createVNode(unref(NForm), null, {
            default: withCtx(() => [
              createVNode(unref(NFormItem), {
                label: "Scale",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.flags.highres.scale,
                    "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.flags.highres.scale = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Scaling Mode",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSelect), {
                    options: [
                      {
                        label: "Nearest",
                        value: "nearest"
                      },
                      {
                        label: "Linear",
                        value: "linear"
                      },
                      {
                        label: "Bilinear",
                        value: "bilinear"
                      },
                      {
                        label: "Bicubic",
                        value: "bicubic"
                      },
                      {
                        label: "Bislerp",
                        value: "bislerp"
                      },
                      {
                        label: "Nearest Exact",
                        value: "nearest-exact"
                      }
                    ],
                    value: unref(settings).defaultSettings.flags.highres.latent_scale_mode,
                    "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.flags.highres.latent_scale_mode = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Strength",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.flags.highres.strength,
                    "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.flags.highres.strength = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Steps",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.flags.highres.steps,
                    "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).defaultSettings.flags.highres.steps = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Antialiased",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSwitch), {
                    value: unref(settings).defaultSettings.flags.highres.antialiased,
                    "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).defaultSettings.flags.highres.antialiased = $event)
                  }, null, 8, ["value"])
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
const _sfc_main$6 = /* @__PURE__ */ defineComponent({
  __name: "FrontendSettings",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NTabs), null, {
        default: withCtx(() => [
          createVNode(unref(NTabPane), { name: "Text to Image" }, {
            default: withCtx(() => [
              createVNode(unref(_sfc_main$c))
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Image to Image" }, {
            default: withCtx(() => [
              createVNode(unref(_sfc_main$e))
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "ControlNet" }, {
            default: withCtx(() => [
              createVNode(unref(_sfc_main$g))
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Inpainting" }, {
            default: withCtx(() => [
              createVNode(unref(_sfc_main$d))
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Image Browser" }, {
            default: withCtx(() => [
              createVNode(unref(_sfc_main$f))
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
const _sfc_main$5 = /* @__PURE__ */ defineComponent({
  __name: "GeneralSettings",
  setup(__props) {
    const settings = useSettings();
    watch(settings.defaultSettings.frontend, () => {
      settings.data.settings.frontend.on_change_timer = settings.defaultSettings.frontend.on_change_timer;
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), { title: "Timings" }, {
        default: withCtx(() => [
          createVNode(unref(NForm), null, {
            default: withCtx(() => [
              createVNode(unref(NFormItem), {
                label: "Continuous generation timeout (0 for disabled) [ms]",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NInputNumber), {
                    value: unref(settings).defaultSettings.frontend.on_change_timer,
                    "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.frontend.on_change_timer = $event),
                    min: 0,
                    step: 50
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Enable sending logs to UI",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSwitch), {
                    value: unref(settings).defaultSettings.api.enable_websocket_logging,
                    "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.api.enable_websocket_logging = $event)
                  }, null, 8, ["value"])
                ]),
                _: 1
              }),
              createVNode(unref(NFormItem), {
                label: "Disable Analytics",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSwitch), {
                    value: unref(settings).defaultSettings.frontend.disable_analytics,
                    "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.frontend.disable_analytics = $event)
                  }, null, 8, ["value"])
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
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "NSFWSettings",
  setup(__props) {
    const settings = useSettings();
    watch(
      () => settings.defaultSettings.frontend.nsfw_ok_threshold,
      (value) => {
        settings.data.settings.frontend.nsfw_ok_threshold = value;
      }
    );
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), null, {
        default: withCtx(() => [
          createVNode(unref(NForm), null, {
            default: withCtx(() => [
              createVNode(unref(NFormItem), {
                label: "NSFW OK threshold (if you don't get the reference, select `I'm too young to die`)",
                "label-placement": "left"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NSelect), {
                    value: unref(settings).defaultSettings.frontend.nsfw_ok_threshold,
                    "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.frontend.nsfw_ok_threshold = $event),
                    options: [
                      {
                        label: "I'm too young to die",
                        value: 0
                      },
                      {
                        label: "Hurt me plenty",
                        value: 1
                      },
                      {
                        label: "Ultra violence",
                        value: 2
                      },
                      {
                        label: "Nightmare",
                        value: 3
                      }
                    ]
                  }, null, 8, ["value"])
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
const _hoisted_1$2 = {
  key: 0,
  class: "flex-container"
};
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Subquadratic chunk size (affects VRAM usage)", -1);
const _hoisted_3$1 = { "flex-direction": "row" };
const _hoisted_4$1 = { key: 1 };
const _hoisted_5$1 = { key: 2 };
const _hoisted_6$1 = { class: "flex-container" };
const _hoisted_7$1 = /* @__PURE__ */ createBaseVNode("p", { class: "switch-label" }, "Don't merge latents", -1);
const _hoisted_8$1 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "PyTorch ONLY.", -1);
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "OptimizationSettings",
  setup(__props) {
    const settings = useSettings();
    const global = useState();
    const theme = inject(themeKey);
    const compileColor = computed(() => {
      var _a;
      if (settings.defaultSettings.api.torch_compile)
        return (_a = theme == null ? void 0 : theme.value.Button.common) == null ? void 0 : _a.successColor;
      return void 0;
    });
    const traceColor = computed(() => {
      var _a;
      if (settings.defaultSettings.api.trace_model)
        return (_a = theme == null ? void 0 : theme.value.Button.common) == null ? void 0 : _a.successColor;
      return void 0;
    });
    const sfastColor = computed(() => {
      if (settings.defaultSettings.api.sfast_compile)
        return "#f1f1f1";
      return void 0;
    });
    const disableColor = computed(() => {
      var _a;
      if (settings.defaultSettings.api.torch_compile || settings.defaultSettings.api.trace_model || settings.defaultSettings.api.sfast_compile)
        return void 0;
      return (_a = theme == null ? void 0 : theme.value.Button.common) == null ? void 0 : _a.successColor;
    });
    function change_compilation(a) {
      settings.defaultSettings.api.torch_compile = a === "compile";
      settings.defaultSettings.api.trace_model = a === "trace";
      settings.defaultSettings.api.sfast_compile = a === "sfast";
    }
    const availableTorchCompileBackends = computed(() => {
      return global.state.capabilities.supported_torch_compile_backends.map(
        (value) => {
          return { value, label: value };
        }
      );
    });
    const availableAttentions = computed(() => {
      return global.state.capabilities.supported_self_attentions.map((l) => {
        return { value: l[1], label: l[0] };
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NForm), null, {
        default: withCtx(() => [
          createVNode(unref(NFormItem), {
            label: "Autocast",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.autocast,
                "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.api.autocast = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Attention processor",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                options: availableAttentions.value,
                value: unref(settings).defaultSettings.api.attention_processor,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.api.attention_processor = $event)
              }, null, 8, ["options", "value"])
            ]),
            _: 1
          }),
          unref(settings).defaultSettings.api.attention_processor == "subquadratic" ? (openBlock(), createElementBlock("div", _hoisted_1$2, [
            _hoisted_2$1,
            createVNode(unref(NSlider), {
              value: unref(settings).defaultSettings.api.subquadratic_size,
              "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.api.subquadratic_size = $event),
              step: 64,
              min: 64,
              max: 8192,
              style: { "margin-right": "12px" }
            }, null, 8, ["value"]),
            createVNode(unref(NInputNumber), {
              value: unref(settings).defaultSettings.api.subquadratic_size,
              "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).defaultSettings.api.subquadratic_size = $event),
              size: "small",
              style: { "min-width": "96px", "width": "96px" },
              step: 64,
              min: 64,
              max: 8192
            }, null, 8, ["value"])
          ])) : createCommentVNode("", true),
          createVNode(unref(NFormItem), {
            label: "Compilation method",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_3$1, [
                createVNode(unref(NButton), {
                  onClick: _cache[4] || (_cache[4] = ($event) => change_compilation("disabled")),
                  color: disableColor.value
                }, {
                  default: withCtx(() => [
                    createTextVNode("Disabled")
                  ]),
                  _: 1
                }, 8, ["color"]),
                createVNode(unref(NButton), {
                  onClick: _cache[5] || (_cache[5] = ($event) => change_compilation("trace")),
                  color: traceColor.value
                }, {
                  default: withCtx(() => [
                    createTextVNode("Trace UNet")
                  ]),
                  _: 1
                }, 8, ["color"]),
                createVNode(unref(NButton), {
                  onClick: _cache[6] || (_cache[6] = ($event) => change_compilation("compile")),
                  color: compileColor.value
                }, {
                  default: withCtx(() => [
                    createTextVNode("torch.compile")
                  ]),
                  _: 1
                }, 8, ["color"]),
                createVNode(unref(NButton), {
                  onClick: _cache[7] || (_cache[7] = ($event) => change_compilation("sfast")),
                  color: sfastColor.value
                }, {
                  default: withCtx(() => [
                    createTextVNode("stable-fast")
                  ]),
                  _: 1
                }, 8, ["color"])
              ])
            ]),
            _: 1
          }),
          unref(settings).defaultSettings.api.sfast_compile ? (openBlock(), createElementBlock("div", _hoisted_4$1, [
            createVNode(unref(NFormItem), {
              label: "Use xFormers during compilation",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSwitch), {
                  value: unref(settings).defaultSettings.api.sfast_xformers,
                  "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(settings).defaultSettings.api.sfast_xformers = $event),
                  disabled: !unref(global).state.capabilities.supports_xformers
                }, null, 8, ["value", "disabled"])
              ]),
              _: 1
            }),
            createVNode(unref(NFormItem), {
              label: "Use Triton during compilation",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSwitch), {
                  value: unref(settings).defaultSettings.api.sfast_triton,
                  "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(settings).defaultSettings.api.sfast_triton = $event),
                  disabled: !unref(global).state.capabilities.supports_triton
                }, null, 8, ["value", "disabled"])
              ]),
              _: 1
            }),
            createVNode(unref(NFormItem), {
              label: "Use CUDA graphs during compilation",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSwitch), {
                  value: unref(settings).defaultSettings.api.sfast_cuda_graph,
                  "onUpdate:value": _cache[10] || (_cache[10] = ($event) => unref(settings).defaultSettings.api.sfast_cuda_graph = $event)
                }, null, 8, ["value"])
              ]),
              _: 1
            })
          ])) : createCommentVNode("", true),
          unref(settings).defaultSettings.api.torch_compile ? (openBlock(), createElementBlock("div", _hoisted_5$1, [
            createVNode(unref(NFormItem), {
              label: "Fullgraph compile",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSwitch), {
                  value: unref(settings).defaultSettings.api.torch_compile_fullgraph,
                  "onUpdate:value": _cache[11] || (_cache[11] = ($event) => unref(settings).defaultSettings.api.torch_compile_fullgraph = $event)
                }, null, 8, ["value"])
              ]),
              _: 1
            }),
            createVNode(unref(NFormItem), {
              label: "Dynamic compile",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSwitch), {
                  value: unref(settings).defaultSettings.api.torch_compile_dynamic,
                  "onUpdate:value": _cache[12] || (_cache[12] = ($event) => unref(settings).defaultSettings.api.torch_compile_dynamic = $event)
                }, null, 8, ["value"])
              ]),
              _: 1
            }),
            createVNode(unref(NFormItem), {
              label: "Compilation backend",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSelect), {
                  options: availableTorchCompileBackends.value,
                  value: unref(settings).defaultSettings.api.torch_compile_backend,
                  "onUpdate:value": _cache[13] || (_cache[13] = ($event) => unref(settings).defaultSettings.api.torch_compile_backend = $event)
                }, null, 8, ["options", "value"])
              ]),
              _: 1
            }),
            createVNode(unref(NFormItem), {
              label: "Compilation mode",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSelect), {
                  options: [
                    { value: "default", label: "Default" },
                    { value: "reduce-overhead", label: "Reduce Overhead" },
                    { value: "max-autotune", label: "Max Autotune" }
                  ],
                  value: unref(settings).defaultSettings.api.torch_compile_mode,
                  "onUpdate:value": _cache[14] || (_cache[14] = ($event) => unref(settings).defaultSettings.api.torch_compile_mode = $event)
                }, null, 8, ["value"])
              ]),
              _: 1
            })
          ])) : createCommentVNode("", true),
          createVNode(unref(NFormItem), {
            label: "Attention Slicing",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                options: [
                  {
                    value: "disabled",
                    label: "None"
                  },
                  {
                    value: "auto",
                    label: "Auto"
                  }
                ],
                value: unref(settings).defaultSettings.api.attention_slicing,
                "onUpdate:value": _cache[15] || (_cache[15] = ($event) => unref(settings).defaultSettings.api.attention_slicing = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Channels Last",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.channels_last,
                "onUpdate:value": _cache[16] || (_cache[16] = ($event) => unref(settings).defaultSettings.api.channels_last = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Reduced Precision (RTX 30xx and newer cards)",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.reduced_precision,
                "onUpdate:value": _cache[17] || (_cache[17] = ($event) => unref(settings).defaultSettings.api.reduced_precision = $event),
                disabled: !unref(global).state.capabilities.has_tensorfloat
              }, null, 8, ["value", "disabled"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "CudNN Benchmark (big VRAM spikes - use on 8GB+ cards only)",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.cudnn_benchmark,
                "onUpdate:value": _cache[18] || (_cache[18] = ($event) => unref(settings).defaultSettings.api.cudnn_benchmark = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Clean Memory",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                options: [
                  {
                    value: "always",
                    label: "Always"
                  },
                  {
                    value: "never",
                    label: "Never"
                  },
                  {
                    value: "after_disconnect",
                    label: "After disconnect"
                  }
                ],
                value: unref(settings).defaultSettings.api.clear_memory_policy,
                "onUpdate:value": _cache[19] || (_cache[19] = ($event) => unref(settings).defaultSettings.api.clear_memory_policy = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "VAE Slicing",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.vae_slicing,
                "onUpdate:value": _cache[20] || (_cache[20] = ($event) => unref(settings).defaultSettings.api.vae_slicing = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "VAE Tiling",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.vae_tiling,
                "onUpdate:value": _cache[21] || (_cache[21] = ($event) => unref(settings).defaultSettings.api.vae_tiling = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Offload",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                options: [
                  {
                    value: "disabled",
                    label: "Disabled"
                  },
                  {
                    value: "model",
                    label: "Offload the whole model to RAM when not used"
                  },
                  {
                    value: "module",
                    label: "Offload individual modules to RAM when not used"
                  }
                ],
                value: unref(settings).defaultSettings.api.offload,
                "onUpdate:value": _cache[22] || (_cache[22] = ($event) => unref(settings).defaultSettings.api.offload = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createBaseVNode("div", _hoisted_6$1, [
            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
              trigger: withCtx(() => [
                _hoisted_7$1
              ]),
              default: withCtx(() => [
                _hoisted_8$1,
                createTextVNode(" Doesn't merge latents into a single one during UNet inference, and instead does both the negatives and positives separately. Saves around 200-300mBs of VRAM during inference for a ~10% speed regression. ")
              ]),
              _: 1
            }),
            createVNode(unref(NSwitch), {
              value: unref(settings).defaultSettings.api.dont_merge_latents,
              "onUpdate:value": _cache[23] || (_cache[23] = ($event) => unref(settings).defaultSettings.api.dont_merge_latents = $event)
            }, null, 8, ["value"])
          ])
        ]),
        _: 1
      });
    };
  }
});
const _hoisted_1$1 = { key: 1 };
const _hoisted_2 = { class: "flex-container" };
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Hypertile UNet chunk size", -1);
const _hoisted_4 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, 'PyTorch ONLY. Recommended sizes are 1/4th your desired resolution or plain "256."', -1);
const _hoisted_5 = /* @__PURE__ */ createBaseVNode("b", null, "LARGE (1024x1024+)", -1);
const _hoisted_6 = { key: 2 };
const _hoisted_7 = { key: 3 };
const _hoisted_8 = { key: 4 };
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "ReproducibilitySettings",
  setup(__props) {
    const settings = useSettings();
    const global = useState();
    const enabledCfg = computed({
      get() {
        return settings.defaultSettings.api.cfg_rescale_threshold != "off";
      },
      set(value) {
        if (!value) {
          settings.defaultSettings.api.cfg_rescale_threshold = "off";
        } else {
          settings.defaultSettings.api.cfg_rescale_threshold = 10;
        }
      }
    });
    const cfgRescaleValue = computed({
      get() {
        if (settings.defaultSettings.api.cfg_rescale_threshold == "off") {
          return 1;
        }
        return settings.defaultSettings.api.cfg_rescale_threshold;
      },
      set(value) {
        settings.defaultSettings.api.cfg_rescale_threshold = value;
      }
    });
    const availableDtypes = computed(() => {
      if (settings.defaultSettings.api.device.includes("cpu")) {
        return global.state.capabilities.supported_precisions_cpu.map((value) => {
          var description = "";
          switch (value) {
            case "float32":
              description = "32-bit float";
              break;
            case "float16":
              description = "16-bit float";
              break;
            case "float8_e5m2":
              description = "8-bit float (5-data)";
              break;
            case "float8_e4m3fn":
              description = "8-bit float (4-data)";
              break;
            default:
              description = "16-bit bfloat";
          }
          return { value, label: description };
        });
      }
      return global.state.capabilities.supported_precisions_gpu.map((value) => {
        var description = "";
        switch (value) {
          case "float32":
            description = "32-bit float";
            break;
          case "float16":
            description = "16-bit float";
            break;
          case "float8_e5m2":
            description = "8-bit float (5-data)";
            break;
          case "float8_e4m3fn":
            description = "8-bit float (4-data)";
            break;
          default:
            description = "16-bit bfloat";
        }
        return { value, label: description };
      });
    });
    const availableBackends = computed(() => {
      return global.state.capabilities.supported_backends.map((l) => {
        return { value: l[1], label: l[0] };
      });
    });
    const availableQuantizations = computed(() => {
      return [
        { value: "full", label: "Full precision" },
        ...global.state.capabilities.supports_int8 ? [
          { value: "int8", label: "Quantized (int8)" },
          { value: "int4", label: "Quantized (int4)" }
        ] : []
      ];
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NForm), null, {
        default: withCtx(() => [
          createVNode(unref(NFormItem), {
            label: "Device",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                options: availableBackends.value,
                value: unref(settings).defaultSettings.api.device,
                "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.api.device = $event)
              }, null, 8, ["options", "value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Data type",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                options: availableDtypes.value,
                value: unref(settings).defaultSettings.api.data_type,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.api.data_type = $event)
              }, null, 8, ["options", "value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Deterministic generation",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.deterministic_generation,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.api.deterministic_generation = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "SGM Noise multiplier",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.sgm_noise_multiplier,
                "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).defaultSettings.api.sgm_noise_multiplier = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Quantization in k-samplers",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.kdiffusers_quantization,
                "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).defaultSettings.api.kdiffusers_quantization = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Generator",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                value: unref(settings).defaultSettings.api.generator,
                "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(settings).defaultSettings.api.generator = $event),
                options: [
                  { value: "device", label: "On-Device" },
                  { value: "cpu", label: "CPU" },
                  { value: "philox", label: "CPU (device mock)" }
                ]
              }, null, 8, ["value", "options"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "CLIP skip",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.api.clip_skip,
                "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(settings).defaultSettings.api.clip_skip = $event),
                min: 1,
                max: 11,
                step: 1
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          availableQuantizations.value.length != 1 ? (openBlock(), createBlock(unref(NFormItem), {
            key: 0,
            label: "CLIP quantization",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                options: availableQuantizations.value,
                value: unref(settings).defaultSettings.api.clip_quantization,
                "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(settings).defaultSettings.api.clip_quantization = $event)
              }, null, 8, ["options", "value"])
            ]),
            _: 1
          })) : createCommentVNode("", true),
          createVNode(unref(NFormItem), {
            label: "Use HyperTile",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.hypertile,
                "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(settings).defaultSettings.api.hypertile = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          unref(settings).defaultSettings.api.hypertile ? (openBlock(), createElementBlock("div", _hoisted_1$1, [
            createBaseVNode("div", _hoisted_2, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_3
                ]),
                default: withCtx(() => [
                  _hoisted_4,
                  createTextVNode(" Internally splits up the generated image into a grid of this size and does work on them one by one. In practice, this can make generation up to 4x faster on "),
                  _hoisted_5,
                  createTextVNode(" images. ")
                ]),
                _: 1
              }),
              createVNode(unref(NSlider), {
                value: unref(settings).defaultSettings.api.hypertile_unet_chunk,
                "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(settings).defaultSettings.api.hypertile_unet_chunk = $event),
                min: 128,
                max: 1024,
                step: 8,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.api.hypertile_unet_chunk,
                "onUpdate:value": _cache[10] || (_cache[10] = ($event) => unref(settings).defaultSettings.api.hypertile_unet_chunk = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                min: 128,
                max: 1024,
                step: 1
              }, null, 8, ["value"])
            ])
          ])) : createCommentVNode("", true),
          createVNode(unref(NFormItem), {
            label: "Use TomeSD",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.use_tomesd,
                "onUpdate:value": _cache[11] || (_cache[11] = ($event) => unref(settings).defaultSettings.api.use_tomesd = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          unref(settings).defaultSettings.api.use_tomesd ? (openBlock(), createElementBlock("div", _hoisted_6, [
            createVNode(unref(NFormItem), {
              label: "TomeSD Ratio",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NInputNumber), {
                  value: unref(settings).defaultSettings.api.tomesd_ratio,
                  "onUpdate:value": _cache[12] || (_cache[12] = ($event) => unref(settings).defaultSettings.api.tomesd_ratio = $event),
                  min: 0.1,
                  max: 1
                }, null, 8, ["value"])
              ]),
              _: 1
            }),
            createVNode(unref(NFormItem), {
              label: "TomeSD Downsample layers",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSelect), {
                  options: [
                    {
                      value: 1,
                      label: "1"
                    },
                    {
                      value: 2,
                      label: "2"
                    },
                    {
                      value: 4,
                      label: "4"
                    },
                    {
                      value: 8,
                      label: "8"
                    }
                  ],
                  value: unref(settings).defaultSettings.api.tomesd_downsample_layers,
                  "onUpdate:value": _cache[13] || (_cache[13] = ($event) => unref(settings).defaultSettings.api.tomesd_downsample_layers = $event)
                }, null, 8, ["value"])
              ]),
              _: 1
            })
          ])) : createCommentVNode("", true),
          createVNode(unref(NFormItem), {
            label: "Huggingface-style prompting",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.huggingface_style_parsing,
                "onUpdate:value": _cache[14] || (_cache[14] = ($event) => unref(settings).defaultSettings.api.huggingface_style_parsing = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Prompt-to-Prompt preprocessing",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.prompt_to_prompt,
                "onUpdate:value": _cache[15] || (_cache[15] = ($event) => unref(settings).defaultSettings.api.prompt_to_prompt = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          unref(settings).defaultSettings.api.prompt_to_prompt ? (openBlock(), createElementBlock("div", _hoisted_7, [
            createVNode(unref(NFormItem), {
              label: "Prompt-to-Prompt model",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSelect), {
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
                  value: unref(settings).defaultSettings.api.prompt_to_prompt_model,
                  "onUpdate:value": _cache[16] || (_cache[16] = ($event) => unref(settings).defaultSettings.api.prompt_to_prompt_model = $event)
                }, null, 8, ["value"])
              ]),
              _: 1
            }),
            createVNode(unref(NFormItem), {
              label: "Prompt-to-Prompt device",
              "label-placement": "left"
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
                  value: unref(settings).defaultSettings.api.prompt_to_prompt_device,
                  "onUpdate:value": _cache[17] || (_cache[17] = ($event) => unref(settings).defaultSettings.api.prompt_to_prompt_device = $event)
                }, null, 8, ["value"])
              ]),
              _: 1
            })
          ])) : createCommentVNode("", true),
          createVNode(unref(NFormItem), {
            label: "Free U",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.free_u,
                "onUpdate:value": _cache[18] || (_cache[18] = ($event) => unref(settings).defaultSettings.api.free_u = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          unref(settings).defaultSettings.api.free_u ? (openBlock(), createElementBlock("div", _hoisted_8, [
            createVNode(unref(NFormItem), {
              label: "Free U S1",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NInputNumber), {
                  value: unref(settings).defaultSettings.api.free_u_s1,
                  "onUpdate:value": _cache[19] || (_cache[19] = ($event) => unref(settings).defaultSettings.api.free_u_s1 = $event),
                  step: 0.01
                }, null, 8, ["value"])
              ]),
              _: 1
            }),
            createVNode(unref(NFormItem), {
              label: "Free U S2",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NInputNumber), {
                  value: unref(settings).defaultSettings.api.free_u_s2,
                  "onUpdate:value": _cache[20] || (_cache[20] = ($event) => unref(settings).defaultSettings.api.free_u_s2 = $event),
                  step: 0.01
                }, null, 8, ["value"])
              ]),
              _: 1
            }),
            createVNode(unref(NFormItem), {
              label: "Free U B1",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NInputNumber), {
                  value: unref(settings).defaultSettings.api.free_u_b1,
                  "onUpdate:value": _cache[21] || (_cache[21] = ($event) => unref(settings).defaultSettings.api.free_u_b1 = $event),
                  step: 0.01
                }, null, 8, ["value"])
              ]),
              _: 1
            }),
            createVNode(unref(NFormItem), {
              label: "Free U B2",
              "label-placement": "left"
            }, {
              default: withCtx(() => [
                createVNode(unref(NInputNumber), {
                  value: unref(settings).defaultSettings.api.free_u_b2,
                  "onUpdate:value": _cache[22] || (_cache[22] = ($event) => unref(settings).defaultSettings.api.free_u_b2 = $event),
                  step: 0.01
                }, null, 8, ["value"])
              ]),
              _: 1
            })
          ])) : createCommentVNode("", true),
          createVNode(unref(NFormItem), {
            label: "Upcast VAE",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.upcast_vae,
                "onUpdate:value": _cache[23] || (_cache[23] = ($event) => unref(settings).defaultSettings.api.upcast_vae = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Apply unsharp mask",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSwitch), {
                value: unref(settings).defaultSettings.api.apply_unsharp_mask,
                "onUpdate:value": _cache[24] || (_cache[24] = ($event) => unref(settings).defaultSettings.api.apply_unsharp_mask = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "CFG Rescale Threshold",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSlider), {
                value: cfgRescaleValue.value,
                "onUpdate:value": _cache[25] || (_cache[25] = ($event) => cfgRescaleValue.value = $event),
                disabled: !enabledCfg.value,
                min: 2,
                max: 30,
                step: 0.5
              }, null, 8, ["value", "disabled"]),
              createVNode(unref(NSwitch), {
                value: enabledCfg.value,
                "onUpdate:value": _cache[26] || (_cache[26] = ($event) => enabledCfg.value = $event)
              }, null, 8, ["value"])
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "UISettings",
  setup(__props) {
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NForm), null, {
        default: withCtx(() => [
          createVNode(unref(NFormItem), {
            label: "Image Preview Interval (seconds)",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.api.live_preview_delay,
                "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).defaultSettings.api.live_preview_delay = $event),
                step: 0.1
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "Image Preview Method",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSelect), {
                value: unref(settings).defaultSettings.api.live_preview_method,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).defaultSettings.api.live_preview_method = $event),
                options: [
                  { value: "disabled", label: "Disabled" },
                  { value: "approximation", label: "Quick approximation (Default)" },
                  { value: "taesd", label: "Tiny VAE" }
                ]
              }, null, 8, ["value", "options"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "WebSocket Performance Monitor Interval",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.api.websocket_perf_interval,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).defaultSettings.api.websocket_perf_interval = $event),
                min: 0.1,
                step: 0.1
              }, null, 8, ["value"])
            ]),
            _: 1
          }),
          createVNode(unref(NFormItem), {
            label: "WebSocket Sync Interval",
            "label-placement": "left"
          }, {
            default: withCtx(() => [
              createVNode(unref(NInputNumber), {
                value: unref(settings).defaultSettings.api.websocket_sync_interval,
                "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).defaultSettings.api.websocket_sync_interval = $event),
                min: 1e-3,
                step: 0.01
              }, null, 8, ["value"])
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
const _hoisted_1 = { class: "main-container" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "SettingsView",
  setup(__props) {
    const message = useMessage();
    const settings = useSettings();
    const notification = useNotification();
    const saving = ref(false);
    function resetSettings() {
      Object.assign(
        settings.defaultSettings,
        JSON.parse(JSON.stringify(defaultSettings))
      );
      message.warning(
        "Settings were reset to default values, please save them if you want to keep them"
      );
    }
    function saveSettings() {
      saving.value = true;
      settings.saveSettings().then(() => {
        message.success("Settings saved");
      }).catch((e) => {
        message.error("Failed to save settings");
        notification.create({
          title: "Failed to save settings",
          content: e,
          type: "error"
        });
      }).finally(() => {
        saving.value = false;
      });
    }
    onUnmounted(() => {
      saveSettings();
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NCard), null, {
          default: withCtx(() => [
            createVNode(unref(NTabs), null, {
              suffix: withCtx(() => [
                createVNode(unref(NButton), {
                  type: "error",
                  ghost: "",
                  style: { "margin-right": "12px" },
                  onClick: resetSettings
                }, {
                  default: withCtx(() => [
                    createTextVNode("Reset Settings")
                  ]),
                  _: 1
                }),
                createVNode(unref(NButton), {
                  type: "success",
                  ghost: "",
                  onClick: saveSettings,
                  loading: saving.value
                }, {
                  default: withCtx(() => [
                    createTextVNode("Save Settings")
                  ]),
                  _: 1
                }, 8, ["loading"])
              ]),
              default: withCtx(() => [
                createVNode(unref(NTabPane), { name: "Autoload" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$a))
                  ]),
                  _: 1
                }),
                createVNode(unref(NTabPane), { name: "Files & Saving" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$8))
                  ]),
                  _: 1
                }),
                createVNode(unref(NTabPane), { name: "Optimizations" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$3))
                  ]),
                  _: 1
                }),
                createVNode(unref(NTabPane), { name: "Reproducibility & Generation" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$2))
                  ]),
                  _: 1
                }),
                createVNode(unref(NTabPane), { name: "Live preview & UI" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$1))
                  ]),
                  _: 1
                }),
                createVNode(unref(NTabPane), { name: "Defaults" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$6))
                  ]),
                  _: 1
                }),
                createVNode(unref(NTabPane), { name: "Bot" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$9))
                  ]),
                  _: 1
                }),
                createVNode(unref(NTabPane), { name: "General" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$5))
                  ]),
                  _: 1
                }),
                createVNode(unref(NTabPane), { name: "Flags" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$7))
                  ]),
                  _: 1
                }),
                createVNode(unref(NTabPane), { name: "Theme" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$b))
                  ]),
                  _: 1
                }),
                createVNode(unref(NTabPane), { name: "NSFW" }, {
                  default: withCtx(() => [
                    createVNode(unref(_sfc_main$4))
                  ]),
                  _: 1
                })
              ]),
              _: 1
            })
          ]),
          _: 1
        })
      ]);
    };
  }
});
export {
  _sfc_main as default
};
