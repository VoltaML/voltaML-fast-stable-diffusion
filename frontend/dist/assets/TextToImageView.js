import { d as defineComponent, u as useSettings, a as useState, o as openBlock, c as createBlock, w as withCtx, b as createBaseVNode, e as createVNode, f as unref, g as createElementBlock, h as createCommentVNode, N as NCard, i as computed, j as NSpace, k as createTextVNode, l as NTooltip, m as NSelect, n as useMessage, p as onUnmounted, q as NGi, r as NGrid, s as serverUrl } from "./index.js";
import { _ as _sfc_main$b } from "./GenerateSection.vue_vue_type_script_setup_true_lang.js";
import { _ as _sfc_main$c } from "./ImageOutput.vue_vue_type_script_setup_true_lang.js";
import { B as BurnerClock, P as Prompt, _ as _sfc_main$5, a as _sfc_main$6, b as _sfc_main$7, c as _sfc_main$8, d as _sfc_main$d } from "./clock.js";
import { _ as _sfc_main$4, a as _sfc_main$9, b as _sfc_main$a } from "./Upscale.vue_vue_type_script_setup_true_lang.js";
import { N as NSwitch } from "./Switch.js";
import { N as NSlider } from "./Slider.js";
import { N as NInputNumber } from "./InputNumber.js";
import { v as v4 } from "./v4.js";
import "./SendOutputTo.vue_vue_type_script_setup_true_lang.js";
import "./TrashBin.js";
import "./DescriptionsItem.js";
import "./Settings.js";
const _hoisted_1$2 = { class: "flex-container" };
const _hoisted_2$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Enabled", -1);
const _hoisted_3$2 = { key: 0 };
const _hoisted_4$2 = { class: "flex-container" };
const _hoisted_5$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1);
const _hoisted_6$2 = { class: "flex-container" };
const _hoisted_7$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1);
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "ResizeFromDimensionsInput",
  setup(__props) {
    const settings = useSettings();
    const global = useState();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), {
        title: "Resize from",
        class: "generate-extra-card"
      }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1$2, [
            _hoisted_2$2,
            createVNode(unref(NSwitch), {
              value: unref(global).state.txt2img.sdxl_resize,
              "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(global).state.txt2img.sdxl_resize = $event)
            }, null, 8, ["value"])
          ]),
          unref(global).state.txt2img.sdxl_resize ? (openBlock(), createElementBlock("div", _hoisted_3$2, [
            createBaseVNode("div", _hoisted_4$2, [
              _hoisted_5$2,
              createVNode(unref(NSlider), {
                value: unref(settings).data.settings.flags.sdxl.original_size.width,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).data.settings.flags.sdxl.original_size.width = $event),
                min: 128,
                max: 2048,
                step: 1,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: unref(settings).data.settings.flags.sdxl.original_size.width,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).data.settings.flags.sdxl.original_size.width = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                step: 1
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_6$2, [
              _hoisted_7$2,
              createVNode(unref(NSlider), {
                value: unref(settings).data.settings.flags.sdxl.original_size.height,
                "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).data.settings.flags.sdxl.original_size.height = $event),
                min: 128,
                max: 2048,
                step: 1,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: unref(settings).data.settings.flags.sdxl.original_size.height,
                "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).data.settings.flags.sdxl.original_size.height = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                step: 1
              }, null, 8, ["value"])
            ])
          ])) : createCommentVNode("", true)
        ]),
        _: 1
      });
    };
  }
});
const _hoisted_1$1 = { class: "flex-container" };
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_3$1 = { class: "flex-container" };
const _hoisted_4$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Refiner model", -1);
const _hoisted_5$1 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, " Generally, the refiner that came with your model is bound to generate the best results. ", -1);
const _hoisted_6$1 = { class: "flex-container" };
const _hoisted_7$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_8$1 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_9$1 = { class: "flex-container" };
const _hoisted_10 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Aesthetic Score", -1);
const _hoisted_11 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "Generally best to keep it around 6.", -1);
const _hoisted_12 = { class: "flex-container" };
const _hoisted_13 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Negative Aesthetic Score", -1);
const _hoisted_14 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "Generally best to keep it around 3.", -1);
const _hoisted_15 = { class: "flex-container" };
const _hoisted_16 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Strength", -1);
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "XLRefiner",
  setup(__props) {
    const settings = useSettings();
    const global = useState();
    const refinerModels = computed(() => {
      return global.state.models.filter((model) => model.type === "SDXL").map((model) => {
        return {
          label: model.name,
          value: model.name
        };
      });
    });
    async function onRefinerChange(modelStr) {
      settings.data.settings.flags.refiner.model = modelStr;
    }
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), {
        title: "SDXL Refiner",
        class: "generate-extra-card"
      }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1$1, [
            _hoisted_2$1,
            createVNode(unref(NSwitch), {
              value: unref(global).state.txt2img.refiner,
              "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(global).state.txt2img.refiner = $event)
            }, null, 8, ["value"])
          ]),
          unref(global).state.txt2img.refiner ? (openBlock(), createBlock(unref(NSpace), {
            key: 0,
            vertical: "",
            class: "left-container"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_3$1, [
                createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                  trigger: withCtx(() => [
                    _hoisted_4$1
                  ]),
                  default: withCtx(() => [
                    createTextVNode(" The SDXL-Refiner model to use for this step of diffusion. "),
                    _hoisted_5$1
                  ]),
                  _: 1
                }),
                createVNode(unref(NSelect), {
                  options: refinerModels.value,
                  placeholder: "None",
                  "onUpdate:value": onRefinerChange,
                  value: unref(settings).data.settings.flags.refiner.model !== null ? unref(settings).data.settings.flags.refiner.model : ""
                }, null, 8, ["options", "value"])
              ]),
              createBaseVNode("div", _hoisted_6$1, [
                createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                  trigger: withCtx(() => [
                    _hoisted_7$1
                  ]),
                  default: withCtx(() => [
                    createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                    _hoisted_8$1
                  ]),
                  _: 1
                }),
                createVNode(unref(NSlider), {
                  value: unref(settings).data.settings.flags.refiner.steps,
                  "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).data.settings.flags.refiner.steps = $event),
                  min: 5,
                  max: 300,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: unref(settings).data.settings.flags.refiner.steps,
                  "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).data.settings.flags.refiner.steps = $event),
                  size: "small",
                  style: { "min-width": "96px", "width": "96px" }
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_9$1, [
                createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                  trigger: withCtx(() => [
                    _hoisted_10
                  ]),
                  default: withCtx(() => [
                    createTextVNode(' Generally higher numbers will produce "more professional" images. '),
                    _hoisted_11
                  ]),
                  _: 1
                }),
                createVNode(unref(NSlider), {
                  value: unref(settings).data.settings.flags.refiner.aesthetic_score,
                  "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).data.settings.flags.refiner.aesthetic_score = $event),
                  min: 0,
                  max: 10,
                  step: 0.5,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: unref(settings).data.settings.flags.refiner.aesthetic_score,
                  "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).data.settings.flags.refiner.aesthetic_score = $event),
                  min: 0,
                  max: 10,
                  step: 0.25,
                  size: "small",
                  style: { "min-width": "96px", "width": "96px" }
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_12, [
                createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                  trigger: withCtx(() => [
                    _hoisted_13
                  ]),
                  default: withCtx(() => [
                    createTextVNode(" Makes sense to keep this lower than aesthetic score. "),
                    _hoisted_14
                  ]),
                  _: 1
                }),
                createVNode(unref(NSlider), {
                  value: unref(settings).data.settings.flags.refiner.negative_aesthetic_score,
                  "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(settings).data.settings.flags.refiner.negative_aesthetic_score = $event),
                  min: 0,
                  max: 10,
                  step: 0.5,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: unref(settings).data.settings.flags.refiner.negative_aesthetic_score,
                  "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(settings).data.settings.flags.refiner.negative_aesthetic_score = $event),
                  min: 0,
                  max: 10,
                  step: 0.25,
                  size: "small",
                  style: { "min-width": "96px", "width": "96px" }
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_15, [
                _hoisted_16,
                createVNode(unref(NSlider), {
                  value: unref(settings).data.settings.flags.refiner.strength,
                  "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(settings).data.settings.flags.refiner.strength = $event),
                  min: 0.1,
                  max: 0.9,
                  step: 0.05,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: unref(settings).data.settings.flags.refiner.strength,
                  "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(settings).data.settings.flags.refiner.strength = $event),
                  size: "small",
                  style: { "min-width": "96px", "width": "96px" },
                  min: 0.1,
                  max: 0.9,
                  step: 0.05
                }, null, 8, ["value"])
              ])
            ]),
            _: 1
          })) : createCommentVNode("", true)
        ]),
        _: 1
      });
    };
  }
});
const _hoisted_1 = { class: "main-container" };
const _hoisted_2 = { class: "flex-container" };
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_4 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_5 = { class: "flex-container" };
const _hoisted_6 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Count", -1);
const _hoisted_7 = { class: "flex-container" };
const _hoisted_8 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "75px" } }, "Seed", -1);
const _hoisted_9 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "For random seed use -1.", -1);
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "Txt2Img",
  setup(__props) {
    const global = useState();
    const settings = useSettings();
    const messageHandler = useMessage();
    const isSelectedModelSDXL = computed(() => {
      var _a;
      return ((_a = settings.data.settings.model) == null ? void 0 : _a.type) === "SDXL";
    });
    const checkSeed = (seed) => {
      if (seed === -1) {
        seed = Math.floor(Math.random() * 999999999999);
      }
      return seed;
    };
    const generate = () => {
      var _a, _b;
      if (settings.data.settings.txt2img.seed === null) {
        messageHandler.error("Please set a seed");
        return;
      }
      global.state.generating = true;
      const seed = checkSeed(settings.data.settings.txt2img.seed);
      fetch(`${serverUrl}/api/generate/txt2img`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          data: {
            id: v4(),
            prompt: settings.data.settings.txt2img.prompt,
            negative_prompt: settings.data.settings.txt2img.negative_prompt,
            width: settings.data.settings.txt2img.width,
            height: settings.data.settings.txt2img.height,
            steps: settings.data.settings.txt2img.steps,
            guidance_scale: settings.data.settings.txt2img.cfg_scale,
            seed,
            batch_size: settings.data.settings.txt2img.batch_size,
            batch_count: settings.data.settings.txt2img.batch_count,
            scheduler: settings.data.settings.txt2img.sampler,
            self_attention_scale: settings.data.settings.txt2img.self_attention_scale,
            sigmas: settings.data.settings.txt2img.sigmas,
            sampler_settings: settings.data.settings.sampler_config[settings.data.settings.txt2img.sampler],
            prompt_to_prompt_settings: {
              prompt_to_prompt_model: settings.data.settings.api.prompt_to_prompt_model,
              prompt_to_prompt_model_settings: settings.data.settings.api.prompt_to_prompt_device,
              prompt_to_prompt: settings.data.settings.api.prompt_to_prompt
            }
          },
          model: (_a = settings.data.settings.model) == null ? void 0 : _a.path,
          backend: "PyTorch",
          autoload: false,
          flags: {
            ...isSelectedModelSDXL.value && global.state.txt2img.sdxl_resize ? {
              sdxl: {
                original_size: {
                  width: settings.data.settings.flags.sdxl.original_size.width,
                  height: settings.data.settings.flags.sdxl.original_size.height
                }
              }
            } : {},
            ...settings.data.settings.txt2img.highres.enabled ? {
              highres_fix: {
                mode: settings.data.settings.txt2img.highres.mode,
                image_upscaler: settings.data.settings.txt2img.highres.image_upscaler,
                scale: settings.data.settings.txt2img.highres.scale,
                latent_scale_mode: settings.data.settings.txt2img.highres.latent_scale_mode,
                strength: settings.data.settings.txt2img.highres.strength,
                steps: settings.data.settings.txt2img.highres.steps,
                antialiased: settings.data.settings.txt2img.highres.antialiased
              }
            } : global.state.txt2img.refiner ? {
              refiner: {
                model: settings.data.settings.flags.refiner.model,
                aesthetic_score: settings.data.settings.flags.refiner.aesthetic_score,
                negative_aesthetic_score: settings.data.settings.flags.refiner.negative_aesthetic_score,
                steps: settings.data.settings.flags.refiner.steps,
                strength: settings.data.settings.flags.refiner.strength
              }
            } : {},
            ...settings.data.settings.txt2img.deepshrink.enabled ? {
              deepshrink: {
                early_out: settings.data.settings.txt2img.deepshrink.early_out,
                depth_1: settings.data.settings.txt2img.deepshrink.depth_1,
                stop_at_1: settings.data.settings.txt2img.deepshrink.stop_at_1,
                depth_2: settings.data.settings.txt2img.deepshrink.depth_2,
                stop_at_2: settings.data.settings.txt2img.deepshrink.stop_at_2,
                scaler: settings.data.settings.txt2img.deepshrink.scaler,
                base_scale: settings.data.settings.txt2img.deepshrink.base_scale
              }
            } : {},
            ...settings.data.settings.txt2img.scalecrafter.enabled ? {
              scalecrafter: {
                unsafe_resolutions: settings.data.settings.txt2img.scalecrafter.unsafe_resolutions,
                base: (_b = settings.data.settings.model) == null ? void 0 : _b.type,
                disperse: settings.data.settings.txt2img.scalecrafter.disperse
              }
            } : {},
            ...settings.data.settings.txt2img.upscale.enabled ? {
              upscale: {
                upscale_factor: settings.data.settings.txt2img.upscale.upscale_factor,
                tile_size: settings.data.settings.txt2img.upscale.tile_size,
                tile_padding: settings.data.settings.txt2img.upscale.tile_padding,
                model: settings.data.settings.txt2img.upscale.model
              }
            } : {}
          }
        })
      }).then((res) => {
        if (!res.ok) {
          throw new Error(res.statusText);
        }
        global.state.generating = false;
        res.json().then((data) => {
          global.state.txt2img.images = data.images;
          global.state.txt2img.currentImage = data.images[0];
          global.state.progress = 0;
          global.state.total_steps = 0;
          global.state.current_step = 0;
          global.state.txt2img.genData = {
            time_taken: parseFloat(parseFloat(data.time).toFixed(4)),
            seed
          };
        });
      }).catch((err) => {
        global.state.generating = false;
        messageHandler.error(err);
      });
    };
    const burner = new BurnerClock(
      settings.data.settings.txt2img,
      settings,
      generate
    );
    onUnmounted(() => {
      burner.cleanup();
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NGrid), {
          cols: "1 m:2",
          "x-gap": "12",
          responsive: "screen"
        }, {
          default: withCtx(() => [
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(unref(NCard), { title: "Settings" }, {
                  default: withCtx(() => [
                    createVNode(unref(NSpace), {
                      vertical: "",
                      class: "left-container"
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(Prompt), { tab: "txt2img" }),
                        createVNode(unref(_sfc_main$4), { type: "txt2img" }),
                        createVNode(unref(_sfc_main$5), {
                          "dimensions-object": unref(settings).data.settings.txt2img
                        }, null, 8, ["dimensions-object"]),
                        createBaseVNode("div", _hoisted_2, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_3
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                              _hoisted_4
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(settings).data.settings.txt2img.steps,
                            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(settings).data.settings.txt2img.steps = $event),
                            min: 5,
                            max: 300,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(settings).data.settings.txt2img.steps,
                            "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).data.settings.txt2img.steps = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" }
                          }, null, 8, ["value"])
                        ]),
                        createVNode(unref(_sfc_main$6), { tab: "txt2img" }),
                        createVNode(unref(_sfc_main$7), { tab: "txt2img" }),
                        createBaseVNode("div", _hoisted_5, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_6
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of images to generate after each other. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(settings).data.settings.txt2img.batch_count,
                            "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).data.settings.txt2img.batch_count = $event),
                            min: 1,
                            max: 9,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(settings).data.settings.txt2img.batch_count,
                            "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).data.settings.txt2img.batch_count = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" }
                          }, null, 8, ["value"])
                        ]),
                        createVNode(unref(_sfc_main$8), {
                          "batch-size-object": unref(settings).data.settings.txt2img
                        }, null, 8, ["batch-size-object"]),
                        createBaseVNode("div", _hoisted_7, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_8
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Seed is a number that represents the starting canvas of your image. If you want to create the same image as your friend, you can use the same settings and seed to do so. "),
                              _hoisted_9
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NInputNumber), {
                            value: unref(settings).data.settings.txt2img.seed,
                            "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).data.settings.txt2img.seed = $event),
                            size: "small",
                            style: { "flex-grow": "1" }
                          }, null, 8, ["value"])
                        ])
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                }),
                isSelectedModelSDXL.value ? (openBlock(), createBlock(unref(_sfc_main$3), {
                  key: 0,
                  "dimensions-object": unref(settings).data.settings.txt2img
                }, null, 8, ["dimensions-object"])) : createCommentVNode("", true),
                isSelectedModelSDXL.value ? (openBlock(), createBlock(unref(_sfc_main$2), { key: 1 })) : createCommentVNode("", true),
                createVNode(unref(_sfc_main$9), { tab: "txt2img" }),
                createVNode(unref(_sfc_main$a), { tab: "txt2img" })
              ]),
              _: 1
            }),
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(unref(_sfc_main$b), { generate }),
                createVNode(unref(_sfc_main$c), {
                  "current-image": unref(global).state.txt2img.currentImage,
                  images: unref(global).state.txt2img.images,
                  data: unref(settings).data.settings.txt2img,
                  onImageClicked: _cache[5] || (_cache[5] = ($event) => unref(global).state.txt2img.currentImage = $event)
                }, null, 8, ["current-image", "images", "data"]),
                createVNode(unref(_sfc_main$d), {
                  style: { "margin-top": "12px" },
                  "gen-data": unref(global).state.txt2img.genData
                }, null, 8, ["gen-data"])
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
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TextToImageView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(_sfc_main$1));
    };
  }
});
export {
  _sfc_main as default
};
