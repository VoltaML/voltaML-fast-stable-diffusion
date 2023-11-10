import { d as defineComponent, u as useSettings, a as useState, c as computed, b as upscalerOptions, o as openBlock, e as createBlock, w as withCtx, f as createBaseVNode, g as createVNode, h as unref, N as NSpace, i as NSelect, j as createElementBlock, k as createTextVNode, l as NTooltip, m as createCommentVNode, n as NCard, p as useMessage, q as onUnmounted, r as resolveComponent, s as NGi, t as NGrid, v as serverUrl } from "./index.js";
import { _ as _sfc_main$6 } from "./GenerateSection.vue_vue_type_script_setup_true_lang.js";
import { _ as _sfc_main$7 } from "./ImageOutput.vue_vue_type_script_setup_true_lang.js";
import { B as BurnerClock, P as Prompt, _ as _sfc_main$4, a as _sfc_main$5, b as _sfc_main$8 } from "./clock.js";
import { N as NSwitch, a as NSlider } from "./Switch.js";
import { N as NInputNumber } from "./InputNumber.js";
import { _ as _sfc_main$3 } from "./SamplerPicker.vue_vue_type_script_setup_true_lang.js";
import { v as v4 } from "./v4.js";
import "./SendOutputTo.vue_vue_type_script_setup_true_lang.js";
import "./TrashBin.js";
import "./DescriptionsItem.js";
import "./Settings.js";
const _hoisted_1$1 = { class: "flex-container" };
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_3$1 = { class: "flex-container" };
const _hoisted_4$1 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Mode")
], -1);
const _hoisted_5$1 = { key: 0 };
const _hoisted_6$1 = { class: "flex-container" };
const _hoisted_7$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Upscaler", -1);
const _hoisted_8$1 = { key: 1 };
const _hoisted_9$1 = { class: "flex-container" };
const _hoisted_10$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Antialiased", -1);
const _hoisted_11$1 = { class: "flex-container" };
const _hoisted_12$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Latent Mode", -1);
const _hoisted_13$1 = { class: "flex-container" };
const _hoisted_14$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_15$1 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_16$1 = { class: "flex-container" };
const _hoisted_17$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Scale", -1);
const _hoisted_18$1 = { class: "flex-container" };
const _hoisted_19$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Strength", -1);
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "HighResFix",
  setup(__props) {
    const settings = useSettings();
    const global = useState();
    const imageUpscalerOptions = computed(() => {
      const localModels = global.state.models.filter(
        (model) => model.backend === "Upscaler" && !(upscalerOptions.map((option) => option.label).indexOf(model.name) !== -1)
      ).map((model) => ({
        label: model.name,
        value: model.path
      }));
      return [...upscalerOptions, ...localModels];
    });
    const latentUpscalerOptions = [
      { label: "Nearest", value: "nearest" },
      { label: "Nearest exact", value: "nearest-exact" },
      { label: "Area", value: "area" },
      { label: "Bilinear", value: "bilinear" },
      { label: "Bicubic", value: "bicubic" },
      {
        label: "Bislerp (Original, slow)",
        value: "bislerp-original"
      },
      {
        label: "Bislerp (Tortured, fast)",
        value: "bislerp-tortured"
      }
    ];
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), { title: "Highres fix" }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1$1, [
            _hoisted_2$1,
            createVNode(unref(NSwitch), {
              value: unref(global).state.txt2img.highres,
              "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(global).state.txt2img.highres = $event)
            }, null, 8, ["value"])
          ]),
          unref(global).state.txt2img.highres ? (openBlock(), createBlock(unref(NSpace), {
            key: 0,
            vertical: "",
            class: "left-container"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_3$1, [
                _hoisted_4$1,
                createVNode(unref(NSelect), {
                  value: unref(settings).data.settings.flags.highres.mode,
                  "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).data.settings.flags.highres.mode = $event),
                  options: [
                    { label: "Latent", value: "latent" },
                    { label: "Image", value: "image" }
                  ]
                }, null, 8, ["value"])
              ]),
              unref(settings).data.settings.flags.highres.mode === "image" ? (openBlock(), createElementBlock("div", _hoisted_5$1, [
                createBaseVNode("div", _hoisted_6$1, [
                  _hoisted_7$1,
                  createVNode(unref(NSelect), {
                    value: unref(settings).data.settings.flags.highres.image_upscaler,
                    "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).data.settings.flags.highres.image_upscaler = $event),
                    size: "small",
                    style: { "flex-grow": "1" },
                    filterable: "",
                    options: imageUpscalerOptions.value
                  }, null, 8, ["value", "options"])
                ])
              ])) : (openBlock(), createElementBlock("div", _hoisted_8$1, [
                createBaseVNode("div", _hoisted_9$1, [
                  _hoisted_10$1,
                  createVNode(unref(NSwitch), {
                    value: unref(settings).data.settings.flags.highres.antialiased,
                    "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).data.settings.flags.highres.antialiased = $event)
                  }, null, 8, ["value"])
                ]),
                createBaseVNode("div", _hoisted_11$1, [
                  _hoisted_12$1,
                  createVNode(unref(NSelect), {
                    value: unref(settings).data.settings.flags.highres.latent_scale_mode,
                    "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).data.settings.flags.highres.latent_scale_mode = $event),
                    size: "small",
                    style: { "flex-grow": "1" },
                    filterable: "",
                    options: latentUpscalerOptions
                  }, null, 8, ["value"])
                ])
              ])),
              createBaseVNode("div", _hoisted_13$1, [
                createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                  trigger: withCtx(() => [
                    _hoisted_14$1
                  ]),
                  default: withCtx(() => [
                    createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                    _hoisted_15$1
                  ]),
                  _: 1
                }),
                createVNode(unref(NSlider), {
                  value: unref(settings).data.settings.flags.highres.steps,
                  "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(settings).data.settings.flags.highres.steps = $event),
                  min: 5,
                  max: 300,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: unref(settings).data.settings.flags.highres.steps,
                  "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(settings).data.settings.flags.highres.steps = $event),
                  size: "small",
                  style: { "min-width": "96px", "width": "96px" }
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_16$1, [
                _hoisted_17$1,
                createVNode(unref(NSlider), {
                  value: unref(settings).data.settings.flags.highres.scale,
                  "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(settings).data.settings.flags.highres.scale = $event),
                  min: 1,
                  max: 8,
                  step: 0.1,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: unref(settings).data.settings.flags.highres.scale,
                  "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(settings).data.settings.flags.highres.scale = $event),
                  size: "small",
                  style: { "min-width": "96px", "width": "96px" },
                  step: 0.1
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_18$1, [
                _hoisted_19$1,
                createVNode(unref(NSlider), {
                  value: unref(settings).data.settings.flags.highres.strength,
                  "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(settings).data.settings.flags.highres.strength = $event),
                  min: 0.1,
                  max: 0.9,
                  step: 0.05,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: unref(settings).data.settings.flags.highres.strength,
                  "onUpdate:value": _cache[10] || (_cache[10] = ($event) => unref(settings).data.settings.flags.highres.strength = $event),
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
const _hoisted_6 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "CFG Scale", -1);
const _hoisted_7 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 3-15 for most images.", -1);
const _hoisted_8 = {
  key: 0,
  class: "flex-container"
};
const _hoisted_9 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Self Attention Scale", -1);
const _hoisted_10 = { class: "flex-container" };
const _hoisted_11 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Count", -1);
const _hoisted_12 = { class: "flex-container" };
const _hoisted_13 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "75px" } }, "Seed", -1);
const _hoisted_14 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "For random seed use -1.", -1);
const _hoisted_15 = { class: "flex-container" };
const _hoisted_16 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_17 = { class: "flex-container" };
const _hoisted_18 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Refiner model", -1);
const _hoisted_19 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, " Generally, the refiner that came with your model is bound to generate the best results. ", -1);
const _hoisted_20 = { class: "flex-container" };
const _hoisted_21 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_22 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_23 = { class: "flex-container" };
const _hoisted_24 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Aesthetic Score", -1);
const _hoisted_25 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "Generally best to keep it around 6.", -1);
const _hoisted_26 = { class: "flex-container" };
const _hoisted_27 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Negative Aesthetic Score", -1);
const _hoisted_28 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "Generally best to keep it around 3.", -1);
const _hoisted_29 = { class: "flex-container" };
const _hoisted_30 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Strength", -1);
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
    const refinerModels = computed(() => {
      return global.state.models.filter(
        (model) => model.type === "SDXL" && model.name.toLowerCase().includes("refiner")
      ).map((model) => {
        return {
          label: model.name,
          value: model.name
        };
      });
    });
    async function onRefinerChange(modelStr) {
      settings.data.settings.flags.refiner.model = modelStr;
    }
    const checkSeed = (seed) => {
      if (seed === -1) {
        seed = Math.floor(Math.random() * 999999999999);
      }
      return seed;
    };
    const generate = () => {
      var _a;
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
          model: (_a = settings.data.settings.model) == null ? void 0 : _a.name,
          backend: "PyTorch",
          autoload: false,
          flags: {
            ...isSelectedModelSDXL.value ? {
              sdxl: {
                original_size: [1024, 1024]
              }
            } : {},
            ...global.state.txt2img.highres ? {
              highres_fix: {
                mode: settings.data.settings.flags.highres.mode,
                image_upscaler: settings.data.settings.flags.highres.image_upscaler,
                scale: settings.data.settings.flags.highres.scale,
                latent_scale_mode: settings.data.settings.flags.highres.latent_scale_mode,
                strength: settings.data.settings.flags.highres.strength,
                steps: settings.data.settings.flags.highres.steps,
                antialiased: settings.data.settings.flags.highres.antialiased
              }
            } : global.state.txt2img.refiner ? {
              refiner: {
                model: settings.data.settings.flags.refiner.model,
                aesthetic_score: settings.data.settings.flags.refiner.aesthetic_score,
                negative_aesthetic_score: settings.data.settings.flags.refiner.negative_aesthetic_score,
                steps: settings.data.settings.flags.refiner.steps,
                strength: settings.data.settings.flags.refiner.strength
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
      const _component_NSwitch = resolveComponent("NSwitch");
      const _component_NSelect = resolveComponent("NSelect");
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
                      default: withCtx(() => {
                        var _a;
                        return [
                          createVNode(unref(Prompt), { tab: "txt2img" }),
                          createVNode(unref(_sfc_main$3), { type: "txt2img" }),
                          createVNode(unref(_sfc_main$4), {
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
                          createBaseVNode("div", _hoisted_5, [
                            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                              trigger: withCtx(() => [
                                _hoisted_6
                              ]),
                              default: withCtx(() => [
                                createTextVNode(' Guidance scale indicates how much should model stay close to the prompt. Higher values might be exactly what you want, but generated images might have some artefacts. Lower values indicates that model can "dream" about this prompt more. '),
                                _hoisted_7
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSlider), {
                              value: unref(settings).data.settings.txt2img.cfg_scale,
                              "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).data.settings.txt2img.cfg_scale = $event),
                              min: 1,
                              max: 30,
                              step: 0.5,
                              style: { "margin-right": "12px" }
                            }, null, 8, ["value"]),
                            createVNode(unref(NInputNumber), {
                              value: unref(settings).data.settings.txt2img.cfg_scale,
                              "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(settings).data.settings.txt2img.cfg_scale = $event),
                              size: "small",
                              style: { "min-width": "96px", "width": "96px" },
                              step: 0.5
                            }, null, 8, ["value"])
                          ]),
                          Number.isInteger(unref(settings).data.settings.txt2img.sampler) && ((_a = unref(settings).data.settings.model) == null ? void 0 : _a.backend) === "PyTorch" ? (openBlock(), createElementBlock("div", _hoisted_8, [
                            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                              trigger: withCtx(() => [
                                _hoisted_9
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" If self attention is >0, SAG will guide the model and improve the quality of the image at the cost of speed. Higher values will follow the guidance more closely, which can lead to better, more sharp and detailed outputs. ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSlider), {
                              value: unref(settings).data.settings.txt2img.self_attention_scale,
                              "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(settings).data.settings.txt2img.self_attention_scale = $event),
                              min: 0,
                              max: 1,
                              step: 0.05,
                              style: { "margin-right": "12px" }
                            }, null, 8, ["value"]),
                            createVNode(unref(NInputNumber), {
                              value: unref(settings).data.settings.txt2img.self_attention_scale,
                              "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(settings).data.settings.txt2img.self_attention_scale = $event),
                              size: "small",
                              style: { "min-width": "96px", "width": "96px" },
                              step: 0.05
                            }, null, 8, ["value"])
                          ])) : createCommentVNode("", true),
                          createBaseVNode("div", _hoisted_10, [
                            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                              trigger: withCtx(() => [
                                _hoisted_11
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" Number of images to generate after each other. ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSlider), {
                              value: unref(settings).data.settings.txt2img.batch_count,
                              "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(settings).data.settings.txt2img.batch_count = $event),
                              min: 1,
                              max: 9,
                              style: { "margin-right": "12px" }
                            }, null, 8, ["value"]),
                            createVNode(unref(NInputNumber), {
                              value: unref(settings).data.settings.txt2img.batch_count,
                              "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(settings).data.settings.txt2img.batch_count = $event),
                              size: "small",
                              style: { "min-width": "96px", "width": "96px" }
                            }, null, 8, ["value"])
                          ]),
                          createVNode(unref(_sfc_main$5), {
                            "batch-size-object": unref(settings).data.settings.txt2img
                          }, null, 8, ["batch-size-object"]),
                          createBaseVNode("div", _hoisted_12, [
                            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                              trigger: withCtx(() => [
                                _hoisted_13
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" Seed is a number that represents the starting canvas of your image. If you want to create the same image as your friend, you can use the same settings and seed to do so. "),
                                _hoisted_14
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NInputNumber), {
                              value: unref(settings).data.settings.txt2img.seed,
                              "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(settings).data.settings.txt2img.seed = $event),
                              size: "small",
                              style: { "flex-grow": "1" }
                            }, null, 8, ["value"])
                          ])
                        ];
                      }),
                      _: 1
                    })
                  ]),
                  _: 1
                }),
                isSelectedModelSDXL.value ? (openBlock(), createBlock(unref(NCard), {
                  key: 0,
                  title: "SDXL Refiner",
                  style: { "margin-top": "12px", "margin-bottom": "12px" }
                }, {
                  default: withCtx(() => [
                    createBaseVNode("div", _hoisted_15, [
                      _hoisted_16,
                      createVNode(_component_NSwitch, {
                        value: unref(global).state.txt2img.refiner,
                        "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(global).state.txt2img.refiner = $event)
                      }, null, 8, ["value"])
                    ]),
                    unref(global).state.txt2img.refiner ? (openBlock(), createBlock(unref(NSpace), {
                      key: 0,
                      vertical: "",
                      class: "left-container"
                    }, {
                      default: withCtx(() => [
                        createBaseVNode("div", _hoisted_17, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_18
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" The SDXL-Refiner model to use for this step of diffusion. "),
                              _hoisted_19
                            ]),
                            _: 1
                          }),
                          createVNode(_component_NSelect, {
                            options: refinerModels.value,
                            placeholder: "None",
                            "onUpdate:value": onRefinerChange,
                            value: unref(settings).data.settings.flags.refiner.model !== null ? unref(settings).data.settings.flags.refiner.model : ""
                          }, null, 8, ["options", "value"])
                        ]),
                        createBaseVNode("div", _hoisted_20, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_21
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                              _hoisted_22
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(settings).data.settings.flags.refiner.steps,
                            "onUpdate:value": _cache[10] || (_cache[10] = ($event) => unref(settings).data.settings.flags.refiner.steps = $event),
                            min: 5,
                            max: 300,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(settings).data.settings.flags.refiner.steps,
                            "onUpdate:value": _cache[11] || (_cache[11] = ($event) => unref(settings).data.settings.flags.refiner.steps = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" }
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_23, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_24
                            ]),
                            default: withCtx(() => [
                              createTextVNode(' Generally higher numbers will produce "more professional" images. '),
                              _hoisted_25
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(settings).data.settings.flags.refiner.aesthetic_score,
                            "onUpdate:value": _cache[12] || (_cache[12] = ($event) => unref(settings).data.settings.flags.refiner.aesthetic_score = $event),
                            min: 0,
                            max: 10,
                            step: 0.5,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(settings).data.settings.flags.refiner.aesthetic_score,
                            "onUpdate:value": _cache[13] || (_cache[13] = ($event) => unref(settings).data.settings.flags.refiner.aesthetic_score = $event),
                            min: 0,
                            max: 10,
                            step: 0.25,
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" }
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_26, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_27
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Makes sense to keep this lower than aesthetic score. "),
                              _hoisted_28
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(settings).data.settings.flags.refiner.negative_aesthetic_score,
                            "onUpdate:value": _cache[14] || (_cache[14] = ($event) => unref(settings).data.settings.flags.refiner.negative_aesthetic_score = $event),
                            min: 0,
                            max: 10,
                            step: 0.5,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(settings).data.settings.flags.refiner.negative_aesthetic_score,
                            "onUpdate:value": _cache[15] || (_cache[15] = ($event) => unref(settings).data.settings.flags.refiner.negative_aesthetic_score = $event),
                            min: 0,
                            max: 10,
                            step: 0.25,
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" }
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_29, [
                          _hoisted_30,
                          createVNode(unref(NSlider), {
                            value: unref(settings).data.settings.flags.refiner.strength,
                            "onUpdate:value": _cache[16] || (_cache[16] = ($event) => unref(settings).data.settings.flags.refiner.strength = $event),
                            min: 0.1,
                            max: 0.9,
                            step: 0.05,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(settings).data.settings.flags.refiner.strength,
                            "onUpdate:value": _cache[17] || (_cache[17] = ($event) => unref(settings).data.settings.flags.refiner.strength = $event),
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
                })) : createCommentVNode("", true),
                createVNode(unref(_sfc_main$2), { style: { "margin-top": "12px", "margin-bottom": "12px" } })
              ]),
              _: 1
            }),
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(unref(_sfc_main$6), { generate }),
                createVNode(unref(_sfc_main$7), {
                  "current-image": unref(global).state.txt2img.currentImage,
                  images: unref(global).state.txt2img.images,
                  data: unref(settings).data.settings.txt2img,
                  onImageClicked: _cache[18] || (_cache[18] = ($event) => unref(global).state.txt2img.currentImage = $event)
                }, null, 8, ["current-image", "images", "data"]),
                createVNode(unref(_sfc_main$8), {
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
