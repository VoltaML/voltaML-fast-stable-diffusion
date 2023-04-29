import { _ as _sfc_main$1 } from "./GenerateSection.vue_vue_type_script_setup_true_lang.js";
import { _ as _sfc_main$2 } from "./ImageOutput.vue_vue_type_script_setup_true_lang.js";
import { _ as _sfc_main$4 } from "./OutputStats.vue_vue_type_script_setup_true_lang.js";
import { _ as _sfc_main$3 } from "./SendOutputTo.vue_vue_type_script_setup_true_lang.js";
import { d as defineComponent, u as useState, a as useSettings, b as useMessage, c as computed, o as openBlock, e as createElementBlock, f as createVNode, w as withCtx, g as unref, N as NGi, h as NCard, i as NSpace, j as NInput, k as createTextVNode, t as toDisplayString, l as createBaseVNode, m as NTooltip, n as NSelect, p as createBlock, q as createCommentVNode, r as NGrid, s as spaceRegex, v as serverUrl } from "./index.js";
import { N as NSlider } from "./Slider.js";
import { N as NInputNumber } from "./InputNumber.js";
import { N as NSwitch } from "./Switch.js";
import { v as v4 } from "./v4.js";
import "./Image.js";
const _hoisted_1 = { class: "main-container" };
const _hoisted_2 = { class: "flex-container" };
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "100px" } }, "Sampler", -1);
const _hoisted_4 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using Euler A for the best results (but it also takes more time). ", -1);
const _hoisted_5 = /* @__PURE__ */ createBaseVNode("a", {
  target: "_blank",
  href: "https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
}, "Learn more", -1);
const _hoisted_6 = {
  key: 0,
  class: "flex-container"
};
const _hoisted_7 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1);
const _hoisted_8 = {
  key: 1,
  class: "flex-container"
};
const _hoisted_9 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1);
const _hoisted_10 = {
  key: 2,
  class: "flex-container"
};
const _hoisted_11 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1);
const _hoisted_12 = {
  key: 3,
  class: "flex-container"
};
const _hoisted_13 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1);
const _hoisted_14 = { class: "flex-container" };
const _hoisted_15 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_16 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_17 = { class: "flex-container" };
const _hoisted_18 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "CFG Scale", -1);
const _hoisted_19 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 3-15 for most images.", -1);
const _hoisted_20 = { class: "flex-container" };
const _hoisted_21 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Count", -1);
const _hoisted_22 = {
  key: 4,
  class: "flex-container"
};
const _hoisted_23 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1);
const _hoisted_24 = {
  key: 5,
  class: "flex-container"
};
const _hoisted_25 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1);
const _hoisted_26 = { class: "flex-container" };
const _hoisted_27 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "75px" } }, "Seed", -1);
const _hoisted_28 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "For random seed use -1.", -1);
const _hoisted_29 = { class: "flex-container" };
const _hoisted_30 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_31 = { class: "flex-container" };
const _hoisted_32 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_33 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_34 = { class: "flex-container" };
const _hoisted_35 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Scale", -1);
const _hoisted_36 = { class: "flex-container" };
const _hoisted_37 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Strength", -1);
const _hoisted_38 = { class: "flex-container" };
const _hoisted_39 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Antialiased", -1);
const _hoisted_40 = { class: "flex-container" };
const _hoisted_41 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Latent Mode", -1);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TextToImageView",
  setup(__props) {
    const global = useState();
    const conf = useSettings();
    const messageHandler = useMessage();
    const promptCount = computed(() => {
      return conf.data.settings.txt2img.prompt.split(spaceRegex).length - 1;
    });
    const negativePromptCount = computed(() => {
      return conf.data.settings.txt2img.negative_prompt.split(spaceRegex).length - 1;
    });
    const checkSeed = (seed) => {
      if (seed === -1) {
        seed = Math.floor(Math.random() * 999999999999);
      }
      return seed;
    };
    const generate = () => {
      var _a;
      if (conf.data.settings.txt2img.seed === null) {
        messageHandler.error("Please set a seed");
        return;
      }
      global.state.generating = true;
      const seed = checkSeed(conf.data.settings.txt2img.seed);
      fetch(`${serverUrl}/api/generate/txt2img`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          data: {
            id: v4(),
            prompt: conf.data.settings.txt2img.prompt,
            negative_prompt: conf.data.settings.txt2img.negative_prompt,
            width: conf.data.settings.aitDim.width ? conf.data.settings.aitDim.width : conf.data.settings.txt2img.width,
            height: conf.data.settings.aitDim.height ? conf.data.settings.aitDim.height : conf.data.settings.txt2img.height,
            steps: conf.data.settings.txt2img.steps,
            guidance_scale: conf.data.settings.txt2img.cfg_scale,
            seed,
            batch_size: conf.data.settings.aitDim.batch_size ? conf.data.settings.aitDim.batch_size : conf.data.settings.txt2img.batch_size,
            batch_count: conf.data.settings.txt2img.batch_count,
            scheduler: conf.data.settings.txt2img.sampler
          },
          model: (_a = conf.data.settings.model) == null ? void 0 : _a.name,
          backend: "PyTorch",
          autoload: false,
          flags: global.state.txt2img.highres ? {
            highres_fix: {
              scale: conf.data.settings.extra.highres.scale,
              latent_scale_mode: conf.data.settings.extra.highres.latent_scale_mode,
              strength: conf.data.settings.extra.highres.strength,
              steps: conf.data.settings.extra.highres.steps,
              antialiased: conf.data.settings.extra.highres.antialiased
            }
          } : {}
        })
      }).then((res) => {
        if (!res.ok) {
          throw new Error(res.statusText);
        }
        global.state.generating = false;
        res.json().then((data) => {
          global.state.txt2img.images = data.images;
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
        console.log(err);
      });
    };
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
                        createVNode(unref(NInput), {
                          value: unref(conf).data.settings.txt2img.prompt,
                          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(conf).data.settings.txt2img.prompt = $event),
                          type: "textarea",
                          placeholder: "Prompt",
                          "show-count": ""
                        }, {
                          count: withCtx(() => [
                            createTextVNode(toDisplayString(unref(promptCount)), 1)
                          ]),
                          _: 1
                        }, 8, ["value"]),
                        createVNode(unref(NInput), {
                          value: unref(conf).data.settings.txt2img.negative_prompt,
                          "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(conf).data.settings.txt2img.negative_prompt = $event),
                          type: "textarea",
                          placeholder: "Negative prompt",
                          "show-count": ""
                        }, {
                          count: withCtx(() => [
                            createTextVNode(toDisplayString(unref(negativePromptCount)), 1)
                          ]),
                          _: 1
                        }, 8, ["value"]),
                        createBaseVNode("div", _hoisted_2, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_3
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" The sampler is the method used to generate the image. Your result may vary drastically depending on the sampler you choose. "),
                              _hoisted_4,
                              _hoisted_5
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSelect), {
                            options: unref(conf).scheduler_options,
                            value: unref(conf).data.settings.txt2img.sampler,
                            "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(conf).data.settings.txt2img.sampler = $event),
                            style: { "flex-grow": "1" }
                          }, null, 8, ["options", "value"])
                        ]),
                        unref(conf).data.settings.aitDim.width ? (openBlock(), createElementBlock("div", _hoisted_6, [
                          _hoisted_7,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.aitDim.width,
                            min: 128,
                            max: 2048,
                            step: 8,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.aitDim.width,
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            step: 8,
                            min: 128,
                            max: 2048
                          }, null, 8, ["value"])
                        ])) : (openBlock(), createElementBlock("div", _hoisted_8, [
                          _hoisted_9,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.txt2img.width,
                            "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(conf).data.settings.txt2img.width = $event),
                            min: 128,
                            max: 2048,
                            step: 8,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.txt2img.width,
                            "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(conf).data.settings.txt2img.width = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            step: 8,
                            min: 128,
                            max: 2048
                          }, null, 8, ["value"])
                        ])),
                        unref(conf).data.settings.aitDim.height ? (openBlock(), createElementBlock("div", _hoisted_10, [
                          _hoisted_11,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.aitDim.height,
                            min: 128,
                            max: 2048,
                            step: 8,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.aitDim.height,
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            step: 8,
                            min: 128,
                            max: 2048
                          }, null, 8, ["value"])
                        ])) : (openBlock(), createElementBlock("div", _hoisted_12, [
                          _hoisted_13,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.txt2img.height,
                            "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(conf).data.settings.txt2img.height = $event),
                            min: 128,
                            max: 2048,
                            step: 8,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.txt2img.height,
                            "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(conf).data.settings.txt2img.height = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            step: 8,
                            min: 128,
                            max: 2048
                          }, null, 8, ["value"])
                        ])),
                        createBaseVNode("div", _hoisted_14, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_15
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                              _hoisted_16
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.txt2img.steps,
                            "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(conf).data.settings.txt2img.steps = $event),
                            min: 5,
                            max: 300,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.txt2img.steps,
                            "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(conf).data.settings.txt2img.steps = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 5,
                            max: 300
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_17, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_18
                            ]),
                            default: withCtx(() => [
                              createTextVNode(' Guidance scale indicates how much should model stay close to the prompt. Higher values might be exactly what you want, but generated images might have some artefacts. Lower values indicates that model can "dream" about this prompt more. '),
                              _hoisted_19
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.txt2img.cfg_scale,
                            "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(conf).data.settings.txt2img.cfg_scale = $event),
                            min: 1,
                            max: 30,
                            step: 0.5,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value", "step"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.txt2img.cfg_scale,
                            "onUpdate:value": _cache[10] || (_cache[10] = ($event) => unref(conf).data.settings.txt2img.cfg_scale = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 30,
                            step: 0.5
                          }, null, 8, ["value", "step"])
                        ]),
                        createBaseVNode("div", _hoisted_20, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_21
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of images to generate after each other. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.txt2img.batch_count,
                            "onUpdate:value": _cache[11] || (_cache[11] = ($event) => unref(conf).data.settings.txt2img.batch_count = $event),
                            min: 1,
                            max: 9,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.txt2img.batch_count,
                            "onUpdate:value": _cache[12] || (_cache[12] = ($event) => unref(conf).data.settings.txt2img.batch_count = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 9
                          }, null, 8, ["value"])
                        ]),
                        unref(conf).data.settings.aitDim.batch_size ? (openBlock(), createElementBlock("div", _hoisted_22, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_23
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of images to generate in paralel. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.aitDim.batch_size,
                            min: 1,
                            max: 9,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.aitDim.batch_size,
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 9
                          }, null, 8, ["value"])
                        ])) : (openBlock(), createElementBlock("div", _hoisted_24, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_25
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of images to generate in paralel. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.txt2img.batch_size,
                            "onUpdate:value": _cache[13] || (_cache[13] = ($event) => unref(conf).data.settings.txt2img.batch_size = $event),
                            min: 1,
                            max: 9,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.txt2img.batch_size,
                            "onUpdate:value": _cache[14] || (_cache[14] = ($event) => unref(conf).data.settings.txt2img.batch_size = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 9
                          }, null, 8, ["value"])
                        ])),
                        createBaseVNode("div", _hoisted_26, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_27
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Seed is a number that represents the starting canvas of your image. If you want to create the same image as your friend, you can use the same settings and seed to do so. "),
                              _hoisted_28
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.txt2img.seed,
                            "onUpdate:value": _cache[15] || (_cache[15] = ($event) => unref(conf).data.settings.txt2img.seed = $event),
                            size: "small",
                            min: -1,
                            max: 999999999999,
                            style: { "flex-grow": "1" }
                          }, null, 8, ["value"])
                        ])
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                }),
                createVNode(unref(NCard), {
                  title: "Highres fix",
                  style: { "margin-top": "12px" }
                }, {
                  default: withCtx(() => [
                    createBaseVNode("div", _hoisted_29, [
                      _hoisted_30,
                      createVNode(unref(NSwitch), {
                        value: unref(global).state.txt2img.highres,
                        "onUpdate:value": _cache[16] || (_cache[16] = ($event) => unref(global).state.txt2img.highres = $event)
                      }, null, 8, ["value"])
                    ]),
                    unref(global).state.txt2img.highres ? (openBlock(), createBlock(unref(NSpace), {
                      key: 0,
                      vertical: "",
                      class: "left-container"
                    }, {
                      default: withCtx(() => [
                        createBaseVNode("div", _hoisted_31, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_32
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                              _hoisted_33
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.extra.highres.steps,
                            "onUpdate:value": _cache[17] || (_cache[17] = ($event) => unref(conf).data.settings.extra.highres.steps = $event),
                            min: 5,
                            max: 300,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.extra.highres.steps,
                            "onUpdate:value": _cache[18] || (_cache[18] = ($event) => unref(conf).data.settings.extra.highres.steps = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 5,
                            max: 300
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_34, [
                          _hoisted_35,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.extra.highres.scale,
                            "onUpdate:value": _cache[19] || (_cache[19] = ($event) => unref(conf).data.settings.extra.highres.scale = $event),
                            min: 1,
                            max: 8,
                            step: 1,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.extra.highres.scale,
                            "onUpdate:value": _cache[20] || (_cache[20] = ($event) => unref(conf).data.settings.extra.highres.scale = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 8,
                            step: 1
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_36, [
                          _hoisted_37,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.extra.highres.strength,
                            "onUpdate:value": _cache[21] || (_cache[21] = ($event) => unref(conf).data.settings.extra.highres.strength = $event),
                            min: 0.1,
                            max: 0.9,
                            step: 0.05,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value", "min", "max", "step"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.extra.highres.strength,
                            "onUpdate:value": _cache[22] || (_cache[22] = ($event) => unref(conf).data.settings.extra.highres.strength = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 0.1,
                            max: 0.9,
                            step: 0.05
                          }, null, 8, ["value", "min", "max", "step"])
                        ]),
                        createBaseVNode("div", _hoisted_38, [
                          _hoisted_39,
                          createVNode(unref(NSwitch), {
                            value: unref(conf).data.settings.extra.highres.antialiased,
                            "onUpdate:value": _cache[23] || (_cache[23] = ($event) => unref(conf).data.settings.extra.highres.antialiased = $event)
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_40, [
                          _hoisted_41,
                          createVNode(unref(NSelect), {
                            value: unref(conf).data.settings.extra.highres.latent_scale_mode,
                            "onUpdate:value": _cache[24] || (_cache[24] = ($event) => unref(conf).data.settings.extra.highres.latent_scale_mode = $event),
                            size: "small",
                            style: { "flex-grow": "1" },
                            options: [
                              { label: "Nearest", value: "nearest" },
                              { label: "Nearest exact", value: "nearest-exact" },
                              { label: "Linear", value: "linear" },
                              { label: "Bilinear", value: "bilinear" },
                              { label: "Bicubic", value: "bicubic" }
                            ]
                          }, null, 8, ["value"])
                        ])
                      ]),
                      _: 1
                    })) : createCommentVNode("", true)
                  ]),
                  _: 1
                })
              ]),
              _: 1
            }),
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(_sfc_main$1, { generate }),
                createVNode(_sfc_main$2, {
                  "current-image": unref(global).state.txt2img.currentImage,
                  images: unref(global).state.txt2img.images,
                  onImageClicked: _cache[25] || (_cache[25] = ($event) => unref(global).state.txt2img.currentImage = $event)
                }, null, 8, ["current-image", "images"]),
                createVNode(_sfc_main$3, {
                  output: unref(global).state.txt2img.currentImage
                }, null, 8, ["output"]),
                createVNode(_sfc_main$4, {
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
export {
  _sfc_main as default
};
