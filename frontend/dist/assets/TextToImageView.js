import { _ as _sfc_main$4 } from "./GenerateSection.vue_vue_type_script_setup_true_lang.js";
import { _ as _sfc_main$5 } from "./ImageOutput.vue_vue_type_script_setup_true_lang.js";
import { B as BurnerClock, _ as _sfc_main$2, a as _sfc_main$3, b as _sfc_main$6 } from "./clock.js";
import { d as defineComponent, u as useSettings, r as ref, o as openBlock, c as createElementBlock, a as createVNode, w as withCtx, b as unref, N as NCard, F as Fragment, e as renderList, f as createBaseVNode, t as toDisplayString, g as createBlock, n as normalizeProps, h as guardReactiveProps, i as resolveDynamicComponent, j as NModal, k as createTextVNode, l as NTooltip, m as NSelect, p as NIcon, q as NButton, s as h, v as useState, x as useMessage, y as computed, z as onUnmounted, A as serverUrl, B as NGi, C as NSpace, D as NInput, E as promptHandleKeyUp, G as promptHandleKeyDown, H as createCommentVNode, I as NGrid, J as spaceRegex } from "./index.js";
import { S as Settings, N as NCheckbox } from "./Settings.js";
import { N as NInputNumber } from "./InputNumber.js";
import { N as NSlider, a as NSwitch } from "./Switch.js";
import { v as v4 } from "./v4.js";
import "./SendOutputTo.vue_vue_type_script_setup_true_lang.js";
import "./TrashBin.js";
import "./DescriptionsItem.js";
const _hoisted_1$1 = { class: "flex-container" };
const _hoisted_2$1 = { class: "flex-container" };
const _hoisted_3$1 = { style: { "margin-right": "12px", "white-space": "nowrap" } };
const _hoisted_4$1 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "100px" } }, "Sampler", -1);
const _hoisted_5$1 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using DPMSolverMultistep for the best results . ", -1);
const _hoisted_6$1 = /* @__PURE__ */ createBaseVNode("a", {
  target: "_blank",
  href: "https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
}, "Learn more", -1);
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "SamplerPicker",
  props: {
    type: {
      type: String,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const showModal = ref(false);
    const samplerSettings = {
      eta_noise_seed_delta: {
        componentType: "number",
        min: 0,
        max: 999999999999,
        step: 1,
        modelValue: 0.5
      },
      denoiser_enable_quantization: {
        componentType: "boolean",
        checked: true
      },
      karras_sigma_scheduler: {
        componentType: "boolean",
        checked: false
      },
      sigma_use_old_karras_scheduler: {
        componentType: "boolean",
        checked: false
      },
      sigma_always_discard_next_to_last: {
        componentType: "boolean",
        checked: false
      },
      sigma_rho: {
        componentType: "slider",
        min: 0,
        max: 1,
        step: 0.1,
        modelValue: 0.5
      },
      sigma_min: {
        componentType: "slider",
        min: 0,
        max: 1,
        step: 0.1,
        modelValue: 0.5
      },
      sigma_max: {
        componentType: "slider",
        min: 0,
        max: 1,
        step: 0.1,
        modelValue: 0.5
      },
      sampler_eta: {
        componentType: "slider",
        min: 0,
        max: 1,
        step: 0.1,
        modelValue: 0.5
      },
      sampler_churn: {
        componentType: "slider",
        min: 0,
        max: 1,
        step: 0.1,
        modelValue: 0.5
      },
      sampler_tmin: {
        componentType: "slider",
        min: 0,
        max: 1,
        step: 0.1,
        modelValue: 0.5
      },
      sampler_tmax: {
        componentType: "slider",
        min: 0,
        max: 1,
        step: 0.1,
        modelValue: 0.5
      },
      sampler_noise_seed_delta: {
        componentType: "slider",
        min: 0,
        max: 1,
        step: 0.1,
        modelValue: 0.5
      }
    };
    function resolveComponent(settings2) {
      switch (settings2.componentType) {
        case "slider":
          return h(NSlider, {
            min: settings2.min,
            max: settings2.max,
            step: settings2.step,
            modelValue: settings2.modelValue
          });
        case "select":
          return h(NSelect);
        case "boolean":
          return h(NCheckbox, {
            checked: settings2.checked
          });
        case "number":
          return h(NInputNumber, {
            min: settings2.min,
            max: settings2.max,
            step: settings2.step,
            defaultValue: settings2.modelValue
          });
      }
    }
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
        createVNode(unref(NModal), {
          show: showModal.value,
          "onUpdate:show": _cache[1] || (_cache[1] = ($event) => showModal.value = $event),
          "close-on-esc": "",
          "mask-closable": ""
        }, {
          default: withCtx(() => [
            createVNode(unref(NCard), {
              title: "Sampler settings",
              style: { "max-width": "90vw", "max-height": "90vh" },
              closable: "",
              onClose: _cache[0] || (_cache[0] = ($event) => showModal.value = false)
            }, {
              default: withCtx(() => [
                (openBlock(true), createElementBlock(Fragment, null, renderList(Object.keys(samplerSettings), (param) => {
                  return openBlock(), createElementBlock("div", _hoisted_2$1, [
                    createBaseVNode("p", _hoisted_3$1, toDisplayString(param), 1),
                    (openBlock(), createBlock(resolveDynamicComponent(resolveComponent(samplerSettings[param])), normalizeProps(guardReactiveProps(samplerSettings[param])), null, 16))
                  ]);
                }), 256))
              ]),
              _: 1
            })
          ]),
          _: 1
        }, 8, ["show"]),
        createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
          trigger: withCtx(() => [
            _hoisted_4$1
          ]),
          default: withCtx(() => [
            createTextVNode(" The sampler is the method used to generate the image. Your result may vary drastically depending on the sampler you choose. "),
            _hoisted_5$1,
            _hoisted_6$1
          ]),
          _: 1
        }),
        createVNode(unref(NSelect), {
          options: unref(settings).scheduler_options,
          filterable: "",
          value: unref(settings).data.settings[props.type].sampler,
          "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(settings).data.settings[props.type].sampler = $event),
          style: { "flex-grow": "1" }
        }, null, 8, ["options", "value"]),
        createVNode(unref(NButton), {
          style: { "margin-left": "4px" },
          onClick: _cache[3] || (_cache[3] = ($event) => showModal.value = true)
        }, {
          default: withCtx(() => [
            createVNode(unref(NIcon), null, {
              default: withCtx(() => [
                createVNode(unref(Settings))
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
const _hoisted_1 = { class: "main-container" };
const _hoisted_2 = { class: "flex-container" };
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("p", { style: { "width": "120px" } }, "Karras Sigmas", -1);
const _hoisted_4 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "Works only with KDPM samplers. Ignored by other samplers.", -1);
const _hoisted_5 = { class: "flex-container" };
const _hoisted_6 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_7 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_8 = { class: "flex-container" };
const _hoisted_9 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "CFG Scale", -1);
const _hoisted_10 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 3-15 for most images.", -1);
const _hoisted_11 = {
  key: 0,
  class: "flex-container"
};
const _hoisted_12 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Self Attention Scale", -1);
const _hoisted_13 = { class: "flex-container" };
const _hoisted_14 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Count", -1);
const _hoisted_15 = { class: "flex-container" };
const _hoisted_16 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "75px" } }, "Seed", -1);
const _hoisted_17 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "For random seed use -1.", -1);
const _hoisted_18 = { class: "flex-container" };
const _hoisted_19 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_20 = { class: "flex-container" };
const _hoisted_21 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_22 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_23 = { class: "flex-container" };
const _hoisted_24 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Scale", -1);
const _hoisted_25 = { class: "flex-container" };
const _hoisted_26 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Strength", -1);
const _hoisted_27 = { class: "flex-container" };
const _hoisted_28 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Antialiased", -1);
const _hoisted_29 = { class: "flex-container" };
const _hoisted_30 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Latent Mode", -1);
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
            width: conf.data.settings.txt2img.width,
            height: conf.data.settings.txt2img.height,
            steps: conf.data.settings.txt2img.steps,
            guidance_scale: conf.data.settings.txt2img.cfg_scale,
            seed,
            batch_size: conf.data.settings.txt2img.batch_size,
            batch_count: conf.data.settings.txt2img.batch_count,
            scheduler: conf.data.settings.txt2img.sampler,
            self_attention_scale: conf.data.settings.txt2img.self_attention_scale,
            use_karras_sigmas: conf.data.settings.txt2img.use_karras_sigmas
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
        console.log(err);
      });
    };
    computed(() => {
      var _a;
      return ((_a = conf.data.settings.model) == null ? void 0 : _a.backend) === "PyTorch";
    });
    const burner = new BurnerClock(conf.data.settings.txt2img, conf, generate);
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
                      default: withCtx(() => {
                        var _a;
                        return [
                          createVNode(unref(NInput), {
                            value: unref(conf).data.settings.txt2img.prompt,
                            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(conf).data.settings.txt2img.prompt = $event),
                            type: "textarea",
                            placeholder: "Prompt",
                            "show-count": "",
                            onKeyup: _cache[1] || (_cache[1] = ($event) => unref(promptHandleKeyUp)(
                              $event,
                              unref(conf).data.settings.txt2img,
                              "prompt",
                              unref(global)
                            )),
                            onKeydown: unref(promptHandleKeyDown)
                          }, {
                            count: withCtx(() => [
                              createTextVNode(toDisplayString(promptCount.value), 1)
                            ]),
                            _: 1
                          }, 8, ["value", "onKeydown"]),
                          createVNode(unref(NInput), {
                            value: unref(conf).data.settings.txt2img.negative_prompt,
                            "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(conf).data.settings.txt2img.negative_prompt = $event),
                            type: "textarea",
                            placeholder: "Negative prompt",
                            "show-count": "",
                            onKeyup: _cache[3] || (_cache[3] = ($event) => unref(promptHandleKeyUp)(
                              $event,
                              unref(conf).data.settings.txt2img,
                              "negative_prompt",
                              unref(global)
                            )),
                            onKeydown: unref(promptHandleKeyDown)
                          }, {
                            count: withCtx(() => [
                              createTextVNode(toDisplayString(negativePromptCount.value), 1)
                            ]),
                            _: 1
                          }, 8, ["value", "onKeydown"]),
                          createVNode(_sfc_main$1, { type: "txt2img" }),
                          createBaseVNode("div", _hoisted_2, [
                            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                              trigger: withCtx(() => [
                                _hoisted_3
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" Changes the sigmas used in the Karras diffusion process. Might provide better results for some images. "),
                                _hoisted_4
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSwitch), {
                              value: unref(conf).data.settings.txt2img.use_karras_sigmas,
                              "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(conf).data.settings.txt2img.use_karras_sigmas = $event),
                              style: { "justify-self": "flex-end" }
                            }, null, 8, ["value"])
                          ]),
                          createVNode(_sfc_main$2, {
                            "dimensions-object": unref(conf).data.settings.txt2img
                          }, null, 8, ["dimensions-object"]),
                          createBaseVNode("div", _hoisted_5, [
                            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                              trigger: withCtx(() => [
                                _hoisted_6
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                                _hoisted_7
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSlider), {
                              value: unref(conf).data.settings.txt2img.steps,
                              "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(conf).data.settings.txt2img.steps = $event),
                              min: 5,
                              max: 300,
                              style: { "margin-right": "12px" }
                            }, null, 8, ["value"]),
                            createVNode(unref(NInputNumber), {
                              value: unref(conf).data.settings.txt2img.steps,
                              "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(conf).data.settings.txt2img.steps = $event),
                              size: "small",
                              style: { "min-width": "96px", "width": "96px" }
                            }, null, 8, ["value"])
                          ]),
                          createBaseVNode("div", _hoisted_8, [
                            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                              trigger: withCtx(() => [
                                _hoisted_9
                              ]),
                              default: withCtx(() => [
                                createTextVNode(' Guidance scale indicates how much should model stay close to the prompt. Higher values might be exactly what you want, but generated images might have some artefacts. Lower values indicates that model can "dream" about this prompt more. '),
                                _hoisted_10
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSlider), {
                              value: unref(conf).data.settings.txt2img.cfg_scale,
                              "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(conf).data.settings.txt2img.cfg_scale = $event),
                              min: 1,
                              max: 30,
                              step: 0.5,
                              style: { "margin-right": "12px" }
                            }, null, 8, ["value"]),
                            createVNode(unref(NInputNumber), {
                              value: unref(conf).data.settings.txt2img.cfg_scale,
                              "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(conf).data.settings.txt2img.cfg_scale = $event),
                              size: "small",
                              style: { "min-width": "96px", "width": "96px" },
                              step: 0.5
                            }, null, 8, ["value"])
                          ]),
                          Number.isInteger(unref(conf).data.settings.txt2img.sampler) && ((_a = unref(conf).data.settings.model) == null ? void 0 : _a.backend) === "PyTorch" ? (openBlock(), createElementBlock("div", _hoisted_11, [
                            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                              trigger: withCtx(() => [
                                _hoisted_12
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" If self attention is >0, SAG will guide the model and improve the quality of the image at the cost of speed. Higher values will follow the guidance more closely, which can lead to better, more sharp and detailed outputs. ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSlider), {
                              value: unref(conf).data.settings.txt2img.self_attention_scale,
                              "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(conf).data.settings.txt2img.self_attention_scale = $event),
                              min: 0,
                              max: 1,
                              step: 0.05,
                              style: { "margin-right": "12px" }
                            }, null, 8, ["value"]),
                            createVNode(unref(NInputNumber), {
                              value: unref(conf).data.settings.txt2img.self_attention_scale,
                              "onUpdate:value": _cache[10] || (_cache[10] = ($event) => unref(conf).data.settings.txt2img.self_attention_scale = $event),
                              size: "small",
                              style: { "min-width": "96px", "width": "96px" },
                              step: 0.05
                            }, null, 8, ["value"])
                          ])) : createCommentVNode("", true),
                          createBaseVNode("div", _hoisted_13, [
                            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                              trigger: withCtx(() => [
                                _hoisted_14
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
                              style: { "min-width": "96px", "width": "96px" }
                            }, null, 8, ["value"])
                          ]),
                          createVNode(_sfc_main$3, {
                            "batch-size-object": unref(conf).data.settings.txt2img
                          }, null, 8, ["batch-size-object"]),
                          createBaseVNode("div", _hoisted_15, [
                            createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                              trigger: withCtx(() => [
                                _hoisted_16
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" Seed is a number that represents the starting canvas of your image. If you want to create the same image as your friend, you can use the same settings and seed to do so. "),
                                _hoisted_17
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NInputNumber), {
                              value: unref(conf).data.settings.txt2img.seed,
                              "onUpdate:value": _cache[13] || (_cache[13] = ($event) => unref(conf).data.settings.txt2img.seed = $event),
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
                createVNode(unref(NCard), {
                  title: "Highres fix",
                  style: { "margin-top": "12px", "margin-bottom": "12px" }
                }, {
                  default: withCtx(() => [
                    createBaseVNode("div", _hoisted_18, [
                      _hoisted_19,
                      createVNode(unref(NSwitch), {
                        value: unref(global).state.txt2img.highres,
                        "onUpdate:value": _cache[14] || (_cache[14] = ($event) => unref(global).state.txt2img.highres = $event)
                      }, null, 8, ["value"])
                    ]),
                    unref(global).state.txt2img.highres ? (openBlock(), createBlock(unref(NSpace), {
                      key: 0,
                      vertical: "",
                      class: "left-container"
                    }, {
                      default: withCtx(() => [
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
                            value: unref(conf).data.settings.extra.highres.steps,
                            "onUpdate:value": _cache[15] || (_cache[15] = ($event) => unref(conf).data.settings.extra.highres.steps = $event),
                            min: 5,
                            max: 300,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.extra.highres.steps,
                            "onUpdate:value": _cache[16] || (_cache[16] = ($event) => unref(conf).data.settings.extra.highres.steps = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" }
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_23, [
                          _hoisted_24,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.extra.highres.scale,
                            "onUpdate:value": _cache[17] || (_cache[17] = ($event) => unref(conf).data.settings.extra.highres.scale = $event),
                            min: 1,
                            max: 8,
                            step: 0.1,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.extra.highres.scale,
                            "onUpdate:value": _cache[18] || (_cache[18] = ($event) => unref(conf).data.settings.extra.highres.scale = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            step: 0.1
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_25, [
                          _hoisted_26,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.extra.highres.strength,
                            "onUpdate:value": _cache[19] || (_cache[19] = ($event) => unref(conf).data.settings.extra.highres.strength = $event),
                            min: 0.1,
                            max: 0.9,
                            step: 0.05,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.extra.highres.strength,
                            "onUpdate:value": _cache[20] || (_cache[20] = ($event) => unref(conf).data.settings.extra.highres.strength = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 0.1,
                            max: 0.9,
                            step: 0.05
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_27, [
                          _hoisted_28,
                          createVNode(unref(NSwitch), {
                            value: unref(conf).data.settings.extra.highres.antialiased,
                            "onUpdate:value": _cache[21] || (_cache[21] = ($event) => unref(conf).data.settings.extra.highres.antialiased = $event)
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_29, [
                          _hoisted_30,
                          createVNode(unref(NSelect), {
                            value: unref(conf).data.settings.extra.highres.latent_scale_mode,
                            "onUpdate:value": _cache[22] || (_cache[22] = ($event) => unref(conf).data.settings.extra.highres.latent_scale_mode = $event),
                            size: "small",
                            style: { "flex-grow": "1" },
                            options: [
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
                            ]
                          }, null, 8, ["value", "options"])
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
                createVNode(_sfc_main$4, { generate }),
                createVNode(_sfc_main$5, {
                  "current-image": unref(global).state.txt2img.currentImage,
                  images: unref(global).state.txt2img.images,
                  data: unref(conf).data.settings.txt2img,
                  onImageClicked: _cache[23] || (_cache[23] = ($event) => unref(global).state.txt2img.currentImage = $event)
                }, null, 8, ["current-image", "images", "data"]),
                createVNode(_sfc_main$6, {
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
