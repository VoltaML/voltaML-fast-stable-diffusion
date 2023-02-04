import { d as defineComponent, u as useState, a as useSettings, b as useMessage, o as openBlock, c as createElementBlock, e as createVNode, w as withCtx, f as unref, N as NGi, g as NCard, h as NSpace, i as NInput, j as createBaseVNode, k as NTooltip, l as createTextVNode, m as NSelect, n as NSlider, p as NInputNumber, _ as _sfc_main$2, q as NImageGroup, r as createBlock, s as NImage, t as createCommentVNode, v as NGrid, x as serverUrl, y as v4, z as pushScopeId, A as popScopeId, B as _export_sfc } from "./index.js";
import { I as ImageUpload } from "./ImageUpload.js";
import { _ as _sfc_main$3 } from "./WIP.vue_vue_type_script_setup_true_lang.js";
import { N as NTabPane, a as NTabs } from "./Tabs.js";
import "./Result.js";
const _withScopeId = (n) => (pushScopeId("data-v-01e2b6c0"), n = n(), popScopeId(), n);
const _hoisted_1 = { style: { "margin": "0 12px" } };
const _hoisted_2 = { class: "flex-container" };
const _hoisted_3 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "150px" } }, "Sampler", -1));
const _hoisted_4 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using Euler A for the best results (but it also takes more time). ", -1));
const _hoisted_5 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("a", {
  target: "_blank",
  href: "https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
}, "Learn more", -1));
const _hoisted_6 = { class: "flex-container" };
const _hoisted_7 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1));
const _hoisted_8 = { class: "flex-container" };
const _hoisted_9 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1));
const _hoisted_10 = { class: "flex-container" };
const _hoisted_11 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1));
const _hoisted_12 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1));
const _hoisted_13 = { class: "flex-container" };
const _hoisted_14 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "CFG Scale", -1));
const _hoisted_15 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 3-15 for most images.", -1));
const _hoisted_16 = { class: "flex-container" };
const _hoisted_17 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Count", -1));
const _hoisted_18 = { class: "flex-container" };
const _hoisted_19 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1));
const _hoisted_20 = { class: "flex-container" };
const _hoisted_21 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Denoising Strength", -1));
const _hoisted_22 = { class: "flex-container" };
const _hoisted_23 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Seed", -1));
const _hoisted_24 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "For random seed use -1.", -1));
const _hoisted_25 = { style: { "height": "70vh", "width": "100%", "display": "flex", "justify-content": "center" } };
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "Img2Img",
  setup(__props) {
    const global = useState();
    const conf = useSettings();
    const messageHandler = useMessage();
    const checkSeed = (seed) => {
      if (seed === -1) {
        seed = Math.floor(Math.random() * 999999999);
      }
      return seed;
    };
    const imageSelectCallback = (base64Image) => {
      conf.data.settings.img2img.image = base64Image;
    };
    const generate = () => {
      if (conf.data.settings.txt2img.seed === null) {
        messageHandler.error("Please set a seed");
        return;
      }
      global.state.generating = true;
      fetch(`${serverUrl}/api/img2img/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          data: {
            prompt: conf.data.settings.img2img.prompt,
            image: conf.data.settings.img2img.image,
            id: v4(),
            negative_prompt: conf.data.settings.img2img.negativePrompt,
            width: conf.data.settings.img2img.width,
            height: conf.data.settings.img2img.height,
            steps: conf.data.settings.img2img.steps,
            guidance_scale: conf.data.settings.img2img.cfgScale,
            seed: checkSeed(conf.data.settings.img2img.seed),
            batch_size: conf.data.settings.img2img.batchSize,
            batch_count: conf.data.settings.img2img.batchCount,
            strength: conf.data.settings.img2img.denoisingStrength
          },
          model: conf.data.settings.model,
          scheduler: conf.data.settings.img2img.sampler
        })
      }).then((res) => {
        global.state.generating = false;
        res.json().then((data) => {
          global.state.txt2img.images = data.images;
          global.state.progress = 0;
          global.state.total_steps = 0;
          global.state.current_step = 0;
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
          cols: "1 850:2",
          "x-gap": "12"
        }, {
          default: withCtx(() => [
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(ImageUpload, {
                  callback: imageSelectCallback,
                  preview: unref(conf).data.settings.img2img.image,
                  style: { "margin-bottom": "12px" }
                }, null, 8, ["preview"]),
                createVNode(unref(NCard), { title: "Settings" }, {
                  default: withCtx(() => [
                    createVNode(unref(NSpace), {
                      vertical: "",
                      class: "left-container"
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NInput), {
                          value: unref(conf).data.settings.img2img.prompt,
                          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(conf).data.settings.img2img.prompt = $event),
                          type: "textarea",
                          placeholder: "Prompt"
                        }, null, 8, ["value"]),
                        createVNode(unref(NInput), {
                          value: unref(conf).data.settings.img2img.negativePrompt,
                          "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(conf).data.settings.img2img.negativePrompt = $event),
                          type: "textarea",
                          placeholder: "Negative prompt"
                        }, null, 8, ["value"]),
                        createBaseVNode("div", _hoisted_2, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
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
                            value: unref(conf).data.settings.img2img.sampler,
                            "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(conf).data.settings.img2img.sampler = $event),
                            style: { "flex-grow": "1" }
                          }, null, 8, ["options", "value"])
                        ]),
                        createBaseVNode("div", _hoisted_6, [
                          _hoisted_7,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.img2img.width,
                            "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(conf).data.settings.img2img.width = $event),
                            min: 128,
                            max: 2048,
                            step: 8,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.img2img.width,
                            "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(conf).data.settings.img2img.width = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            step: 8,
                            min: 128,
                            max: 2048
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_8, [
                          _hoisted_9,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.img2img.height,
                            "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(conf).data.settings.img2img.height = $event),
                            min: 128,
                            max: 2048,
                            step: 8,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.img2img.height,
                            "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(conf).data.settings.img2img.height = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            step: 8,
                            min: 128,
                            max: 2048
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_10, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_11
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                              _hoisted_12
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.img2img.steps,
                            "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(conf).data.settings.img2img.steps = $event),
                            min: 5,
                            max: 300,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.img2img.steps,
                            "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(conf).data.settings.img2img.steps = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 5,
                            max: 300
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_13, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_14
                            ]),
                            default: withCtx(() => [
                              createTextVNode(' Guidance scale indicates how much should model stay close to the prompt. Higher values might be exactly what you want, but generated images might have some artefacts. Lower values indicates that model can "dream" about this prompt more. '),
                              _hoisted_15
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.img2img.cfgScale,
                            "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(conf).data.settings.img2img.cfgScale = $event),
                            min: 1,
                            max: 30,
                            step: 0.5,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value", "step"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.img2img.cfgScale,
                            "onUpdate:value": _cache[10] || (_cache[10] = ($event) => unref(conf).data.settings.img2img.cfgScale = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 30,
                            step: 0.5
                          }, null, 8, ["value", "step"])
                        ]),
                        createBaseVNode("div", _hoisted_16, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_17
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of images to generate after each other. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.img2img.batchCount,
                            "onUpdate:value": _cache[11] || (_cache[11] = ($event) => unref(conf).data.settings.img2img.batchCount = $event),
                            min: 1,
                            max: 9,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.img2img.batchCount,
                            "onUpdate:value": _cache[12] || (_cache[12] = ($event) => unref(conf).data.settings.img2img.batchCount = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 9
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_18, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_19
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of images to generate in paralel. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.img2img.batchSize,
                            "onUpdate:value": _cache[13] || (_cache[13] = ($event) => unref(conf).data.settings.img2img.batchSize = $event),
                            min: 1,
                            max: 9,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.img2img.batchSize,
                            "onUpdate:value": _cache[14] || (_cache[14] = ($event) => unref(conf).data.settings.img2img.batchSize = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 9
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_20, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_21
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Lower values will stick more to the original image, 0.5-0.75 is ideal ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.img2img.denoisingStrength,
                            "onUpdate:value": _cache[15] || (_cache[15] = ($event) => unref(conf).data.settings.img2img.denoisingStrength = $event),
                            min: 0.1,
                            max: 1,
                            style: { "margin-right": "12px" },
                            step: 0.025
                          }, null, 8, ["value", "min", "step"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.img2img.denoisingStrength,
                            "onUpdate:value": _cache[16] || (_cache[16] = ($event) => unref(conf).data.settings.img2img.denoisingStrength = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 0.1,
                            max: 1,
                            step: 0.025
                          }, null, 8, ["value", "min", "step"])
                        ]),
                        createBaseVNode("div", _hoisted_22, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_23
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Seed is a number that represents the starting canvas of your image. If you want to create the same image as your friend, you can use the same settings and seed to do so. "),
                              _hoisted_24
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.img2img.seed,
                            "onUpdate:value": _cache[17] || (_cache[17] = ($event) => unref(conf).data.settings.img2img.seed = $event),
                            size: "small",
                            min: -1,
                            max: 999999999,
                            style: { "flex-grow": "1" }
                          }, null, 8, ["value"])
                        ])
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                })
              ]),
              _: 1
            }),
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(_sfc_main$2, { generate }),
                createVNode(unref(NCard), {
                  title: "Output",
                  hoverable: ""
                }, {
                  default: withCtx(() => [
                    createBaseVNode("div", _hoisted_25, [
                      createVNode(unref(NImageGroup), { style: { "max-width": "100%", "max-height": "70vh", "width": "100%", "height": "100%" } }, {
                        default: withCtx(() => [
                          unref(global).state.img2img.currentImage ? (openBlock(), createBlock(unref(NImage), {
                            key: 0,
                            src: `data:image/png;base64,${unref(global).state.img2img.currentImage}`,
                            "img-props": {
                              style: "max-width: 100%; max-height: 70vh; width: 100%"
                            },
                            style: { "max-width": "100%", "max-height": "70vh", "width": "100%", "height": "100%" },
                            "object-fit": "contain"
                          }, null, 8, ["src"])) : createCommentVNode("", true)
                        ]),
                        _: 1
                      })
                    ])
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
const Img2Img_vue_vue_type_style_index_0_scoped_01e2b6c0_lang = "";
const Img2Img = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["__scopeId", "data-v-01e2b6c0"]]);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "Image2ImageView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NTabs), { type: "segment" }, {
        default: withCtx(() => [
          createVNode(unref(NTabPane), { name: "Image to Image" }, {
            default: withCtx(() => [
              createVNode(Img2Img)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Image variations" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$3)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Inpaint" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$3)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "SD Upscale" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$3)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Depth to Image" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$3)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Pix to Pix" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$3)
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
export {
  _sfc_main as default
};
