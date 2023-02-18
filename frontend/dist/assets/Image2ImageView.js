import { d as defineComponent, r as ref, o as openBlock, c as createBlock, w as withCtx, a as createBaseVNode, b as createVNode, t as toDisplayString, u as unref, N as NSpace, e as NCard, p as pushScopeId, f as popScopeId, _ as _export_sfc, g as useState, h as useSettings, i as useMessage, j as createElementBlock, k as NGi, l as NTooltip, m as createTextVNode, n as NSelect, q as NSlider, s as NInputNumber, v as _sfc_main$5, I as ImageOutput, x as NGrid, F as Fragment, y as serverUrl, z as v4, A as NInput, B as NButton, C as NIcon } from "./index.js";
import { N as NAlert } from "./Alert.js";
import { V as VueDrawingCanvas, A as ArrowUndoSharp, a as ArrowRedoSharp, B as BrushSharp, T as TrashBinSharp } from "./vue-drawing-canvas.esm.js";
import { _ as _sfc_main$6 } from "./WIP.vue_vue_type_script_setup_true_lang.js";
import { N as NTabPane, a as NTabs } from "./Tabs.js";
import "./Result.js";
const _withScopeId$2 = (n) => (pushScopeId("data-v-19b3e0b6"), n = n(), popScopeId(), n);
const _hoisted_1$3 = { class: "image-container" };
const _hoisted_2$3 = ["src"];
const _hoisted_3$3 = /* @__PURE__ */ _withScopeId$2(() => /* @__PURE__ */ createBaseVNode("label", { for: "file-upload" }, [
  /* @__PURE__ */ createBaseVNode("span", { class: "file-upload" }, "Select image")
], -1));
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "ImageUpload",
  props: {
    callback: {
      type: Object
    },
    preview: {
      type: String
    }
  },
  setup(__props) {
    const props = __props;
    const width = ref(0);
    const height = ref(0);
    function previewImage(event) {
      const input = event.target;
      if (input.files) {
        const reader = new FileReader();
        reader.onload = (e) => {
          var _a;
          const i = (_a = e.target) == null ? void 0 : _a.result;
          if (i) {
            const s = i.toString();
            if (props.callback) {
              props.callback(s);
            }
            const img = new Image();
            img.src = s;
            img.onload = () => {
              width.value = img.width;
              height.value = img.height;
            };
          }
        };
        reader.readAsDataURL(input.files[0]);
      }
    }
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), { title: "Input image" }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1$3, [
            createBaseVNode("img", {
              src: _ctx.$props.preview,
              style: { "width": "400px", "height": "auto" }
            }, null, 8, _hoisted_2$3)
          ]),
          createVNode(unref(NSpace), {
            inline: "",
            justify: "space-between",
            align: "center",
            style: { "width": "100%" }
          }, {
            default: withCtx(() => [
              createBaseVNode("p", null, toDisplayString(width.value) + "x" + toDisplayString(height.value), 1),
              _hoisted_3$3
            ]),
            _: 1
          }),
          createBaseVNode("input", {
            type: "file",
            accept: "image/*",
            onChange: previewImage,
            id: "file-upload",
            class: "hidden-input"
          }, null, 32)
        ]),
        _: 1
      });
    };
  }
});
const ImageUpload_vue_vue_type_style_index_0_scoped_19b3e0b6_lang = "";
const ImageUpload = /* @__PURE__ */ _export_sfc(_sfc_main$4, [["__scopeId", "data-v-19b3e0b6"]]);
const _hoisted_1$2 = { style: { "margin": "0 12px" } };
const _hoisted_2$2 = { class: "flex-container" };
const _hoisted_3$2 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "150px" } }, "Sampler", -1);
const _hoisted_4$2 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using Euler A for the best results (but it also takes more time). ", -1);
const _hoisted_5$2 = /* @__PURE__ */ createBaseVNode("a", {
  target: "_blank",
  href: "https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
}, "Learn more", -1);
const _hoisted_6$2 = { class: "flex-container" };
const _hoisted_7$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_8$2 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_9$2 = { class: "flex-container" };
const _hoisted_10$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "CFG Scale", -1);
const _hoisted_11$2 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 3-15 for most images.", -1);
const _hoisted_12$2 = { class: "flex-container" };
const _hoisted_13$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Count", -1);
const _hoisted_14$2 = { class: "flex-container" };
const _hoisted_15$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1);
const _hoisted_16$2 = { class: "flex-container" };
const _hoisted_17$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Seed", -1);
const _hoisted_18$2 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "For random seed use -1.", -1);
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "ImageVariations",
  setup(__props) {
    const global = useState();
    const conf = useSettings();
    const messageHandler = useMessage();
    const imageSelectCallback = (base64Image) => {
      conf.data.settings.imageVariations.image = base64Image;
    };
    const checkSeed = (seed) => {
      if (seed === -1) {
        seed = Math.floor(Math.random() * 999999999);
      }
      return seed;
    };
    const generate = () => {
      if (conf.data.settings.imageVariations.seed === null) {
        messageHandler.error("Please set a seed");
        return;
      }
      global.state.generating = true;
      fetch(`${serverUrl}/api/generate/image_variations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          data: {
            image: conf.data.settings.imageVariations.image,
            id: v4(),
            steps: conf.data.settings.imageVariations.steps,
            guidance_scale: conf.data.settings.imageVariations.cfgScale,
            seed: checkSeed(conf.data.settings.imageVariations.seed),
            batch_size: conf.data.settings.imageVariations.batchSize,
            batch_count: conf.data.settings.imageVariations.batchCount,
            scheduler: conf.data.settings.imageVariations.sampler
          },
          model: conf.data.settings.model
        })
      }).then((res) => {
        global.state.generating = false;
        res.json().then((data) => {
          global.state.imageVariations.images = data.images;
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
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(unref(NAlert), {
          style: { "width": "100%", "margin-bottom": "12px" },
          type: "warning",
          title: "Does not work yet"
        }),
        createBaseVNode("div", _hoisted_1$2, [
          createVNode(unref(NGrid), {
            cols: "1 850:2",
            "x-gap": "12"
          }, {
            default: withCtx(() => [
              createVNode(unref(NGi), null, {
                default: withCtx(() => [
                  createVNode(ImageUpload, {
                    callback: imageSelectCallback,
                    preview: unref(conf).data.settings.imageVariations.image,
                    style: { "margin-bottom": "12px" }
                  }, null, 8, ["preview"]),
                  createVNode(unref(NCard), { title: "Settings" }, {
                    default: withCtx(() => [
                      createVNode(unref(NSpace), {
                        vertical: "",
                        class: "left-container"
                      }, {
                        default: withCtx(() => [
                          createBaseVNode("div", _hoisted_2$2, [
                            createVNode(unref(NTooltip), { "max-width": 600 }, {
                              trigger: withCtx(() => [
                                _hoisted_3$2
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" The sampler is the method used to generate the image. Your result may vary drastically depending on the sampler you choose. "),
                                _hoisted_4$2,
                                _hoisted_5$2
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSelect), {
                              options: unref(conf).scheduler_options,
                              value: unref(conf).data.settings.imageVariations.sampler,
                              "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(conf).data.settings.imageVariations.sampler = $event),
                              style: { "flex-grow": "1" }
                            }, null, 8, ["options", "value"])
                          ]),
                          createBaseVNode("div", _hoisted_6$2, [
                            createVNode(unref(NTooltip), { "max-width": 600 }, {
                              trigger: withCtx(() => [
                                _hoisted_7$2
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                                _hoisted_8$2
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSlider), {
                              value: unref(conf).data.settings.imageVariations.steps,
                              "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(conf).data.settings.imageVariations.steps = $event),
                              min: 5,
                              max: 300,
                              style: { "margin-right": "12px" }
                            }, null, 8, ["value"]),
                            createVNode(unref(NInputNumber), {
                              value: unref(conf).data.settings.imageVariations.steps,
                              "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(conf).data.settings.imageVariations.steps = $event),
                              size: "small",
                              style: { "min-width": "96px", "width": "96px" },
                              min: 5,
                              max: 300
                            }, null, 8, ["value"])
                          ]),
                          createBaseVNode("div", _hoisted_9$2, [
                            createVNode(unref(NTooltip), { "max-width": 600 }, {
                              trigger: withCtx(() => [
                                _hoisted_10$2
                              ]),
                              default: withCtx(() => [
                                createTextVNode(' Guidance scale indicates how much should model stay close to the prompt. Higher values might be exactly what you want, but generated images might have some artefacts. Lower values indicates that model can "dream" about this prompt more. '),
                                _hoisted_11$2
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSlider), {
                              value: unref(conf).data.settings.imageVariations.cfgScale,
                              "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(conf).data.settings.imageVariations.cfgScale = $event),
                              min: 1,
                              max: 30,
                              step: 0.5,
                              style: { "margin-right": "12px" }
                            }, null, 8, ["value", "step"]),
                            createVNode(unref(NInputNumber), {
                              value: unref(conf).data.settings.imageVariations.cfgScale,
                              "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(conf).data.settings.imageVariations.cfgScale = $event),
                              size: "small",
                              style: { "min-width": "96px", "width": "96px" },
                              min: 1,
                              max: 30,
                              step: 0.5
                            }, null, 8, ["value", "step"])
                          ]),
                          createBaseVNode("div", _hoisted_12$2, [
                            createVNode(unref(NTooltip), { "max-width": 600 }, {
                              trigger: withCtx(() => [
                                _hoisted_13$2
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" Number of images to generate after each other. ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSlider), {
                              value: unref(conf).data.settings.imageVariations.batchCount,
                              "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(conf).data.settings.imageVariations.batchCount = $event),
                              min: 1,
                              max: 9,
                              style: { "margin-right": "12px" }
                            }, null, 8, ["value"]),
                            createVNode(unref(NInputNumber), {
                              value: unref(conf).data.settings.imageVariations.batchCount,
                              "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(conf).data.settings.imageVariations.batchCount = $event),
                              size: "small",
                              style: { "min-width": "96px", "width": "96px" },
                              min: 1,
                              max: 9
                            }, null, 8, ["value"])
                          ]),
                          createBaseVNode("div", _hoisted_14$2, [
                            createVNode(unref(NTooltip), { "max-width": 600 }, {
                              trigger: withCtx(() => [
                                _hoisted_15$2
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" Number of images to generate in paralel. ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NSlider), {
                              value: unref(conf).data.settings.imageVariations.batchSize,
                              "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(conf).data.settings.imageVariations.batchSize = $event),
                              min: 1,
                              max: 9,
                              style: { "margin-right": "12px" }
                            }, null, 8, ["value"]),
                            createVNode(unref(NInputNumber), {
                              value: unref(conf).data.settings.imageVariations.batchSize,
                              "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(conf).data.settings.imageVariations.batchSize = $event),
                              size: "small",
                              style: { "min-width": "96px", "width": "96px" },
                              min: 1,
                              max: 9
                            }, null, 8, ["value"])
                          ]),
                          createBaseVNode("div", _hoisted_16$2, [
                            createVNode(unref(NTooltip), { "max-width": 600 }, {
                              trigger: withCtx(() => [
                                _hoisted_17$2
                              ]),
                              default: withCtx(() => [
                                createTextVNode(" Seed is a number that represents the starting canvas of your image. If you want to create the same image as your friend, you can use the same settings and seed to do so. "),
                                _hoisted_18$2
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NInputNumber), {
                              value: unref(conf).data.settings.imageVariations.seed,
                              "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(conf).data.settings.imageVariations.seed = $event),
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
                  createVNode(_sfc_main$5, { generate }),
                  createVNode(ImageOutput, {
                    "current-image": unref(global).state.imageVariations.currentImage,
                    images: unref(global).state.imageVariations.images
                  }, null, 8, ["current-image", "images"])
                ]),
                _: 1
              })
            ]),
            _: 1
          })
        ])
      ], 64);
    };
  }
});
const _withScopeId$1 = (n) => (pushScopeId("data-v-a1e97e18"), n = n(), popScopeId(), n);
const _hoisted_1$1 = { style: { "margin": "0 12px" } };
const _hoisted_2$1 = { class: "flex-container" };
const _hoisted_3$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "150px" } }, "Sampler", -1));
const _hoisted_4$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using Euler A for the best results (but it also takes more time). ", -1));
const _hoisted_5$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("a", {
  target: "_blank",
  href: "https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
}, "Learn more", -1));
const _hoisted_6$1 = { class: "flex-container" };
const _hoisted_7$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1));
const _hoisted_8$1 = { class: "flex-container" };
const _hoisted_9$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1));
const _hoisted_10$1 = { class: "flex-container" };
const _hoisted_11$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1));
const _hoisted_12$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1));
const _hoisted_13$1 = { class: "flex-container" };
const _hoisted_14$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "CFG Scale", -1));
const _hoisted_15$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 3-15 for most images.", -1));
const _hoisted_16$1 = { class: "flex-container" };
const _hoisted_17$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Count", -1));
const _hoisted_18$1 = { class: "flex-container" };
const _hoisted_19$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1));
const _hoisted_20$1 = { class: "flex-container" };
const _hoisted_21$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Denoising Strength", -1));
const _hoisted_22$1 = { class: "flex-container" };
const _hoisted_23$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Seed", -1));
const _hoisted_24$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "For random seed use -1.", -1));
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
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
      if (conf.data.settings.img2img.seed === null) {
        messageHandler.error("Please set a seed");
        return;
      }
      global.state.generating = true;
      fetch(`${serverUrl}/api/generate/img2img`, {
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
            strength: conf.data.settings.img2img.denoisingStrength,
            scheduler: conf.data.settings.img2img.sampler
          },
          model: conf.data.settings.model
        })
      }).then((res) => {
        global.state.generating = false;
        res.json().then((data) => {
          global.state.img2img.images = data.images;
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
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
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
                        createBaseVNode("div", _hoisted_2$1, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_3$1
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" The sampler is the method used to generate the image. Your result may vary drastically depending on the sampler you choose. "),
                              _hoisted_4$1,
                              _hoisted_5$1
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
                        createBaseVNode("div", _hoisted_6$1, [
                          _hoisted_7$1,
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
                        createBaseVNode("div", _hoisted_8$1, [
                          _hoisted_9$1,
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
                        createBaseVNode("div", _hoisted_10$1, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_11$1
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                              _hoisted_12$1
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
                        createBaseVNode("div", _hoisted_13$1, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_14$1
                            ]),
                            default: withCtx(() => [
                              createTextVNode(' Guidance scale indicates how much should model stay close to the prompt. Higher values might be exactly what you want, but generated images might have some artefacts. Lower values indicates that model can "dream" about this prompt more. '),
                              _hoisted_15$1
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
                        createBaseVNode("div", _hoisted_16$1, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_17$1
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
                        createBaseVNode("div", _hoisted_18$1, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_19$1
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
                        createBaseVNode("div", _hoisted_20$1, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_21$1
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
                        createBaseVNode("div", _hoisted_22$1, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_23$1
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Seed is a number that represents the starting canvas of your image. If you want to create the same image as your friend, you can use the same settings and seed to do so. "),
                              _hoisted_24$1
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
                createVNode(_sfc_main$5, { generate }),
                createVNode(ImageOutput, {
                  "current-image": unref(global).state.img2img.currentImage,
                  images: unref(global).state.img2img.images
                }, null, 8, ["current-image", "images"])
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
const Img2Img_vue_vue_type_style_index_0_scoped_a1e97e18_lang = "";
const Img2Img = /* @__PURE__ */ _export_sfc(_sfc_main$2, [["__scopeId", "data-v-a1e97e18"]]);
const _withScopeId = (n) => (pushScopeId("data-v-ba9c9a75"), n = n(), popScopeId(), n);
const _hoisted_1 = { style: { "margin": "0 12px" } };
const _hoisted_2 = { style: { "display": "inline-flex", "align-items": "center" } };
const _hoisted_3 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("svg", {
  xmlns: "http://www.w3.org/2000/svg",
  width: "16",
  height: "16",
  fill: "currentColor",
  class: "bi bi-eraser",
  viewBox: "0 0 16 16"
}, [
  /* @__PURE__ */ createBaseVNode("path", { d: "M8.086 2.207a2 2 0 0 1 2.828 0l3.879 3.879a2 2 0 0 1 0 2.828l-5.5 5.5A2 2 0 0 1 7.879 15H5.12a2 2 0 0 1-1.414-.586l-2.5-2.5a2 2 0 0 1 0-2.828l6.879-6.879zm2.121.707a1 1 0 0 0-1.414 0L4.16 7.547l5.293 5.293 4.633-4.633a1 1 0 0 0 0-1.414l-3.879-3.879zM8.746 13.547 3.453 8.254 1.914 9.793a1 1 0 0 0 0 1.414l2.5 2.5a1 1 0 0 0 .707.293H7.88a1 1 0 0 0 .707-.293l.16-.16z" })
], -1));
const _hoisted_4 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("label", { for: "file-upload" }, [
  /* @__PURE__ */ createBaseVNode("span", { class: "file-upload" }, "Select image")
], -1));
const _hoisted_5 = { class: "flex-container" };
const _hoisted_6 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "150px" } }, "Sampler", -1));
const _hoisted_7 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using Euler A for the best results (but it also takes more time). ", -1));
const _hoisted_8 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("a", {
  target: "_blank",
  href: "https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
}, "Learn more", -1));
const _hoisted_9 = { class: "flex-container" };
const _hoisted_10 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1));
const _hoisted_11 = { class: "flex-container" };
const _hoisted_12 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1));
const _hoisted_13 = { class: "flex-container" };
const _hoisted_14 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1));
const _hoisted_15 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1));
const _hoisted_16 = { class: "flex-container" };
const _hoisted_17 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "CFG Scale", -1));
const _hoisted_18 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 3-15 for most images.", -1));
const _hoisted_19 = { class: "flex-container" };
const _hoisted_20 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Count", -1));
const _hoisted_21 = { class: "flex-container" };
const _hoisted_22 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1));
const _hoisted_23 = { class: "flex-container" };
const _hoisted_24 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Seed", -1));
const _hoisted_25 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "For random seed use -1.", -1));
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "Inpainting",
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
    const generate = () => {
      if (conf.data.settings.inpainting.seed === null) {
        messageHandler.error("Please set a seed");
        return;
      }
      generateMask();
      global.state.generating = true;
      fetch(`${serverUrl}/api/generate/inpainting`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          data: {
            prompt: conf.data.settings.inpainting.prompt,
            image: conf.data.settings.inpainting.image,
            mask_image: conf.data.settings.inpainting.maskImage,
            id: v4(),
            negative_prompt: conf.data.settings.inpainting.negativePrompt,
            width: conf.data.settings.inpainting.width,
            height: conf.data.settings.inpainting.height,
            steps: conf.data.settings.inpainting.steps,
            guidance_scale: conf.data.settings.inpainting.cfgScale,
            seed: checkSeed(conf.data.settings.inpainting.seed),
            batch_size: conf.data.settings.inpainting.batchSize,
            batch_count: conf.data.settings.inpainting.batchCount,
            scheduler: conf.data.settings.inpainting.sampler
          },
          model: conf.data.settings.model
        })
      }).then((res) => {
        global.state.generating = false;
        res.json().then((data) => {
          global.state.inpainting.images = data.images;
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
    const canvas = ref();
    const maskCanvas = ref();
    const width = ref(512);
    const height = ref(512);
    const strokeWidth = ref(10);
    const eraser = ref(false);
    const preview = ref("");
    const imageContainer = ref();
    function previewImage(event) {
      const input = event.target;
      if (input.files) {
        const reader = new FileReader();
        reader.onload = (e) => {
          var _a;
          const i = (_a = e.target) == null ? void 0 : _a.result;
          if (i) {
            const s = i.toString();
            preview.value = s;
            const img = new Image();
            img.src = s;
            img.onload = () => {
              var _a2, _b;
              const containerWidth = (_a2 = imageContainer.value) == null ? void 0 : _a2.clientWidth;
              if (containerWidth === void 0)
                return;
              const containerScaledWidth = containerWidth;
              const containerScaledHeight = img.height * containerScaledWidth / img.width;
              const screenHeight = window.innerHeight;
              const screenHeightScaledHeight = containerScaledHeight * 0.7 * screenHeight / containerScaledHeight;
              const screenHeightScaledWidth = img.width * screenHeightScaledHeight / img.height;
              if (containerScaledWidth < screenHeightScaledWidth) {
                width.value = containerScaledWidth;
                height.value = containerScaledHeight;
              } else {
                width.value = screenHeightScaledWidth;
                height.value = screenHeightScaledHeight;
              }
              conf.data.settings.inpainting.image = s;
              (_b = canvas.value) == null ? void 0 : _b.redraw(false);
            };
          }
        };
        reader.readAsDataURL(input.files[0]);
      }
    }
    async function clearCanvas() {
      var _a;
      (_a = canvas.value) == null ? void 0 : _a.reset();
    }
    function undo() {
      var _a;
      (_a = canvas.value) == null ? void 0 : _a.undo();
    }
    function redo() {
      var _a;
      (_a = canvas.value) == null ? void 0 : _a.redo();
    }
    function toggleEraser() {
      console.log(eraser.value);
      eraser.value = !eraser.value;
      console.log(eraser.value);
    }
    function generateMask() {
      var _a;
      const x = (_a = canvas.value) == null ? void 0 : _a.getAllStrokes();
      if (maskCanvas.value !== void 0) {
        maskCanvas.value.images = x;
        maskCanvas.value.redraw(true);
      }
    }
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NGrid), {
          cols: "1 850:2",
          "x-gap": "12"
        }, {
          default: withCtx(() => [
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(unref(NCard), { title: "Input image" }, {
                  default: withCtx(() => [
                    createBaseVNode("div", {
                      class: "image-container",
                      ref_key: "imageContainer",
                      ref: imageContainer
                    }, [
                      createVNode(unref(VueDrawingCanvas), {
                        width: width.value,
                        height: height.value,
                        backgroundImage: preview.value,
                        lineWidth: strokeWidth.value,
                        strokeType: "dash",
                        lineCap: "round",
                        lineJoin: "round",
                        fillShape: false,
                        eraser: eraser.value,
                        color: "black",
                        ref_key: "canvas",
                        ref: canvas,
                        saveAs: "png",
                        "canvas-id": "VueDrawingCanvas1"
                      }, null, 8, ["width", "height", "backgroundImage", "lineWidth", "eraser"]),
                      createVNode(unref(VueDrawingCanvas), {
                        image: unref(conf).data.settings.inpainting.maskImage,
                        "onUpdate:image": _cache[0] || (_cache[0] = ($event) => unref(conf).data.settings.inpainting.maskImage = $event),
                        width: width.value,
                        height: height.value,
                        ref_key: "maskCanvas",
                        ref: maskCanvas,
                        saveAs: "png",
                        style: { "display": "none" },
                        "canvas-id": "VueDrawingCanvas2"
                      }, null, 8, ["image", "width", "height"])
                    ], 512),
                    createVNode(unref(NSpace), {
                      inline: "",
                      justify: "space-between",
                      align: "center",
                      style: { "width": "100%", "margin-top": "12px" }
                    }, {
                      default: withCtx(() => [
                        createBaseVNode("div", _hoisted_2, [
                          createVNode(unref(NButton), {
                            class: "utility-button",
                            onClick: undo
                          }, {
                            default: withCtx(() => [
                              createVNode(unref(NIcon), null, {
                                default: withCtx(() => [
                                  createVNode(unref(ArrowUndoSharp))
                                ]),
                                _: 1
                              })
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NButton), {
                            class: "utility-button",
                            onClick: redo
                          }, {
                            default: withCtx(() => [
                              createVNode(unref(NIcon), null, {
                                default: withCtx(() => [
                                  createVNode(unref(ArrowRedoSharp))
                                ]),
                                _: 1
                              })
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NButton), {
                            class: "utility-button",
                            onClick: toggleEraser
                          }, {
                            default: withCtx(() => [
                              eraser.value ? (openBlock(), createBlock(unref(NIcon), { key: 0 }, {
                                default: withCtx(() => [
                                  _hoisted_3
                                ]),
                                _: 1
                              })) : (openBlock(), createBlock(unref(NIcon), { key: 1 }, {
                                default: withCtx(() => [
                                  createVNode(unref(BrushSharp))
                                ]),
                                _: 1
                              }))
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NButton), {
                            class: "utility-button",
                            onClick: clearCanvas
                          }, {
                            default: withCtx(() => [
                              createVNode(unref(NIcon), null, {
                                default: withCtx(() => [
                                  createVNode(unref(TrashBinSharp))
                                ]),
                                _: 1
                              })
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: strokeWidth.value,
                            "onUpdate:value": _cache[1] || (_cache[1] = ($event) => strokeWidth.value = $event),
                            min: 1,
                            max: 50,
                            step: 1,
                            style: { "width": "100px", "margin": "0 8px" }
                          }, null, 8, ["value"]),
                          createBaseVNode("p", null, toDisplayString(width.value) + "x" + toDisplayString(height.value), 1)
                        ]),
                        _hoisted_4
                      ]),
                      _: 1
                    }),
                    createBaseVNode("input", {
                      type: "file",
                      accept: "image/*",
                      onChange: previewImage,
                      id: "file-upload",
                      class: "hidden-input"
                    }, null, 32)
                  ]),
                  _: 1
                }),
                createVNode(unref(NCard), { title: "Settings" }, {
                  default: withCtx(() => [
                    createVNode(unref(NSpace), {
                      vertical: "",
                      class: "left-container"
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NInput), {
                          value: unref(conf).data.settings.inpainting.prompt,
                          "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(conf).data.settings.inpainting.prompt = $event),
                          type: "textarea",
                          placeholder: "Prompt"
                        }, null, 8, ["value"]),
                        createVNode(unref(NInput), {
                          value: unref(conf).data.settings.inpainting.negativePrompt,
                          "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(conf).data.settings.inpainting.negativePrompt = $event),
                          type: "textarea",
                          placeholder: "Negative prompt"
                        }, null, 8, ["value"]),
                        createBaseVNode("div", _hoisted_5, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_6
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" The sampler is the method used to generate the image. Your result may vary drastically depending on the sampler you choose. "),
                              _hoisted_7,
                              _hoisted_8
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSelect), {
                            options: unref(conf).scheduler_options,
                            value: unref(conf).data.settings.inpainting.sampler,
                            "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(conf).data.settings.inpainting.sampler = $event),
                            style: { "flex-grow": "1" }
                          }, null, 8, ["options", "value"])
                        ]),
                        createBaseVNode("div", _hoisted_9, [
                          _hoisted_10,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.inpainting.width,
                            "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(conf).data.settings.inpainting.width = $event),
                            min: 128,
                            max: 2048,
                            step: 8,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.inpainting.width,
                            "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(conf).data.settings.inpainting.width = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            step: 8,
                            min: 128,
                            max: 2048
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_11, [
                          _hoisted_12,
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.inpainting.height,
                            "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(conf).data.settings.inpainting.height = $event),
                            min: 128,
                            max: 2048,
                            step: 8,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.inpainting.height,
                            "onUpdate:value": _cache[8] || (_cache[8] = ($event) => unref(conf).data.settings.inpainting.height = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            step: 8,
                            min: 128,
                            max: 2048
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_13, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_14
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                              _hoisted_15
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.inpainting.steps,
                            "onUpdate:value": _cache[9] || (_cache[9] = ($event) => unref(conf).data.settings.inpainting.steps = $event),
                            min: 5,
                            max: 300,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.inpainting.steps,
                            "onUpdate:value": _cache[10] || (_cache[10] = ($event) => unref(conf).data.settings.inpainting.steps = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 5,
                            max: 300
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_16, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_17
                            ]),
                            default: withCtx(() => [
                              createTextVNode(' Guidance scale indicates how much should model stay close to the prompt. Higher values might be exactly what you want, but generated images might have some artefacts. Lower values indicates that model can "dream" about this prompt more. '),
                              _hoisted_18
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.inpainting.cfgScale,
                            "onUpdate:value": _cache[11] || (_cache[11] = ($event) => unref(conf).data.settings.inpainting.cfgScale = $event),
                            min: 1,
                            max: 30,
                            step: 0.5,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value", "step"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.inpainting.cfgScale,
                            "onUpdate:value": _cache[12] || (_cache[12] = ($event) => unref(conf).data.settings.inpainting.cfgScale = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 30,
                            step: 0.5
                          }, null, 8, ["value", "step"])
                        ]),
                        createBaseVNode("div", _hoisted_19, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_20
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of images to generate after each other. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.inpainting.batchCount,
                            "onUpdate:value": _cache[13] || (_cache[13] = ($event) => unref(conf).data.settings.inpainting.batchCount = $event),
                            min: 1,
                            max: 9,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.inpainting.batchCount,
                            "onUpdate:value": _cache[14] || (_cache[14] = ($event) => unref(conf).data.settings.inpainting.batchCount = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 9
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_21, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_22
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Number of images to generate in paralel. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.inpainting.batchSize,
                            "onUpdate:value": _cache[15] || (_cache[15] = ($event) => unref(conf).data.settings.inpainting.batchSize = $event),
                            min: 1,
                            max: 9,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.inpainting.batchSize,
                            "onUpdate:value": _cache[16] || (_cache[16] = ($event) => unref(conf).data.settings.inpainting.batchSize = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 9
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_23, [
                          createVNode(unref(NTooltip), { "max-width": 600 }, {
                            trigger: withCtx(() => [
                              _hoisted_24
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Seed is a number that represents the starting canvas of your image. If you want to create the same image as your friend, you can use the same settings and seed to do so. "),
                              _hoisted_25
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.inpainting.seed,
                            "onUpdate:value": _cache[17] || (_cache[17] = ($event) => unref(conf).data.settings.inpainting.seed = $event),
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
                createVNode(_sfc_main$5, { generate }),
                createVNode(ImageOutput, {
                  "current-image": unref(global).state.inpainting.currentImage,
                  images: unref(global).state.inpainting.images
                }, null, 8, ["current-image", "images"])
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
const Inpainting_vue_vue_type_style_index_0_scoped_ba9c9a75_lang = "";
const Inpainting = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["__scopeId", "data-v-ba9c9a75"]]);
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
          createVNode(unref(NTabPane), { name: "Inpainting" }, {
            default: withCtx(() => [
              createVNode(Inpainting)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Image variations" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$3)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "SD Upscale" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$6)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Depth to Image" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$6)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Pix to Pix" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$6)
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
