import { _ as _sfc_main$2 } from "./GenerateSection.vue_vue_type_script_setup_true_lang.js";
import { _ as _sfc_main$3 } from "./ImageOutput.vue_vue_type_script_setup_true_lang.js";
import { I as ImageUpload } from "./ImageUpload.js";
import { d as defineComponent, u as useState, a as useSettings, b as useMessage, e as openBlock, f as createElementBlock, g as createVNode, w as withCtx, h as unref, N as NGi, i as NCard, j as NSpace, n as createBaseVNode, r as NSelect, K as upscalerOptions, q as NTooltip, m as createTextVNode, v as NSlider, z as NGrid, s as serverUrl, B as pushScopeId, C as popScopeId, _ as _export_sfc, x as createBlock, H as NTabPane, I as NTabs } from "./index.js";
import { N as NInputNumber } from "./InputNumber.js";
import "./Image.js";
import "./CloudUpload.js";
const _withScopeId = (n) => (pushScopeId("data-v-630bfcb5"), n = n(), popScopeId(), n);
const _hoisted_1 = { style: { "margin": "0 12px" } };
const _hoisted_2 = { class: "flex-container" };
const _hoisted_3 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Model", -1));
const _hoisted_4 = { class: "flex-container" };
const _hoisted_5 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Scale Factor", -1));
const _hoisted_6 = { class: "flex-container" };
const _hoisted_7 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Tile Size", -1));
const _hoisted_8 = { class: "flex-container" };
const _hoisted_9 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Tile Padding", -1));
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "RealESRGAN",
  setup(__props) {
    const global = useState();
    const conf = useSettings();
    const messageHandler = useMessage();
    const imageSelectCallback = (base64Image) => {
      conf.data.settings.upscale.image = base64Image;
    };
    const generate = () => {
      global.state.generating = true;
      fetch(`${serverUrl}/api/generate/upscale`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          data: {
            image: conf.data.settings.upscale.image,
            upscale_factor: conf.data.settings.upscale.upscale_factor,
            model: conf.data.settings.upscale.model,
            tile_size: conf.data.settings.upscale.tile_size,
            tile_padding: conf.data.settings.upscale.tile_padding
          },
          model: conf.data.settings.upscale.model
        })
      }).then((res) => {
        global.state.generating = false;
        res.json().then((data) => {
          console.log(data);
          global.state.extra.images = [data.images];
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
          cols: "1 m:2",
          "x-gap": "12",
          responsive: "screen"
        }, {
          default: withCtx(() => [
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(ImageUpload, {
                  callback: imageSelectCallback,
                  preview: unref(conf).data.settings.upscale.image,
                  style: { "margin-bottom": "12px" },
                  onFileDropped: _cache[0] || (_cache[0] = ($event) => unref(conf).data.settings.upscale.image = $event)
                }, null, 8, ["preview"]),
                createVNode(unref(NCard), {
                  title: "Settings",
                  style: { "margin-bottom": "12px" }
                }, {
                  default: withCtx(() => [
                    createVNode(unref(NSpace), {
                      vertical: "",
                      class: "left-container"
                    }, {
                      default: withCtx(() => [
                        createBaseVNode("div", _hoisted_2, [
                          _hoisted_3,
                          createVNode(unref(NSelect), {
                            value: unref(conf).data.settings.upscale.model,
                            "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(conf).data.settings.upscale.model = $event),
                            style: { "margin-right": "12px" },
                            options: unref(upscalerOptions)
                          }, null, 8, ["value", "options"])
                        ]),
                        createBaseVNode("div", _hoisted_4, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_5
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" TODO ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.upscale.upscale_factor,
                            "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(conf).data.settings.upscale.upscale_factor = $event),
                            min: 1,
                            max: 4,
                            step: 0.1,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.upscale.upscale_factor,
                            "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(conf).data.settings.upscale.upscale_factor = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 1,
                            max: 4,
                            step: 0.1
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_6, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_7
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" How large each tile should be. Larger tiles will use more memory. 0 will disable tiling. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.upscale.tile_size,
                            "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(conf).data.settings.upscale.tile_size = $event),
                            min: 32,
                            max: 2048,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.upscale.tile_size,
                            "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(conf).data.settings.upscale.tile_size = $event),
                            size: "small",
                            min: 32,
                            max: 2048,
                            style: { "min-width": "96px", "width": "96px" }
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_8, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_9
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" How much should tiles overlap. Larger padding will use more memory, but image should not have visible seams. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.upscale.tile_padding,
                            "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(conf).data.settings.upscale.tile_padding = $event),
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.upscale.tile_padding,
                            "onUpdate:value": _cache[7] || (_cache[7] = ($event) => unref(conf).data.settings.upscale.tile_padding = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" }
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
                createVNode(_sfc_main$2, {
                  generate,
                  "do-not-disable-generate": ""
                }),
                createVNode(_sfc_main$3, {
                  "current-image": unref(global).state.extra.currentImage,
                  images: unref(global).state.extra.images,
                  onImageClicked: _cache[8] || (_cache[8] = ($event) => unref(global).state.extra.currentImage = $event)
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
const RealESRGAN_vue_vue_type_style_index_0_scoped_630bfcb5_lang = "";
const RealESRGAN = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["__scopeId", "data-v-630bfcb5"]]);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ExtraView",
  setup(__props) {
    const state = useState();
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NTabs), {
        type: "segment",
        value: unref(state).state.extra.tab,
        "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(state).state.extra.tab = $event)
      }, {
        default: withCtx(() => [
          createVNode(unref(NTabPane), { name: "Upscale" }, {
            default: withCtx(() => [
              createVNode(RealESRGAN)
            ]),
            _: 1
          })
        ]),
        _: 1
      }, 8, ["value"]);
    };
  }
});
export {
  _sfc_main as default
};
