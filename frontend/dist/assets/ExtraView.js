import { d as defineComponent, u as useState, b as useSettings, e as useMessage, o as openBlock, c as createElementBlock, g as createVNode, w as withCtx, h as unref, N as NGi, D as NButton, l as createTextVNode, i as NCard, j as NSpace, a as createBaseVNode, m as NTooltip, p as NSlider, q as NInputNumber, _ as _sfc_main$2, I as ImageOutput, r as NGrid, v as serverUrl, y as pushScopeId, z as popScopeId, A as _export_sfc, F as createBlock, G as NTabPane, H as NTabs } from "./index.js";
import { I as ImageUpload } from "./ImageUpload.js";
const _withScopeId = (n) => (pushScopeId("data-v-cba9f72b"), n = n(), popScopeId(), n);
const _hoisted_1 = { style: { "margin": "0 12px" } };
const _hoisted_2 = { class: "flex-container" };
const _hoisted_3 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Scale Factor", -1));
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "RealESRGAN",
  setup(__props) {
    const global = useState();
    const conf = useSettings();
    const messageHandler = useMessage();
    const imageSelectCallback = (base64Image) => {
      conf.data.settings.realesrgan.image = base64Image;
    };
    const loadUpscaler = () => {
      const url = new URL(`${serverUrl}/api/models/load`);
      url.searchParams.append("model", conf.data.settings.realesrgan.model);
      url.searchParams.append("backend", "PyTorch");
      fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        }
      }).then((res) => {
        res.json().then((data) => {
          if (data.error) {
            messageHandler.error(data.error);
          } else {
            messageHandler.success("Upscaler loaded");
          }
        });
      }).catch((err) => {
        messageHandler.error(err);
        console.log(err);
      });
    };
    const generate = () => {
      global.state.generating = true;
      fetch(`${serverUrl}/api/generate/realesrgan-upscale`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          data: {
            image: conf.data.settings.realesrgan.image,
            scale_factor: conf.data.settings.realesrgan.scale_factor,
            model: conf.data.settings.realesrgan.model
          },
          model: conf.data.settings.realesrgan.model
        })
      }).then((res) => {
        global.state.generating = false;
        res.json().then((data) => {
          console.log(data);
          global.state.realesrgan.images = data.images;
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
                createVNode(unref(NButton), { onClick: loadUpscaler }, {
                  default: withCtx(() => [
                    createTextVNode(" Load Upscaler ")
                  ]),
                  _: 1
                }),
                createVNode(ImageUpload, {
                  callback: imageSelectCallback,
                  preview: unref(conf).data.settings.realesrgan.image,
                  style: { "margin-bottom": "12px" }
                }, null, 8, ["preview"]),
                createVNode(unref(NCard), { title: "Settings" }, {
                  default: withCtx(() => [
                    createVNode(unref(NSpace), {
                      vertical: "",
                      class: "left-container"
                    }, {
                      default: withCtx(() => [
                        createBaseVNode("div", _hoisted_2, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_3
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" TODO ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.realesrgan.scale_factor,
                            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => unref(conf).data.settings.realesrgan.scale_factor = $event),
                            min: 2,
                            max: 4,
                            style: { "margin-right": "12px" }
                          }, null, 8, ["value"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.realesrgan.scale_factor,
                            "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(conf).data.settings.realesrgan.scale_factor = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 2,
                            max: 4
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
                createVNode(ImageOutput, {
                  "current-image": unref(global).state.realesrgan.currentImage,
                  images: unref(global).state.realesrgan.images
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
const RealESRGAN_vue_vue_type_style_index_0_scoped_cba9f72b_lang = "";
const RealESRGAN = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["__scopeId", "data-v-cba9f72b"]]);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ExtraView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NTabs), { type: "segment" }, {
        default: withCtx(() => [
          createVNode(unref(NTabPane), { name: "Real-ESRGAN" }, {
            default: withCtx(() => [
              createVNode(RealESRGAN)
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
