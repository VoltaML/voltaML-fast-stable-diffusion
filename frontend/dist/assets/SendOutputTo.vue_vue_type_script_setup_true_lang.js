import { d as defineComponent, e as openBlock, f as createElementBlock, n as createBaseVNode, bq as useRouter, a as useSettings, u as useState, x as createBlock, w as withCtx, g as createVNode, h as unref, N as NGi, G as NButton, m as createTextVNode, z as NGrid, i as NCard, y as createCommentVNode } from "./index.js";
const _hoisted_1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M376 160H272v153.37l52.69-52.68a16 16 0 0 1 22.62 22.62l-80 80a16 16 0 0 1-22.62 0l-80-80a16 16 0 0 1 22.62-22.62L240 313.37V160H136a56.06 56.06 0 0 0-56 56v208a56.06 56.06 0 0 0 56 56h240a56.06 56.06 0 0 0 56-56V216a56.06 56.06 0 0 0-56-56z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M272 48a16 16 0 0 0-32 0v112h32z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4 = [_hoisted_2, _hoisted_3];
const Download = defineComponent({
  name: "Download",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1, _hoisted_4);
  }
});
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "SendOutputTo",
  props: {
    output: {
      type: String,
      required: true
    },
    card: {
      type: Boolean,
      default: true
    }
  },
  setup(__props) {
    const props = __props;
    const router = useRouter();
    const conf = useSettings();
    const state = useState();
    async function toImg2Img() {
      conf.data.settings.img2img.image = props.output;
      state.state.img2img.tab = "Image to Image";
      await router.push("/image2image");
    }
    async function toControlNet() {
      conf.data.settings.controlnet.image = props.output;
      state.state.img2img.tab = "ControlNet";
      await router.push("/image2image");
    }
    async function toInpainting() {
      conf.data.settings.inpainting.image = props.output;
      state.state.img2img.tab = "Inpainting";
      await router.push("/image2image");
    }
    async function toUpscale() {
      conf.data.settings.upscale.image = props.output;
      state.state.extra.tab = "Upscale";
      await router.push("/extra");
    }
    async function toTagger() {
      conf.data.settings.tagger.image = props.output;
      await router.push("/tagger");
    }
    return (_ctx, _cache) => {
      return __props.output && __props.card ? (openBlock(), createBlock(unref(NCard), {
        key: 0,
        style: { "margin": "12px 0" },
        title: "Send To"
      }, {
        default: withCtx(() => [
          createVNode(unref(NGrid), {
            cols: "4",
            "x-gap": "4",
            "y-gap": "4"
          }, {
            default: withCtx(() => [
              createVNode(unref(NGi), null, {
                default: withCtx(() => [
                  createVNode(unref(NButton), {
                    type: "default",
                    onClick: toImg2Img,
                    style: { "width": "100%" },
                    ghost: ""
                  }, {
                    default: withCtx(() => [
                      createTextVNode("Img2Img")
                    ]),
                    _: 1
                  })
                ]),
                _: 1
              }),
              createVNode(unref(NGi), null, {
                default: withCtx(() => [
                  createVNode(unref(NButton), {
                    type: "default",
                    onClick: toControlNet,
                    style: { "width": "100%" },
                    ghost: ""
                  }, {
                    default: withCtx(() => [
                      createTextVNode("ControlNet")
                    ]),
                    _: 1
                  })
                ]),
                _: 1
              }),
              createVNode(unref(NGi), null, {
                default: withCtx(() => [
                  createVNode(unref(NButton), {
                    type: "default",
                    onClick: toInpainting,
                    style: { "width": "100%" },
                    ghost: ""
                  }, {
                    default: withCtx(() => [
                      createTextVNode("Inpainting")
                    ]),
                    _: 1
                  })
                ]),
                _: 1
              }),
              createVNode(unref(NGi), null, {
                default: withCtx(() => [
                  createVNode(unref(NButton), {
                    type: "default",
                    onClick: toUpscale,
                    style: { "width": "100%" },
                    ghost: ""
                  }, {
                    default: withCtx(() => [
                      createTextVNode("Upscale")
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
        _: 1
      })) : __props.output ? (openBlock(), createBlock(unref(NGrid), {
        key: 1,
        cols: "3",
        "x-gap": "4",
        "y-gap": "4"
      }, {
        default: withCtx(() => [
          createVNode(unref(NGi), null, {
            default: withCtx(() => [
              createVNode(unref(NButton), {
                type: "default",
                onClick: toImg2Img,
                style: { "width": "100%" },
                ghost: ""
              }, {
                default: withCtx(() => [
                  createTextVNode("Img2Img")
                ]),
                _: 1
              })
            ]),
            _: 1
          }),
          createVNode(unref(NGi), null, {
            default: withCtx(() => [
              createVNode(unref(NButton), {
                type: "default",
                onClick: toControlNet,
                style: { "width": "100%" },
                ghost: ""
              }, {
                default: withCtx(() => [
                  createTextVNode("ControlNet")
                ]),
                _: 1
              })
            ]),
            _: 1
          }),
          createVNode(unref(NGi), null, {
            default: withCtx(() => [
              createVNode(unref(NButton), {
                type: "default",
                onClick: toInpainting,
                style: { "width": "100%" },
                ghost: ""
              }, {
                default: withCtx(() => [
                  createTextVNode("Inpainting")
                ]),
                _: 1
              })
            ]),
            _: 1
          }),
          createVNode(unref(NGi), null, {
            default: withCtx(() => [
              createVNode(unref(NButton), {
                type: "default",
                onClick: toUpscale,
                style: { "width": "100%" },
                ghost: ""
              }, {
                default: withCtx(() => [
                  createTextVNode("Upscale")
                ]),
                _: 1
              })
            ]),
            _: 1
          }),
          createVNode(unref(NGi), null, {
            default: withCtx(() => [
              createVNode(unref(NButton), {
                type: "default",
                onClick: toTagger,
                style: { "width": "100%" },
                ghost: ""
              }, {
                default: withCtx(() => [
                  createTextVNode("Tagger")
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
export {
  Download as D,
  _sfc_main as _
};
