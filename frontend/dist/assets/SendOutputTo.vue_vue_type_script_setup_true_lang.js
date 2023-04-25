import { N as NDescriptionsItem, a as NDescriptions } from "./DescriptionsItem.js";
import { d as defineComponent, o as openBlock, p as createBlock, w as withCtx, f as createVNode, g as unref, k as createTextVNode, t as toDisplayString, h as NCard, q as createCommentVNode, a as useSettings, u as useState, N as NGi, B as NButton, r as NGrid, F as router } from "./index.js";
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "OutputStats",
  props: {
    genData: {
      type: Object,
      required: true
    }
  },
  setup(__props) {
    return (_ctx, _cache) => {
      return __props.genData.time_taken || __props.genData.seed ? (openBlock(), createBlock(unref(NCard), {
        key: 0,
        title: "Stats"
      }, {
        default: withCtx(() => [
          createVNode(unref(NDescriptions), null, {
            default: withCtx(() => [
              createVNode(unref(NDescriptionsItem), { label: "Total Time" }, {
                default: withCtx(() => [
                  createTextVNode(toDisplayString(__props.genData.time_taken) + "s ", 1)
                ]),
                _: 1
              }),
              createVNode(unref(NDescriptionsItem), { label: "Seed" }, {
                default: withCtx(() => [
                  createTextVNode(toDisplayString(__props.genData.seed), 1)
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
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "SendOutputTo",
  props: {
    output: {
      type: String,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
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
      conf.data.settings.realesrgan.image = props.output;
      state.state.extra.tab = "Upscale";
      await router.push("/extra");
    }
    return (_ctx, _cache) => {
      return __props.output ? (openBlock(), createBlock(unref(NCard), {
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
      })) : createCommentVNode("", true);
    };
  }
});
export {
  _sfc_main as _,
  _sfc_main$1 as a
};
