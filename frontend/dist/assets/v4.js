import { d as defineComponent, a as useSettings, u as useState, o as openBlock, p as createBlock, w as withCtx, f as createVNode, g as unref, N as NGi, B as NButton, k as createTextVNode, r as NGrid, h as NCard, q as createCommentVNode, F as router } from "./index.js";
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
let getRandomValues;
const rnds8 = new Uint8Array(16);
function rng() {
  if (!getRandomValues) {
    getRandomValues = typeof crypto !== "undefined" && crypto.getRandomValues && crypto.getRandomValues.bind(crypto);
    if (!getRandomValues) {
      throw new Error("crypto.getRandomValues() not supported. See https://github.com/uuidjs/uuid#getrandomvalues-not-supported");
    }
  }
  return getRandomValues(rnds8);
}
const byteToHex = [];
for (let i = 0; i < 256; ++i) {
  byteToHex.push((i + 256).toString(16).slice(1));
}
function unsafeStringify(arr, offset = 0) {
  return (byteToHex[arr[offset + 0]] + byteToHex[arr[offset + 1]] + byteToHex[arr[offset + 2]] + byteToHex[arr[offset + 3]] + "-" + byteToHex[arr[offset + 4]] + byteToHex[arr[offset + 5]] + "-" + byteToHex[arr[offset + 6]] + byteToHex[arr[offset + 7]] + "-" + byteToHex[arr[offset + 8]] + byteToHex[arr[offset + 9]] + "-" + byteToHex[arr[offset + 10]] + byteToHex[arr[offset + 11]] + byteToHex[arr[offset + 12]] + byteToHex[arr[offset + 13]] + byteToHex[arr[offset + 14]] + byteToHex[arr[offset + 15]]).toLowerCase();
}
const randomUUID = typeof crypto !== "undefined" && crypto.randomUUID && crypto.randomUUID.bind(crypto);
const native = {
  randomUUID
};
function v4(options, buf, offset) {
  if (native.randomUUID && !buf && !options) {
    return native.randomUUID();
  }
  options = options || {};
  const rnds = options.random || (options.rng || rng)();
  rnds[6] = rnds[6] & 15 | 64;
  rnds[8] = rnds[8] & 63 | 128;
  if (buf) {
    offset = offset || 0;
    for (let i = 0; i < 16; ++i) {
      buf[offset + i] = rnds[i];
    }
    return buf;
  }
  return unsafeStringify(rnds);
}
export {
  _sfc_main as _,
  v4 as v
};
