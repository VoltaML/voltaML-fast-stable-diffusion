import { d as defineComponent, u as useState, o as openBlock, p as createBlock, w as withCtx, f as createVNode, g as unref, N as NGi, B as NButton, k as createTextVNode, r as NGrid, h as NCard, v as serverUrl, c as computed, l as createBaseVNode, q as createCommentVNode, e as createElementBlock, I as Fragment, J as renderList, K as NScrollbar } from "./index.js";
import { N as NImage } from "./Image.js";
const _2img = "";
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "GenerateSection",
  props: {
    generate: {
      type: Function,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const global = useState();
    function interrupt() {
      fetch(`${serverUrl}/api/general/interrupt`, {
        method: "POST"
      }).then((res) => {
        if (res.status === 200) {
          global.state.generating = false;
        }
      });
    }
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), { style: { "margin-bottom": "12px" } }, {
        default: withCtx(() => [
          createVNode(unref(NGrid), {
            cols: "2",
            "x-gap": "24"
          }, {
            default: withCtx(() => [
              createVNode(unref(NGi), null, {
                default: withCtx(() => [
                  createVNode(unref(NButton), {
                    type: "success",
                    onClick: props.generate,
                    disabled: unref(global).state.generating,
                    loading: unref(global).state.generating,
                    style: { "width": "100%" },
                    ghost: ""
                  }, {
                    default: withCtx(() => [
                      createTextVNode("Generate")
                    ]),
                    _: 1
                  }, 8, ["onClick", "disabled", "loading"])
                ]),
                _: 1
              }),
              createVNode(unref(NGi), null, {
                default: withCtx(() => [
                  createVNode(unref(NButton), {
                    type: "error",
                    onClick: interrupt,
                    style: { "width": "100%" },
                    ghost: ""
                  }, {
                    default: withCtx(() => [
                      createTextVNode("Interrupt")
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
      });
    };
  }
});
const _hoisted_1 = { style: { "width": "100%", "display": "flex", "justify-content": "center" } };
const _hoisted_2 = {
  key: 0,
  style: { "height": "150px", "margin-top": "12px" }
};
const _hoisted_3 = ["onClick"];
const _hoisted_4 = ["src"];
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ImageOutput",
  props: {
    currentImage: {
      type: String,
      required: true
    },
    images: {
      type: Array,
      required: false,
      default: () => []
    }
  },
  emits: ["image-clicked"],
  setup(__props) {
    const props = __props;
    const displayedImage = computed(() => {
      if (props.currentImage) {
        return props.currentImage;
      } else if (props.images.length > 0) {
        return props.images[0];
      } else {
        return "";
      }
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), {
        title: "Output",
        hoverable: ""
      }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1, [
            unref(displayedImage) ? (openBlock(), createBlock(unref(NImage), {
              key: 0,
              src: unref(displayedImage).toString(),
              "img-props": {
                style: "max-width: 100%; max-height: 70vh; width: 100%"
              },
              style: { "max-width": "100%", "max-height": "60vh", "width": "100%", "height": "100%" },
              "object-fit": "contain"
            }, null, 8, ["src"])) : createCommentVNode("", true)
          ]),
          __props.images.length > 1 ? (openBlock(), createElementBlock("div", _hoisted_2, [
            createVNode(unref(NScrollbar), { "x-scrollable": "" }, {
              default: withCtx(() => [
                (openBlock(true), createElementBlock(Fragment, null, renderList(props.images, (image, i) => {
                  return openBlock(), createElementBlock("span", {
                    key: i,
                    onClick: ($event) => _ctx.$emit("image-clicked", image.toString()),
                    style: { "cursor": "pointer" }
                  }, [
                    createBaseVNode("img", {
                      src: image.toString(),
                      style: { "height": "100px", "width": "100px", "margin": "5px", "object-fit": "contain" }
                    }, null, 8, _hoisted_4)
                  ], 8, _hoisted_3);
                }), 128))
              ]),
              _: 1
            })
          ])) : createCommentVNode("", true)
        ]),
        _: 1
      });
    };
  }
});
export {
  _sfc_main$1 as _,
  _sfc_main as a
};
