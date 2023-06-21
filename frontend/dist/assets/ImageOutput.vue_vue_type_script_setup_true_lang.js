import { N as NImage } from "./Image.js";
import { d as defineComponent, c as computed, e as openBlock, x as createBlock, w as withCtx, n as createBaseVNode, h as unref, y as createCommentVNode, f as createElementBlock, g as createVNode, L as Fragment, M as renderList, O as NScrollbar, i as NCard } from "./index.js";
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
            displayedImage.value ? (openBlock(), createBlock(unref(NImage), {
              key: 0,
              src: displayedImage.value.toString(),
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
  _sfc_main as _
};
