import { d as defineComponent, y as ref, a as useState, o as openBlock, c as createBlock, w as withCtx, f as unref, q as NGi, e as createVNode, A as NIcon, k as createTextVNode, z as NButton, h as createCommentVNode, r as NGrid, i as computed, b as createBaseVNode, g as createElementBlock, F as Fragment, L as renderList, M as NScrollbar, N as NCard } from "./index.js";
import { D as Download, _ as _sfc_main$2 } from "./SendOutputTo.vue_vue_type_script_setup_true_lang.js";
import { T as TrashBin, N as NImage } from "./TrashBin.js";
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "DownloadDelete",
  props: {
    imagePath: {
      type: String,
      required: false
    },
    base64image: {
      type: String,
      required: true
    }
  },
  setup(__props) {
    const showDeleteModal = ref(false);
    const global = useState();
    const props = __props;
    function downloadImage() {
      const a = document.createElement("a");
      a.href = props.base64image;
      a.download = global.state.imageBrowser.currentImage.id;
      a.target = "_blank";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NGrid), {
        cols: "2",
        "x-gap": "4",
        style: { "margin-top": "12px" }
      }, {
        default: withCtx(() => [
          props.base64image ? (openBlock(), createBlock(unref(NGi), { key: 0 }, {
            default: withCtx(() => [
              createVNode(unref(NButton), {
                type: "success",
                onClick: downloadImage,
                style: { "width": "100%" },
                ghost: ""
              }, {
                icon: withCtx(() => [
                  createVNode(unref(NIcon), null, {
                    default: withCtx(() => [
                      createVNode(unref(Download))
                    ]),
                    _: 1
                  })
                ]),
                default: withCtx(() => [
                  createTextVNode("Download")
                ]),
                _: 1
              })
            ]),
            _: 1
          })) : createCommentVNode("", true),
          createVNode(unref(NGi), null, {
            default: withCtx(() => [
              createVNode(unref(NButton), {
                type: "error",
                onClick: _cache[0] || (_cache[0] = ($event) => showDeleteModal.value = true),
                style: { "width": "100%" },
                ghost: "",
                disabled: props.imagePath === void 0
              }, {
                icon: withCtx(() => [
                  createVNode(unref(NIcon), null, {
                    default: withCtx(() => [
                      createVNode(unref(TrashBin))
                    ]),
                    _: 1
                  })
                ]),
                default: withCtx(() => [
                  createTextVNode(" Delete")
                ]),
                _: 1
              }, 8, ["disabled"])
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
  style: { "margin-top": "12px" }
};
const _hoisted_3 = ["onClick"];
const _hoisted_4 = ["src"];
const _hoisted_5 = { key: 1 };
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
    },
    data: {
      type: Object,
      required: false,
      default: () => ({})
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
          ])) : createCommentVNode("", true),
          props.currentImage ? (openBlock(), createElementBlock("div", _hoisted_5, [
            createVNode(unref(_sfc_main$1), {
              base64image: props.currentImage,
              style: { "margin-bottom": "4px" }
            }, null, 8, ["base64image"]),
            createVNode(unref(_sfc_main$2), {
              output: props.currentImage,
              card: false,
              data: __props.data
            }, null, 8, ["output", "data"])
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
