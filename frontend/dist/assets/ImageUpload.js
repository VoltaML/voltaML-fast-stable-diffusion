import { d as defineComponent, A as ref, c as computed, ba as onMounted, o as openBlock, g as createBlock, w as withCtx, b as createBaseVNode, bd as withModifiers, a as createElementBlock, e as createVNode, f as unref, C as NIcon, D as toDisplayString, l as NCard, x as pushScopeId, y as popScopeId, _ as _export_sfc } from "./index.js";
import { C as CloudUpload } from "./CloudUpload.js";
const _withScopeId = (n) => (pushScopeId("data-v-9ed1514f"), n = n(), popScopeId(), n);
const _hoisted_1 = { class: "image-container" };
const _hoisted_2 = {
  for: "file-upload",
  style: { "width": "100%", "height": "100%", "cursor": "pointer" }
};
const _hoisted_3 = ["onDrop"];
const _hoisted_4 = ["src"];
const _hoisted_5 = {
  key: 1,
  style: { "margin-bottom": "12px", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%", "widows": "100%", "border": "1px dashed #666" }
};
const _hoisted_6 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { style: { "margin-left": "12px" } }, "Drag and drop or click to upload", -1));
const _hoisted_7 = { style: { "width": "100%", "display": "inline-flex", "align-items": "center", "justify-content": "space-between" } };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ImageUpload",
  props: {
    callback: {
      type: Function
    },
    preview: {
      type: String
    }
  },
  emits: ["file-dropped"],
  setup(__props, { emit }) {
    const props = __props;
    const image = ref();
    const width = computed(() => {
      var _a;
      return image.value ? (_a = image.value) == null ? void 0 : _a.naturalWidth : 0;
    });
    const height = computed(() => {
      var _a;
      return image.value ? (_a = image.value) == null ? void 0 : _a.naturalHeight : 0;
    });
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
              image.value = img;
            };
          }
        };
        reader.readAsDataURL(input.files[0]);
      }
    }
    function onDrop(e) {
      var _a;
      if ((_a = e.dataTransfer) == null ? void 0 : _a.files) {
        const reader = new FileReader();
        reader.onload = (e2) => {
          var _a2;
          const i = (_a2 = e2.target) == null ? void 0 : _a2.result;
          if (i) {
            const s = i.toString();
            if (props.callback) {
              props.callback(s);
            }
            const img = new Image();
            img.src = s;
            img.onload = () => {
              emit("file-dropped", s);
            };
          }
        };
        reader.readAsDataURL(e.dataTransfer.files[0]);
      }
    }
    function preventDefaults(e) {
      e.preventDefault();
    }
    const events = ["dragenter", "dragover", "dragleave", "drop"];
    onMounted(() => {
      events.forEach((eventName) => {
        document.body.addEventListener(eventName, preventDefaults);
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), { title: "Input image" }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1, [
            createBaseVNode("label", _hoisted_2, [
              createBaseVNode("span", {
                style: { "width": "100%", "height": "100%" },
                onDrop: withModifiers(onDrop, ["prevent"])
              }, [
                _ctx.$props.preview ? (openBlock(), createElementBlock("img", {
                  key: 0,
                  src: _ctx.$props.preview,
                  style: { "width": "100%" }
                }, null, 8, _hoisted_4)) : (openBlock(), createElementBlock("div", _hoisted_5, [
                  createVNode(unref(NIcon), {
                    size: "48",
                    depth: 3
                  }, {
                    default: withCtx(() => [
                      createVNode(unref(CloudUpload))
                    ]),
                    _: 1
                  }),
                  _hoisted_6
                ]))
              ], 40, _hoisted_3)
            ])
          ]),
          createBaseVNode("div", _hoisted_7, [
            createBaseVNode("p", null, toDisplayString(width.value) + "x" + toDisplayString(height.value), 1)
          ]),
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
const ImageUpload = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-9ed1514f"]]);
export {
  ImageUpload as I
};
