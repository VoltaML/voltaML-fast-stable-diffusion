import { d as defineComponent, o as openBlock, e as createElementBlock, l as createBaseVNode, A as ref, c as computed, b8 as onMounted, p as createBlock, w as withCtx, bs as withModifiers, f as createVNode, g as unref, C as NIcon, t as toDisplayString, h as NCard, x as pushScopeId, y as popScopeId, _ as _export_sfc } from "./index.js";
const _hoisted_1$1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M473.66 210c-14-10.38-31.2-18-49.36-22.11a16.11 16.11 0 0 1-12.19-12.22c-7.8-34.75-24.59-64.55-49.27-87.13C334.15 62.25 296.21 47.79 256 47.79c-35.35 0-68 11.08-94.37 32.05a150.07 150.07 0 0 0-42.06 53a16 16 0 0 1-11.31 8.87c-26.75 5.4-50.9 16.87-69.34 33.12C13.46 197.33 0 227.24 0 261.39c0 34.52 14.49 66 40.79 88.76c25.12 21.69 58.94 33.64 95.21 33.64h104V230.42l-36.69 36.69a16 16 0 0 1-23.16-.56c-5.8-6.37-5.24-16.3.85-22.39l63.69-63.68a16 16 0 0 1 22.62 0L331 244.14c6.28 6.29 6.64 16.6.39 22.91a16 16 0 0 1-22.68.06L272 230.42v153.37h124c31.34 0 59.91-8.8 80.45-24.77c23.26-18.1 35.55-44 35.55-74.83c0-29.94-13.26-55.61-38.34-74.19z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$1 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M240 448.21a16 16 0 1 0 32 0v-64.42h-32z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$1 = [_hoisted_2$1, _hoisted_3$1];
const CloudUpload = defineComponent({
  name: "CloudUpload",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$1, _hoisted_4$1);
  }
});
const _withScopeId = (n) => (pushScopeId("data-v-4f5be896"), n = n(), popScopeId(), n);
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
      return image.value ? (_a = image.value) == null ? void 0 : _a.width : 0;
    });
    const height = computed(() => {
      var _a;
      return image.value ? (_a = image.value) == null ? void 0 : _a.height : 0;
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
      var _a, _b;
      console.log((_a = e.dataTransfer) == null ? void 0 : _a.files);
      if ((_b = e.dataTransfer) == null ? void 0 : _b.files) {
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
            createBaseVNode("p", null, toDisplayString(unref(width)) + "x" + toDisplayString(unref(height)), 1)
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
const ImageUpload_vue_vue_type_style_index_0_scoped_4f5be896_lang = "";
const ImageUpload = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-4f5be896"]]);
export {
  ImageUpload as I
};
