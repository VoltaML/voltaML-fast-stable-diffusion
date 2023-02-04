import { d as defineComponent, D as ref, o as openBlock, r as createBlock, w as withCtx, j as createBaseVNode, e as createVNode, ar as toDisplayString, f as unref, h as NSpace, g as NCard, z as pushScopeId, A as popScopeId, B as _export_sfc } from "./index.js";
const _withScopeId = (n) => (pushScopeId("data-v-19b3e0b6"), n = n(), popScopeId(), n);
const _hoisted_1 = { class: "image-container" };
const _hoisted_2 = ["src"];
const _hoisted_3 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("label", { for: "file-upload" }, [
  /* @__PURE__ */ createBaseVNode("span", { class: "file-upload" }, "Select image")
], -1));
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ImageUpload",
  props: {
    callback: {
      type: Object
    },
    preview: {
      type: String
    }
  },
  setup(__props) {
    const props = __props;
    const width = ref(0);
    const height = ref(0);
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
              width.value = img.width;
              height.value = img.height;
            };
          }
        };
        reader.readAsDataURL(input.files[0]);
      }
    }
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), { title: "Input image" }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1, [
            createBaseVNode("img", {
              src: _ctx.$props.preview,
              style: { "width": "400px", "height": "auto" }
            }, null, 8, _hoisted_2)
          ]),
          createVNode(unref(NSpace), {
            inline: "",
            justify: "space-between",
            align: "center",
            style: { "width": "100%" }
          }, {
            default: withCtx(() => [
              createBaseVNode("p", null, toDisplayString(width.value) + "x" + toDisplayString(height.value), 1),
              _hoisted_3
            ]),
            _: 1
          }),
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
const ImageUpload_vue_vue_type_style_index_0_scoped_19b3e0b6_lang = "";
const ImageUpload = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-19b3e0b6"]]);
export {
  ImageUpload as I
};
