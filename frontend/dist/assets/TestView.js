import { V as VueDrawingCanvas, A as ArrowUndoSharp, a as ArrowRedoSharp, B as BrushSharp, T as TrashBinSharp } from "./vue-drawing-canvas.esm.js";
import { d as defineComponent, r as ref, c as createBlock, w as withCtx, u as unref, o as openBlock, a as createBaseVNode, b as createVNode, B as NButton, C as NIcon, q as NSlider, N as NSpace, e as NCard, p as pushScopeId, f as popScopeId, _ as _export_sfc } from "./index.js";
const _withScopeId = (n) => (pushScopeId("data-v-b8202046"), n = n(), popScopeId(), n);
const _hoisted_1 = { style: { "display": "inline-flex", "align-items": "center" } };
const _hoisted_2 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("svg", {
  xmlns: "http://www.w3.org/2000/svg",
  width: "16",
  height: "16",
  fill: "currentColor",
  class: "bi bi-eraser",
  viewBox: "0 0 16 16"
}, [
  /* @__PURE__ */ createBaseVNode("path", { d: "M8.086 2.207a2 2 0 0 1 2.828 0l3.879 3.879a2 2 0 0 1 0 2.828l-5.5 5.5A2 2 0 0 1 7.879 15H5.12a2 2 0 0 1-1.414-.586l-2.5-2.5a2 2 0 0 1 0-2.828l6.879-6.879zm2.121.707a1 1 0 0 0-1.414 0L4.16 7.547l5.293 5.293 4.633-4.633a1 1 0 0 0 0-1.414l-3.879-3.879zM8.746 13.547 3.453 8.254 1.914 9.793a1 1 0 0 0 0 1.414l2.5 2.5a1 1 0 0 0 .707.293H7.88a1 1 0 0 0 .707-.293l.16-.16z" })
], -1));
const _hoisted_3 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("label", { for: "file-upload" }, [
  /* @__PURE__ */ createBaseVNode("span", { class: "file-upload" }, "Select image")
], -1));
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TestView",
  setup(__props) {
    const canvas = ref();
    const width = ref(512);
    const height = ref(512);
    const strokeWidth = ref(10);
    const eraser = ref(false);
    const preview = ref("");
    const imageContainer = ref();
    function previewImage(event) {
      const input = event.target;
      if (input.files) {
        const reader = new FileReader();
        reader.onload = (e) => {
          var _a;
          const i = (_a = e.target) == null ? void 0 : _a.result;
          if (i) {
            const s = i.toString();
            preview.value = s;
            const img = new Image();
            img.src = s;
            img.onload = () => {
              var _a2, _b;
              const containerWidth = (_a2 = imageContainer.value) == null ? void 0 : _a2.clientWidth;
              const containerScaledWidth = containerWidth || img.width;
              const containerScaledHeight = img.height * containerScaledWidth / containerScaledWidth;
              const screenHeight = window.innerHeight;
              const screenHeightScaledHeight = containerScaledHeight * 0.7 * screenHeight / containerScaledHeight;
              const screenHeightScaledWidth = img.width * screenHeightScaledHeight / img.height;
              if (containerScaledWidth < screenHeightScaledWidth) {
                width.value = containerScaledWidth;
                height.value = containerScaledHeight;
              } else {
                width.value = screenHeightScaledWidth;
                height.value = screenHeightScaledHeight;
              }
              (_b = canvas.value) == null ? void 0 : _b.redraw(false);
            };
          }
        };
        reader.readAsDataURL(input.files[0]);
      }
    }
    async function clearCanvas() {
      var _a;
      (_a = canvas.value) == null ? void 0 : _a.reset();
    }
    function undo() {
      var _a;
      (_a = canvas.value) == null ? void 0 : _a.undo();
    }
    function redo() {
      var _a;
      (_a = canvas.value) == null ? void 0 : _a.redo();
    }
    function toggleEraser() {
      console.log(eraser.value);
      eraser.value = !eraser.value;
      console.log(eraser.value);
    }
    const image = ref("");
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), { title: "Input image" }, {
        default: withCtx(() => [
          createBaseVNode("div", {
            class: "image-container",
            ref_key: "imageContainer",
            ref: imageContainer
          }, [
            createVNode(unref(VueDrawingCanvas), {
              image: image.value,
              "onUpdate:image": _cache[0] || (_cache[0] = ($event) => image.value = $event),
              width: width.value,
              height: height.value,
              backgroundImage: preview.value,
              lineWidth: strokeWidth.value,
              strokeType: "dash",
              lineCap: "round",
              lineJoin: "round",
              fillShape: false,
              eraser: eraser.value,
              color: "black",
              ref_key: "canvas",
              ref: canvas,
              saveAs: "png"
            }, null, 8, ["image", "width", "height", "backgroundImage", "lineWidth", "eraser"])
          ], 512),
          createVNode(unref(NSpace), {
            inline: "",
            justify: "space-between",
            align: "center",
            style: { "width": "100%", "margin-top": "12px" }
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_1, [
                createVNode(unref(NButton), {
                  class: "utility-button",
                  onClick: undo
                }, {
                  default: withCtx(() => [
                    createVNode(unref(NIcon), null, {
                      default: withCtx(() => [
                        createVNode(unref(ArrowUndoSharp))
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                }),
                createVNode(unref(NButton), {
                  class: "utility-button",
                  onClick: redo
                }, {
                  default: withCtx(() => [
                    createVNode(unref(NIcon), null, {
                      default: withCtx(() => [
                        createVNode(unref(ArrowRedoSharp))
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                }),
                createVNode(unref(NButton), {
                  class: "utility-button",
                  onClick: toggleEraser
                }, {
                  default: withCtx(() => [
                    eraser.value ? (openBlock(), createBlock(unref(NIcon), { key: 0 }, {
                      default: withCtx(() => [
                        _hoisted_2
                      ]),
                      _: 1
                    })) : (openBlock(), createBlock(unref(NIcon), { key: 1 }, {
                      default: withCtx(() => [
                        createVNode(unref(BrushSharp))
                      ]),
                      _: 1
                    }))
                  ]),
                  _: 1
                }),
                createVNode(unref(NButton), {
                  class: "utility-button",
                  onClick: clearCanvas
                }, {
                  default: withCtx(() => [
                    createVNode(unref(NIcon), null, {
                      default: withCtx(() => [
                        createVNode(unref(TrashBinSharp))
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                }),
                createVNode(unref(NSlider), {
                  value: strokeWidth.value,
                  "onUpdate:value": _cache[1] || (_cache[1] = ($event) => strokeWidth.value = $event),
                  min: 1,
                  max: 50,
                  step: 1,
                  style: { "width": "100px", "margin": "0 8px" }
                }, null, 8, ["value"])
              ]),
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
const TestView_vue_vue_type_style_index_0_scoped_b8202046_lang = "";
const TestView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-b8202046"]]);
export {
  TestView as default
};
