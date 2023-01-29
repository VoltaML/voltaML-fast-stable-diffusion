import { C as cB, ap as cNotM, D as cE, a0 as cM, d as defineComponent, E as useConfig, F as useTheme, G as computed, H as useThemeClass, B as h, P as Fragment, b3 as dividerLight, O as ref, o as openBlock, c as createElementBlock, i as createBaseVNode, e as unref, k as createTextVNode, b as createVNode, w as withCtx, Q as NButton, A as _export_sfc } from "./index.js";
const style = cB("divider", `
 position: relative;
 display: flex;
 width: 100%;
 box-sizing: border-box;
 font-size: 16px;
 color: var(--n-text-color);
 transition:
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
`, [cNotM("vertical", `
 margin-top: 24px;
 margin-bottom: 24px;
 `, [cNotM("no-title", `
 display: flex;
 align-items: center;
 `)]), cE("title", `
 display: flex;
 align-items: center;
 margin-left: 12px;
 margin-right: 12px;
 white-space: nowrap;
 font-weight: var(--n-font-weight);
 `), cM("title-position-left", [cE("line", [cM("left", {
  width: "28px"
})])]), cM("title-position-right", [cE("line", [cM("right", {
  width: "28px"
})])]), cM("dashed", [cE("line", `
 background-color: #0000;
 height: 0px;
 width: 100%;
 border-style: dashed;
 border-width: 1px 0 0;
 `)]), cM("vertical", `
 display: inline-block;
 height: 1em;
 margin: 0 8px;
 vertical-align: middle;
 width: 1px;
 `), cE("line", `
 border: none;
 transition: background-color .3s var(--n-bezier), border-color .3s var(--n-bezier);
 height: 1px;
 width: 100%;
 margin: 0;
 `), cNotM("dashed", [cE("line", {
  backgroundColor: "var(--n-color)"
})]), cM("dashed", [cE("line", {
  borderColor: "var(--n-color)"
})]), cM("vertical", {
  backgroundColor: "var(--n-color)"
})]);
const dividerProps = Object.assign(Object.assign({}, useTheme.props), { titlePlacement: {
  type: String,
  default: "center"
}, dashed: Boolean, vertical: Boolean });
const NDivider = defineComponent({
  name: "Divider",
  props: dividerProps,
  setup(props) {
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const themeRef = useTheme("Divider", "-divider", style, dividerLight, props, mergedClsPrefixRef);
    const cssVarsRef = computed(() => {
      const { common: { cubicBezierEaseInOut }, self: { color, textColor, fontWeight } } = themeRef.value;
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-color": color,
        "--n-text-color": textColor,
        "--n-font-weight": fontWeight
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("divider", void 0, cssVarsRef, props) : void 0;
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    var _a;
    const { $slots, titlePlacement, vertical, dashed, cssVars, mergedClsPrefix } = this;
    (_a = this.onRender) === null || _a === void 0 ? void 0 : _a.call(this);
    return h(
      "div",
      { role: "separator", class: [
        `${mergedClsPrefix}-divider`,
        this.themeClass,
        {
          [`${mergedClsPrefix}-divider--vertical`]: vertical,
          [`${mergedClsPrefix}-divider--no-title`]: !$slots.default,
          [`${mergedClsPrefix}-divider--dashed`]: dashed,
          [`${mergedClsPrefix}-divider--title-position-${titlePlacement}`]: $slots.default && titlePlacement
        }
      ], style: cssVars },
      !vertical ? h("div", { class: `${mergedClsPrefix}-divider__line ${mergedClsPrefix}-divider__line--left` }) : null,
      !vertical && $slots.default ? h(
        Fragment,
        null,
        h("div", { class: `${mergedClsPrefix}-divider__title` }, this.$slots),
        h("div", { class: `${mergedClsPrefix}-divider__line ${mergedClsPrefix}-divider__line--right` })
      ) : null
    );
  }
});
const _hoisted_1 = ["src"];
const _hoisted_2 = { for: "file-upload" };
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "ImageUpload",
  setup(__props) {
    let preview = ref("");
    function previewImage(event) {
      const input = event.target;
      if (input.files) {
        const reader = new FileReader();
        reader.onload = (e) => {
          var _a;
          const i = (_a = e.target) == null ? void 0 : _a.result;
          if (i) {
            preview.value = i.toString();
          }
        };
        reader.readAsDataURL(input.files[0]);
      }
    }
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createBaseVNode("img", {
          src: unref(preview),
          style: { "width": "400px", "height": "auto" }
        }, null, 8, _hoisted_1),
        createBaseVNode("label", _hoisted_2, [
          createTextVNode("Hello world"),
          createVNode(unref(NButton), {
            type: "warning",
            ghost: "",
            bordered: ""
          }, {
            default: withCtx(() => [
              createTextVNode("Upload an Image")
            ]),
            _: 1
          })
        ]),
        createBaseVNode("input", {
          type: "file",
          accept: "image/*",
          onChange: previewImage,
          id: "file-upload",
          class: "hidden-input"
        }, null, 32)
      ], 64);
    };
  }
});
const ImageUpload_vue_vue_type_style_index_0_scoped_900adde3_lang = "";
const ImageUpload = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["__scopeId", "data-v-900adde3"]]);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TestView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(ImageUpload),
        createVNode(unref(NDivider))
      ], 64);
    };
  }
});
export {
  _sfc_main as default
};
