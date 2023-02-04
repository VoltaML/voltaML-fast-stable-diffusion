import { I as ImageUpload } from "./ImageUpload.js";
import { X as cB, a0 as cNotM, $ as cE, Y as cM, d as defineComponent, a1 as useConfig, a2 as useTheme, P as computed, ab as useThemeClass, G as h, R as Fragment, bl as dividerLight, c as createElementBlock, e as createVNode, f as unref, o as openBlock } from "./index.js";
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
