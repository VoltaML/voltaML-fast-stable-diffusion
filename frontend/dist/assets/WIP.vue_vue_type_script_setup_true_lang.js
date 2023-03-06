import { o as h, e as cB, f as cE, j as defineComponent, u as useConfig, k as useTheme, m as computed, n as useThemeClass, x as NBaseIcon, I as InfoIcon, S as SuccessIcon, W as WarningIcon, E as ErrorIcon, a6 as resultLight, w as createKey, y as openBlock, z as createElementBlock, A as createBaseVNode, B as createBlock, C as withCtx, D as createVNode, G as unref, a5 as NIcon } from "./index.js";
const image404 = h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 36 36" },
  h("circle", { fill: "#FFCB4C", cx: "18", cy: "17.018", r: "17" }),
  h("path", { fill: "#65471B", d: "M14.524 21.036c-.145-.116-.258-.274-.312-.464-.134-.46.13-.918.59-1.021 4.528-1.021 7.577 1.363 7.706 1.465.384.306.459.845.173 1.205-.286.358-.828.401-1.211.097-.11-.084-2.523-1.923-6.182-1.098-.274.061-.554-.016-.764-.184z" }),
  h("ellipse", { fill: "#65471B", cx: "13.119", cy: "11.174", rx: "2.125", ry: "2.656" }),
  h("ellipse", { fill: "#65471B", cx: "24.375", cy: "12.236", rx: "2.125", ry: "2.656" }),
  h("path", { fill: "#F19020", d: "M17.276 35.149s1.265-.411 1.429-1.352c.173-.972-.624-1.167-.624-1.167s1.041-.208 1.172-1.376c.123-1.101-.861-1.363-.861-1.363s.97-.4 1.016-1.539c.038-.959-.995-1.428-.995-1.428s5.038-1.221 5.556-1.341c.516-.12 1.32-.615 1.069-1.694-.249-1.08-1.204-1.118-1.697-1.003-.494.115-6.744 1.566-8.9 2.068l-1.439.334c-.54.127-.785-.11-.404-.512.508-.536.833-1.129.946-2.113.119-1.035-.232-2.313-.433-2.809-.374-.921-1.005-1.649-1.734-1.899-1.137-.39-1.945.321-1.542 1.561.604 1.854.208 3.375-.833 4.293-2.449 2.157-3.588 3.695-2.83 6.973.828 3.575 4.377 5.876 7.952 5.048l3.152-.681z" }),
  h("path", { fill: "#65471B", d: "M9.296 6.351c-.164-.088-.303-.224-.391-.399-.216-.428-.04-.927.393-1.112 4.266-1.831 7.699-.043 7.843.034.433.231.608.747.391 1.154-.216.405-.74.546-1.173.318-.123-.063-2.832-1.432-6.278.047-.257.109-.547.085-.785-.042zm12.135 3.75c-.156-.098-.286-.243-.362-.424-.187-.442.023-.927.468-1.084 4.381-1.536 7.685.48 7.823.567.415.26.555.787.312 1.178-.242.39-.776.495-1.191.238-.12-.072-2.727-1.621-6.267-.379-.266.091-.553.046-.783-.096z" })
);
const image500 = h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 36 36" },
  h("path", { fill: "#FFCC4D", d: "M36 18c0 9.941-8.059 18-18 18-9.94 0-18-8.059-18-18C0 8.06 8.06 0 18 0c9.941 0 18 8.06 18 18" }),
  h("ellipse", { fill: "#664500", cx: "18", cy: "27", rx: "5", ry: "6" }),
  h("path", { fill: "#664500", d: "M5.999 11c-.208 0-.419-.065-.599-.2-.442-.331-.531-.958-.2-1.4C8.462 5.05 12.816 5 13 5c.552 0 1 .448 1 1 0 .551-.445.998-.996 1-.155.002-3.568.086-6.204 3.6-.196.262-.497.4-.801.4zm24.002 0c-.305 0-.604-.138-.801-.4-2.64-3.521-6.061-3.598-6.206-3.6-.55-.006-.994-.456-.991-1.005C22.006 5.444 22.45 5 23 5c.184 0 4.537.05 7.8 4.4.332.442.242 1.069-.2 1.4-.18.135-.39.2-.599.2zm-16.087 4.5l1.793-1.793c.391-.391.391-1.023 0-1.414s-1.023-.391-1.414 0L12.5 14.086l-1.793-1.793c-.391-.391-1.023-.391-1.414 0s-.391 1.023 0 1.414l1.793 1.793-1.793 1.793c-.391.391-.391 1.023 0 1.414.195.195.451.293.707.293s.512-.098.707-.293l1.793-1.793 1.793 1.793c.195.195.451.293.707.293s.512-.098.707-.293c.391-.391.391-1.023 0-1.414L13.914 15.5zm11 0l1.793-1.793c.391-.391.391-1.023 0-1.414s-1.023-.391-1.414 0L23.5 14.086l-1.793-1.793c-.391-.391-1.023-.391-1.414 0s-.391 1.023 0 1.414l1.793 1.793-1.793 1.793c-.391.391-.391 1.023 0 1.414.195.195.451.293.707.293s.512-.098.707-.293l1.793-1.793 1.793 1.793c.195.195.451.293.707.293s.512-.098.707-.293c.391-.391.391-1.023 0-1.414L24.914 15.5z" })
);
const image418 = h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 36 36" },
  h("ellipse", { fill: "#292F33", cx: "18", cy: "26", rx: "18", ry: "10" }),
  h("ellipse", { fill: "#66757F", cx: "18", cy: "24", rx: "18", ry: "10" }),
  h("path", { fill: "#E1E8ED", d: "M18 31C3.042 31 1 16 1 12h34c0 2-1.958 19-17 19z" }),
  h("path", { fill: "#77B255", d: "M35 12.056c0 5.216-7.611 9.444-17 9.444S1 17.271 1 12.056C1 6.84 8.611 3.611 18 3.611s17 3.229 17 8.445z" }),
  h("ellipse", { fill: "#A6D388", cx: "18", cy: "13", rx: "15", ry: "7" }),
  h("path", { d: "M21 17c-.256 0-.512-.098-.707-.293-2.337-2.337-2.376-4.885-.125-8.262.739-1.109.9-2.246.478-3.377-.461-1.236-1.438-1.996-1.731-2.077-.553 0-.958-.443-.958-.996 0-.552.491-.995 1.043-.995.997 0 2.395 1.153 3.183 2.625 1.034 1.933.91 4.039-.351 5.929-1.961 2.942-1.531 4.332-.125 5.738.391.391.391 1.023 0 1.414-.195.196-.451.294-.707.294zm-6-2c-.256 0-.512-.098-.707-.293-2.337-2.337-2.376-4.885-.125-8.262.727-1.091.893-2.083.494-2.947-.444-.961-1.431-1.469-1.684-1.499-.552 0-.989-.447-.989-1 0-.552.458-1 1.011-1 .997 0 2.585.974 3.36 2.423.481.899 1.052 2.761-.528 5.131-1.961 2.942-1.531 4.332-.125 5.738.391.391.391 1.023 0 1.414-.195.197-.451.295-.707.295z", fill: "#5C913B" })
);
const image403 = h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 36 36" },
  h("path", { fill: "#EF9645", d: "M15.5 2.965c1.381 0 2.5 1.119 2.5 2.5v.005L20.5.465c1.381 0 2.5 1.119 2.5 2.5V4.25l2.5-1.535c1.381 0 2.5 1.119 2.5 2.5V8.75L29 18H15.458L15.5 2.965z" }),
  h("path", { fill: "#FFDC5D", d: "M4.625 16.219c1.381-.611 3.354.208 4.75 2.188.917 1.3 1.187 3.151 2.391 3.344.46.073 1.234-.313 1.234-1.397V4.5s0-2 2-2 2 2 2 2v11.633c0-.029 1-.064 1-.082V2s0-2 2-2 2 2 2 2v14.053c0 .017 1 .041 1 .069V4.25s0-2 2-2 2 2 2 2v12.638c0 .118 1 .251 1 .398V8.75s0-2 2-2 2 2 2 2V24c0 6.627-5.373 12-12 12-4.775 0-8.06-2.598-9.896-5.292C8.547 28.423 8.096 26.051 8 25.334c0 0-.123-1.479-1.156-2.865-1.469-1.969-2.5-3.156-3.125-3.866-.317-.359-.625-1.707.906-2.384z" })
);
const style = cB("result", `
 color: var(--n-text-color);
 line-height: var(--n-line-height);
 font-size: var(--n-font-size);
 transition:
 color .3s var(--n-bezier);
`, [cB("result-icon", `
 display: flex;
 justify-content: center;
 transition: color .3s var(--n-bezier);
 `, [cE("status-image", `
 font-size: var(--n-icon-size);
 width: 1em;
 height: 1em;
 `), cB("base-icon", `
 color: var(--n-icon-color);
 font-size: var(--n-icon-size);
 `)]), cB("result-content", {
  marginTop: "24px"
}), cB("result-footer", `
 margin-top: 24px;
 text-align: center;
 `), cB("result-header", [cE("title", `
 margin-top: 16px;
 font-weight: var(--n-title-font-weight);
 transition: color .3s var(--n-bezier);
 text-align: center;
 color: var(--n-title-text-color);
 font-size: var(--n-title-font-size);
 `), cE("description", `
 margin-top: 4px;
 text-align: center;
 font-size: var(--n-font-size);
 `)])]);
const iconMap = {
  403: image403,
  404: image404,
  418: image418,
  500: image500,
  info: h(InfoIcon, null),
  success: h(SuccessIcon, null),
  warning: h(WarningIcon, null),
  error: h(ErrorIcon, null)
};
const resultProps = Object.assign(Object.assign({}, useTheme.props), { size: {
  type: String,
  default: "medium"
}, status: {
  type: String,
  default: "info"
}, title: String, description: String });
const NResult = defineComponent({
  name: "Result",
  props: resultProps,
  setup(props) {
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const themeRef = useTheme("Result", "-result", style, resultLight, props, mergedClsPrefixRef);
    const cssVarsRef = computed(() => {
      const { size, status } = props;
      const { common: { cubicBezierEaseInOut }, self: { textColor, lineHeight, titleTextColor, titleFontWeight, [createKey("iconColor", status)]: iconColor, [createKey("fontSize", size)]: fontSize, [createKey("titleFontSize", size)]: titleFontSize, [createKey("iconSize", size)]: iconSize } } = themeRef.value;
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-font-size": fontSize,
        "--n-icon-size": iconSize,
        "--n-line-height": lineHeight,
        "--n-text-color": textColor,
        "--n-title-font-size": titleFontSize,
        "--n-title-font-weight": titleFontWeight,
        "--n-title-text-color": titleTextColor,
        "--n-icon-color": iconColor || ""
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("result", computed(() => {
      const { size, status } = props;
      let hash = "";
      if (size) {
        hash += size[0];
      }
      if (status) {
        hash += status[0];
      }
      return hash;
    }), cssVarsRef, props) : void 0;
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    var _a;
    const { status, $slots, mergedClsPrefix, onRender } = this;
    onRender === null || onRender === void 0 ? void 0 : onRender();
    return h(
      "div",
      { class: [`${mergedClsPrefix}-result`, this.themeClass], style: this.cssVars },
      h("div", { class: `${mergedClsPrefix}-result-icon` }, ((_a = $slots.icon) === null || _a === void 0 ? void 0 : _a.call($slots)) || h(NBaseIcon, { clsPrefix: mergedClsPrefix }, { default: () => iconMap[status] })),
      h(
        "div",
        { class: `${mergedClsPrefix}-result-header` },
        this.title ? h("div", { class: `${mergedClsPrefix}-result-header__title` }, this.title) : null,
        this.description ? h("div", { class: `${mergedClsPrefix}-result-header__description` }, this.description) : null
      ),
      $slots.default && h("div", { class: `${mergedClsPrefix}-result-content` }, $slots),
      $slots.footer && h("div", { class: `${mergedClsPrefix}-result-footer` }, $slots.footer())
    );
  }
});
const _hoisted_1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M393.87 190a32.1 32.1 0 0 1-45.25 0l-26.57-26.57a32.09 32.09 0 0 1 0-45.26L382.19 58a1 1 0 0 0-.3-1.64c-38.82-16.64-89.15-8.16-121.11 23.57c-30.58 30.35-32.32 76-21.12 115.84a31.93 31.93 0 0 1-9.06 32.08L64 380a48.17 48.17 0 1 0 68 68l153.86-167a31.93 31.93 0 0 1 31.6-9.13c39.54 10.59 84.54 8.6 114.72-21.19c32.49-32 39.5-88.56 23.75-120.93a1 1 0 0 0-1.6-.26z",
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-miterlimit": "10",
    "stroke-width": "32"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3 = /* @__PURE__ */ createBaseVNode(
  "circle",
  {
    cx: "96",
    cy: "416",
    r: "16",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4 = [_hoisted_2, _hoisted_3];
const BuildOutline = defineComponent({
  name: "BuildOutline",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1, _hoisted_4);
  }
});
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "WIP",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NResult), {
        title: "Work in progress",
        description: "This page is still under development.",
        style: { "height": "70vh", "display": "flex", "align-items": "center", "justify-content": "center" }
      }, {
        icon: withCtx(() => [
          createVNode(unref(NIcon), { size: "250" }, {
            default: withCtx(() => [
              createVNode(unref(BuildOutline))
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
export {
  _sfc_main as _
};
