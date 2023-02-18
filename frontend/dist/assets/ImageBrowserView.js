import { Z as c, X as cB, a0 as cNotM, Y as cM, $ as cE, aD as insideModal, aE as insidePopover, d as defineComponent, a1 as useConfig, a2 as useTheme, Q as computed, ab as useThemeClass, a3 as useCompitable, a4 as flatten, G as h, a$ as getSlot, br as descriptionsLight, ah as createKey, r as ref, b2 as NScrollbar$1, g as useState, bo as reactive, y as serverUrl, c as createBlock, w as withCtx, u as unref, o as openBlock, b as createVNode, bq as NImage, bs as createCommentVNode, k as NGi, a as createBaseVNode, e as NCard, j as createElementBlock, bt as renderList, F as Fragment, m as createTextVNode, t as toDisplayString, x as NGrid, _ as _export_sfc } from "./index.js";
import { a as NTabs, N as NTabPane } from "./Tabs.js";
function getVNodeChildren(vNode, slotName = "default", fallback = []) {
  const { children } = vNode;
  if (children !== null && typeof children === "object" && !Array.isArray(children)) {
    const slot = children[slotName];
    if (typeof slot === "function") {
      return slot();
    }
  }
  return fallback;
}
const DESCRIPTION_ITEM_FLAG = "DESCRIPTION_ITEM_FLAG";
function isDescriptionsItem(vNode) {
  if (typeof vNode === "object" && vNode && !Array.isArray(vNode)) {
    return vNode.type && vNode.type[DESCRIPTION_ITEM_FLAG];
  }
  return false;
}
const style = c([cB("descriptions", {
  fontSize: "var(--n-font-size)"
}, [cB("descriptions-separator", `
 display: inline-block;
 margin: 0 8px 0 2px;
 `), cB("descriptions-table-wrapper", [cB("descriptions-table", [cB("descriptions-table-row", [cB("descriptions-table-header", {
  padding: "var(--n-th-padding)"
}), cB("descriptions-table-content", {
  padding: "var(--n-td-padding)"
})])])]), cNotM("bordered", [cB("descriptions-table-wrapper", [cB("descriptions-table", [cB("descriptions-table-row", [c("&:last-child", [cB("descriptions-table-content", {
  paddingBottom: 0
})])])])])]), cM("left-label-placement", [cB("descriptions-table-content", [c("> *", {
  verticalAlign: "top"
})])]), cM("left-label-align", [c("th", {
  textAlign: "left"
})]), cM("center-label-align", [c("th", {
  textAlign: "center"
})]), cM("right-label-align", [c("th", {
  textAlign: "right"
})]), cM("bordered", [cB("descriptions-table-wrapper", `
 border-radius: var(--n-border-radius);
 overflow: hidden;
 background: var(--n-merged-td-color);
 border: 1px solid var(--n-merged-border-color);
 `, [cB("descriptions-table", [cB("descriptions-table-row", [c("&:not(:last-child)", [cB("descriptions-table-content", {
  borderBottom: "1px solid var(--n-merged-border-color)"
}), cB("descriptions-table-header", {
  borderBottom: "1px solid var(--n-merged-border-color)"
})]), cB("descriptions-table-header", `
 font-weight: 400;
 background-clip: padding-box;
 background-color: var(--n-merged-th-color);
 `, [c("&:not(:last-child)", {
  borderRight: "1px solid var(--n-merged-border-color)"
})]), cB("descriptions-table-content", [c("&:not(:last-child)", {
  borderRight: "1px solid var(--n-merged-border-color)"
})])])])])]), cB("descriptions-header", `
 font-weight: var(--n-th-font-weight);
 font-size: 18px;
 transition: color .3s var(--n-bezier);
 line-height: var(--n-line-height);
 margin-bottom: 16px;
 color: var(--n-title-text-color);
 `), cB("descriptions-table-wrapper", `
 transition:
 background-color .3s var(--n-bezier),
 border-color .3s var(--n-bezier);
 `, [cB("descriptions-table", `
 width: 100%;
 border-collapse: separate;
 border-spacing: 0;
 box-sizing: border-box;
 `, [cB("descriptions-table-row", `
 box-sizing: border-box;
 transition: border-color .3s var(--n-bezier);
 `, [cB("descriptions-table-header", `
 font-weight: var(--n-th-font-weight);
 line-height: var(--n-line-height);
 display: table-cell;
 box-sizing: border-box;
 color: var(--n-th-text-color);
 transition:
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier),
 border-color .3s var(--n-bezier);
 `), cB("descriptions-table-content", `
 vertical-align: top;
 line-height: var(--n-line-height);
 display: table-cell;
 box-sizing: border-box;
 color: var(--n-td-text-color);
 transition:
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier),
 border-color .3s var(--n-bezier);
 `, [cE("content", `
 transition: color .3s var(--n-bezier);
 display: inline-block;
 color: var(--n-td-text-color);
 `)]), cE("label", `
 font-weight: var(--n-th-font-weight);
 transition: color .3s var(--n-bezier);
 display: inline-block;
 margin-right: 14px;
 color: var(--n-th-text-color);
 `)])])])]), cB("descriptions-table-wrapper", `
 --n-merged-th-color: var(--n-th-color);
 --n-merged-td-color: var(--n-td-color);
 --n-merged-border-color: var(--n-border-color);
 `), insideModal(cB("descriptions-table-wrapper", `
 --n-merged-th-color: var(--n-th-color-modal);
 --n-merged-td-color: var(--n-td-color-modal);
 --n-merged-border-color: var(--n-border-color-modal);
 `)), insidePopover(cB("descriptions-table-wrapper", `
 --n-merged-th-color: var(--n-th-color-popover);
 --n-merged-td-color: var(--n-td-color-popover);
 --n-merged-border-color: var(--n-border-color-popover);
 `))]);
const descriptionsProps = Object.assign(Object.assign({}, useTheme.props), { title: String, column: {
  type: Number,
  default: 3
}, columns: Number, labelPlacement: {
  type: String,
  default: "top"
}, labelAlign: {
  type: String,
  default: "left"
}, separator: {
  type: String,
  default: ":"
}, size: {
  type: String,
  default: "medium"
}, bordered: Boolean, labelStyle: [Object, String], contentStyle: [Object, String] });
const NDescriptions = defineComponent({
  name: "Descriptions",
  props: descriptionsProps,
  setup(props) {
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const themeRef = useTheme("Descriptions", "-descriptions", style, descriptionsLight, props, mergedClsPrefixRef);
    const cssVarsRef = computed(() => {
      const { size, bordered } = props;
      const { common: { cubicBezierEaseInOut }, self: { titleTextColor, thColor, thColorModal, thColorPopover, thTextColor, thFontWeight, tdTextColor, tdColor, tdColorModal, tdColorPopover, borderColor, borderColorModal, borderColorPopover, borderRadius, lineHeight, [createKey("fontSize", size)]: fontSize, [createKey(bordered ? "thPaddingBordered" : "thPadding", size)]: thPadding, [createKey(bordered ? "tdPaddingBordered" : "tdPadding", size)]: tdPadding } } = themeRef.value;
      return {
        "--n-title-text-color": titleTextColor,
        "--n-th-padding": thPadding,
        "--n-td-padding": tdPadding,
        "--n-font-size": fontSize,
        "--n-bezier": cubicBezierEaseInOut,
        "--n-th-font-weight": thFontWeight,
        "--n-line-height": lineHeight,
        "--n-th-text-color": thTextColor,
        "--n-td-text-color": tdTextColor,
        "--n-th-color": thColor,
        "--n-th-color-modal": thColorModal,
        "--n-th-color-popover": thColorPopover,
        "--n-td-color": tdColor,
        "--n-td-color-modal": tdColorModal,
        "--n-td-color-popover": tdColorPopover,
        "--n-border-radius": borderRadius,
        "--n-border-color": borderColor,
        "--n-border-color-modal": borderColorModal,
        "--n-border-color-popover": borderColorPopover
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("descriptions", computed(() => {
      let hash = "";
      const { size, bordered } = props;
      if (bordered)
        hash += "a";
      hash += size[0];
      return hash;
    }), cssVarsRef, props) : void 0;
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender,
      compitableColumn: useCompitable(props, ["columns", "column"]),
      inlineThemeDisabled
    };
  },
  render() {
    const defaultSlots = this.$slots.default;
    const children = defaultSlots ? flatten(defaultSlots()) : [];
    children.length;
    const { compitableColumn, labelPlacement, labelAlign, size, bordered, title, cssVars, mergedClsPrefix, separator, onRender } = this;
    onRender === null || onRender === void 0 ? void 0 : onRender();
    const filteredChildren = children.filter((child) => isDescriptionsItem(child));
    const defaultState = {
      span: 0,
      row: [],
      secondRow: [],
      rows: []
    };
    const itemState = filteredChildren.reduce((state, vNode, index) => {
      const props = vNode.props || {};
      const isLastIteration = filteredChildren.length - 1 === index;
      const itemLabel = [
        "label" in props ? props.label : getVNodeChildren(vNode, "label")
      ];
      const itemChildren = [getVNodeChildren(vNode)];
      const itemSpan = props.span || 1;
      const memorizedSpan = state.span;
      state.span += itemSpan;
      const labelStyle = props.labelStyle || props["label-style"] || this.labelStyle;
      const contentStyle = props.contentStyle || props["content-style"] || this.contentStyle;
      if (labelPlacement === "left") {
        if (bordered) {
          state.row.push(h("th", { class: `${mergedClsPrefix}-descriptions-table-header`, colspan: 1, style: labelStyle }, itemLabel), h("td", { class: `${mergedClsPrefix}-descriptions-table-content`, colspan: isLastIteration ? (compitableColumn - memorizedSpan) * 2 + 1 : itemSpan * 2 - 1, style: contentStyle }, itemChildren));
        } else {
          state.row.push(h(
            "td",
            { class: `${mergedClsPrefix}-descriptions-table-content`, colspan: isLastIteration ? (compitableColumn - memorizedSpan) * 2 : itemSpan * 2 },
            h("span", { class: `${mergedClsPrefix}-descriptions-table-content__label`, style: labelStyle }, [
              ...itemLabel,
              separator && h("span", { class: `${mergedClsPrefix}-descriptions-separator` }, separator)
            ]),
            h("span", { class: `${mergedClsPrefix}-descriptions-table-content__content`, style: contentStyle }, itemChildren)
          ));
        }
      } else {
        const colspan = isLastIteration ? (compitableColumn - memorizedSpan) * 2 : itemSpan * 2;
        state.row.push(h("th", { class: `${mergedClsPrefix}-descriptions-table-header`, colspan, style: labelStyle }, itemLabel));
        state.secondRow.push(h("td", { class: `${mergedClsPrefix}-descriptions-table-content`, colspan, style: contentStyle }, itemChildren));
      }
      if (state.span >= compitableColumn || isLastIteration) {
        state.span = 0;
        if (state.row.length) {
          state.rows.push(state.row);
          state.row = [];
        }
        if (labelPlacement !== "left") {
          if (state.secondRow.length) {
            state.rows.push(state.secondRow);
            state.secondRow = [];
          }
        }
      }
      return state;
    }, defaultState);
    const rows = itemState.rows.map((row) => h("tr", { class: `${mergedClsPrefix}-descriptions-table-row` }, row));
    return h(
      "div",
      { style: cssVars, class: [
        `${mergedClsPrefix}-descriptions`,
        this.themeClass,
        `${mergedClsPrefix}-descriptions--${labelPlacement}-label-placement`,
        `${mergedClsPrefix}-descriptions--${labelAlign}-label-align`,
        `${mergedClsPrefix}-descriptions--${size}-size`,
        bordered && `${mergedClsPrefix}-descriptions--bordered`
      ] },
      title || this.$slots.header ? h("div", { class: `${mergedClsPrefix}-descriptions-header` }, title || getSlot(this, "header")) : null,
      h(
        "div",
        { class: `${mergedClsPrefix}-descriptions-table-wrapper` },
        h(
          "table",
          { class: `${mergedClsPrefix}-descriptions-table` },
          h("tbody", null, rows)
        )
      )
    );
  }
});
const descriptionsItemProps = {
  label: String,
  span: {
    type: Number,
    default: 1
  },
  labelStyle: [Object, String],
  contentStyle: [Object, String]
};
const NDescriptionsItem = defineComponent({
  name: "DescriptionsItem",
  [DESCRIPTION_ITEM_FLAG]: true,
  props: descriptionsItemProps,
  render() {
    return null;
  }
});
const scrollbarProps = Object.assign(Object.assign({}, useTheme.props), { trigger: String, xScrollable: Boolean, onScroll: Function });
const Scrollbar = defineComponent({
  name: "Scrollbar",
  props: scrollbarProps,
  setup() {
    const scrollbarInstRef = ref(null);
    const exposedMethods = {
      scrollTo: (...args) => {
        var _a;
        (_a = scrollbarInstRef.value) === null || _a === void 0 ? void 0 : _a.scrollTo(args[0], args[1]);
      },
      scrollBy: (...args) => {
        var _a;
        (_a = scrollbarInstRef.value) === null || _a === void 0 ? void 0 : _a.scrollBy(args[0], args[1]);
      }
    };
    return Object.assign(Object.assign({}, exposedMethods), { scrollbarInstRef });
  },
  render() {
    return h(NScrollbar$1, Object.assign({ ref: "scrollbarInstRef" }, this.$props), this.$slots);
  }
});
const NScrollbar = Scrollbar;
const _hoisted_1 = { style: { "height": "100%", "width": "100%" } };
const _hoisted_2 = ["onClick"];
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ImageBrowserView",
  setup(__props) {
    const global = useState();
    function urlFromPath(path) {
      const url = new URL(path, serverUrl);
      return url.href;
    }
    const imageSrc = computed(() => {
      const url = urlFromPath(global.state.imageBrowser.currentImage.path);
      return url;
    });
    function txt2imgClick(i) {
      global.state.imageBrowser.currentImage = txt2imgData[i];
      console.log(txt2imgData[i].path);
      const url = new URL(`${serverUrl}/api/output/data/`);
      url.searchParams.append("filename", txt2imgData[i].path);
      console.log(url);
      fetch(url).then((res) => res.json()).then((data) => {
        global.state.imageBrowser.currentImageMetadata = data;
      });
    }
    const txt2imgData = reactive([]);
    fetch(`${serverUrl}/api/output/txt2img`).then((res) => res.json()).then((data) => {
      data.forEach((item) => {
        txt2imgData.push(item);
      });
      txt2imgData.sort((a, b) => {
        return b.time - a.time;
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NGrid), { cols: "1 850:3" }, {
        default: withCtx(() => [
          createVNode(unref(NGi), {
            span: "2",
            style: { "height": "calc((100vh - 170px) - 24px)", "display": "inline-flex", "justify-content": "center" }
          }, {
            default: withCtx(() => [
              unref(global).state.imageBrowser.currentImage.path !== "" ? (openBlock(), createBlock(unref(NImage), {
                key: 0,
                src: unref(imageSrc),
                "object-fit": "contain",
                style: { "width": "100%", "height": "100%", "justify-content": "center", "margin": "8px" },
                "img-props": { style: { maxWidth: "95%", maxHeight: "95%" } }
              }, null, 8, ["src"])) : createCommentVNode("", true)
            ]),
            _: 1
          }),
          createVNode(unref(NGi), null, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_1, [
                createVNode(unref(NCard), null, {
                  default: withCtx(() => [
                    createVNode(unref(NTabs), {
                      type: "segment",
                      style: { "height": "100%" }
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NTabPane), {
                          name: "Txt2Img",
                          style: { "height": "calc(((100vh - 200px) - 53px) - 24px)" }
                        }, {
                          default: withCtx(() => [
                            createVNode(unref(NScrollbar), {
                              trigger: "hover",
                              style: { "height": "100%" }
                            }, {
                              default: withCtx(() => [
                                (openBlock(true), createElementBlock(Fragment, null, renderList(txt2imgData, (i, index) => {
                                  return openBlock(), createElementBlock("span", {
                                    onClick: ($event) => txt2imgClick(index),
                                    key: index,
                                    class: "img-container"
                                  }, [
                                    createVNode(unref(NImage), {
                                      class: "img-slider",
                                      src: urlFromPath(i.path),
                                      lazy: "",
                                      "preview-disabled": "",
                                      style: { "justify-content": "center" }
                                    }, null, 8, ["src"])
                                  ], 8, _hoisted_2);
                                }), 128))
                              ]),
                              _: 1
                            })
                          ]),
                          _: 1
                        }),
                        createVNode(unref(NTabPane), { name: "Img2Img" })
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                })
              ])
            ]),
            _: 1
          }),
          createVNode(unref(NGi), { span: "3" }, {
            default: withCtx(() => [
              createVNode(unref(NDescriptions), { bordered: "" }, {
                default: withCtx(() => [
                  createVNode(unref(NDescriptionsItem), {
                    label: "File",
                    "content-style": "max-width: 100px"
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(unref(global).state.imageBrowser.currentImage.path.split("/").pop()), 1)
                    ]),
                    _: 1
                  }),
                  createVNode(unref(NDescriptionsItem), {
                    label: "Model",
                    "content-style": "max-width: 100px"
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(unref(global).state.imageBrowser.currentImageMetadata.model), 1)
                    ]),
                    _: 1
                  }),
                  createVNode(unref(NDescriptionsItem), {
                    label: "Seed",
                    "content-style": "max-width: 100px"
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(unref(global).state.imageBrowser.currentImageMetadata.seed), 1)
                    ]),
                    _: 1
                  }),
                  createVNode(unref(NDescriptionsItem), {
                    label: "Prompt",
                    "content-style": "max-width: 100px"
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(unref(global).state.imageBrowser.currentImageMetadata.prompt), 1)
                    ]),
                    _: 1
                  }),
                  createVNode(unref(NDescriptionsItem), {
                    label: "Negative Prompt",
                    "content-style": "max-width: 100px"
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(unref(global).state.imageBrowser.currentImageMetadata.negative_prompt), 1)
                    ]),
                    _: 1
                  }),
                  createVNode(unref(NDescriptionsItem), {
                    label: "Steps",
                    "content-style": "max-width: 100px"
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(unref(global).state.imageBrowser.currentImageMetadata.steps), 1)
                    ]),
                    _: 1
                  }),
                  createVNode(unref(NDescriptionsItem), {
                    label: "Width",
                    "content-style": "max-width: 100px"
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(unref(global).state.imageBrowser.currentImageMetadata.width), 1)
                    ]),
                    _: 1
                  }),
                  createVNode(unref(NDescriptionsItem), {
                    label: "Height",
                    "content-style": "max-width: 100px"
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(unref(global).state.imageBrowser.currentImageMetadata.height), 1)
                    ]),
                    _: 1
                  }),
                  createVNode(unref(NDescriptionsItem), {
                    label: "Guidance Scale",
                    "content-style": "max-width: 100px"
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(unref(global).state.imageBrowser.currentImageMetadata.guidance_scale), 1)
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
const ImageBrowserView_vue_vue_type_style_index_0_scoped_6bd478f3_lang = "";
const ImageBrowserView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-6bd478f3"]]);
export {
  ImageBrowserView as default
};
