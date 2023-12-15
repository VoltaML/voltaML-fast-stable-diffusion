import { aa as c, Q as cB, ac as cNotM, ab as cM, at as cE, aU as insideModal, aV as insidePopover, d as defineComponent, S as useConfig, T as useTheme, i as computed, ah as createKey, Y as useThemeClass, bW as useCompitable, aw as flatten, x as h, aQ as repeat, ax as getSlot, bX as descriptionsLight } from "./index.js";
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
          h(
            "tbody",
            null,
            labelPlacement === "top" && h("tr", { class: `${mergedClsPrefix}-descriptions-table-row`, style: {
              visibility: "collapse"
            } }, repeat(compitableColumn * 2, h("td", null))),
            rows
          )
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
export {
  NDescriptionsItem as N,
  NDescriptions as a
};
