import { d as defineComponent, O as ref, b4 as useSsrAdapter, b5 as cssrAnchorMetaName, B as h, b6 as c, b7 as isSymbol, b8 as isObject, b9 as root, $ as c$1, C as cB, ap as cNotM, a0 as cM, D as cE, a2 as insideModal, a3 as insidePopover, E as useConfig, F as useTheme, G as computed, H as useThemeClass, ba as useCompitable, aA as flatten, aB as getSlot, bb as descriptionsLight, M as createKey, aE as NScrollbar$1, Y as createInjectionKey, a4 as inject, aZ as throwError, ax as mergeProps, P as Fragment, I as NBaseIcon, bc as AddIcon, bd as render, be as NBaseClose, am as omit, U as useMergedState, ac as watch, V as provide, X as toRef, bf as onFontsReady, ar as watchEffect, az as resolveWrappedSlot, aP as VResizeObserver, bg as tabsLight, Z as call, ah as nextTick, bh as withDirectives, bi as vShow, bj as TransitionGroup, bk as cloneVNode, u as useState, b0 as reactive, v as serverUrl, q as createBlock, w as withCtx, e as unref, o as openBlock, b as createVNode, r as NImage, s as createCommentVNode, N as NGi, i as createBaseVNode, f as NCard, c as createElementBlock, bl as renderList, k as createTextVNode, bm as toDisplayString, t as NGrid, A as _export_sfc } from "./index.js";
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
const styles = c(".v-x-scroll", {
  overflow: "auto",
  scrollbarWidth: "none"
}, [
  c("&::-webkit-scrollbar", {
    width: 0,
    height: 0
  })
]);
const VXScroll = defineComponent({
  name: "XScroll",
  props: {
    disabled: Boolean,
    onScroll: Function
  },
  setup() {
    const selfRef = ref(null);
    function handleWheel(e) {
      const preventYWheel = e.currentTarget.offsetWidth < e.currentTarget.scrollWidth;
      if (!preventYWheel || e.deltaY === 0)
        return;
      e.currentTarget.scrollLeft += e.deltaY + e.deltaX;
      e.preventDefault();
    }
    const ssrAdapter = useSsrAdapter();
    styles.mount({
      id: "vueuc/x-scroll",
      head: true,
      anchorMetaName: cssrAnchorMetaName,
      ssr: ssrAdapter
    });
    const exposedMethods = {
      scrollTo(...args) {
        var _a;
        (_a = selfRef.value) === null || _a === void 0 ? void 0 : _a.scrollTo(...args);
      }
    };
    return Object.assign({
      selfRef,
      handleWheel
    }, exposedMethods);
  },
  render() {
    return h("div", {
      ref: "selfRef",
      onScroll: this.onScroll,
      onWheel: this.disabled ? void 0 : this.handleWheel,
      class: "v-x-scroll"
    }, this.$slots);
  }
});
var reWhitespace = /\s/;
function trimmedEndIndex(string) {
  var index = string.length;
  while (index-- && reWhitespace.test(string.charAt(index))) {
  }
  return index;
}
var reTrimStart = /^\s+/;
function baseTrim(string) {
  return string ? string.slice(0, trimmedEndIndex(string) + 1).replace(reTrimStart, "") : string;
}
var NAN = 0 / 0;
var reIsBadHex = /^[-+]0x[0-9a-f]+$/i;
var reIsBinary = /^0b[01]+$/i;
var reIsOctal = /^0o[0-7]+$/i;
var freeParseInt = parseInt;
function toNumber(value) {
  if (typeof value == "number") {
    return value;
  }
  if (isSymbol(value)) {
    return NAN;
  }
  if (isObject(value)) {
    var other = typeof value.valueOf == "function" ? value.valueOf() : value;
    value = isObject(other) ? other + "" : other;
  }
  if (typeof value != "string") {
    return value === 0 ? value : +value;
  }
  value = baseTrim(value);
  var isBinary = reIsBinary.test(value);
  return isBinary || reIsOctal.test(value) ? freeParseInt(value.slice(2), isBinary ? 2 : 8) : reIsBadHex.test(value) ? NAN : +value;
}
var now = function() {
  return root.Date.now();
};
const now$1 = now;
var FUNC_ERROR_TEXT$1 = "Expected a function";
var nativeMax = Math.max, nativeMin = Math.min;
function debounce(func, wait, options) {
  var lastArgs, lastThis, maxWait, result, timerId, lastCallTime, lastInvokeTime = 0, leading = false, maxing = false, trailing = true;
  if (typeof func != "function") {
    throw new TypeError(FUNC_ERROR_TEXT$1);
  }
  wait = toNumber(wait) || 0;
  if (isObject(options)) {
    leading = !!options.leading;
    maxing = "maxWait" in options;
    maxWait = maxing ? nativeMax(toNumber(options.maxWait) || 0, wait) : maxWait;
    trailing = "trailing" in options ? !!options.trailing : trailing;
  }
  function invokeFunc(time) {
    var args = lastArgs, thisArg = lastThis;
    lastArgs = lastThis = void 0;
    lastInvokeTime = time;
    result = func.apply(thisArg, args);
    return result;
  }
  function leadingEdge(time) {
    lastInvokeTime = time;
    timerId = setTimeout(timerExpired, wait);
    return leading ? invokeFunc(time) : result;
  }
  function remainingWait(time) {
    var timeSinceLastCall = time - lastCallTime, timeSinceLastInvoke = time - lastInvokeTime, timeWaiting = wait - timeSinceLastCall;
    return maxing ? nativeMin(timeWaiting, maxWait - timeSinceLastInvoke) : timeWaiting;
  }
  function shouldInvoke(time) {
    var timeSinceLastCall = time - lastCallTime, timeSinceLastInvoke = time - lastInvokeTime;
    return lastCallTime === void 0 || timeSinceLastCall >= wait || timeSinceLastCall < 0 || maxing && timeSinceLastInvoke >= maxWait;
  }
  function timerExpired() {
    var time = now$1();
    if (shouldInvoke(time)) {
      return trailingEdge(time);
    }
    timerId = setTimeout(timerExpired, remainingWait(time));
  }
  function trailingEdge(time) {
    timerId = void 0;
    if (trailing && lastArgs) {
      return invokeFunc(time);
    }
    lastArgs = lastThis = void 0;
    return result;
  }
  function cancel() {
    if (timerId !== void 0) {
      clearTimeout(timerId);
    }
    lastInvokeTime = 0;
    lastArgs = lastCallTime = lastThis = timerId = void 0;
  }
  function flush() {
    return timerId === void 0 ? result : trailingEdge(now$1());
  }
  function debounced() {
    var time = now$1(), isInvoking = shouldInvoke(time);
    lastArgs = arguments;
    lastThis = this;
    lastCallTime = time;
    if (isInvoking) {
      if (timerId === void 0) {
        return leadingEdge(lastCallTime);
      }
      if (maxing) {
        clearTimeout(timerId);
        timerId = setTimeout(timerExpired, wait);
        return invokeFunc(lastCallTime);
      }
    }
    if (timerId === void 0) {
      timerId = setTimeout(timerExpired, wait);
    }
    return result;
  }
  debounced.cancel = cancel;
  debounced.flush = flush;
  return debounced;
}
var FUNC_ERROR_TEXT = "Expected a function";
function throttle(func, wait, options) {
  var leading = true, trailing = true;
  if (typeof func != "function") {
    throw new TypeError(FUNC_ERROR_TEXT);
  }
  if (isObject(options)) {
    leading = "leading" in options ? !!options.leading : leading;
    trailing = "trailing" in options ? !!options.trailing : trailing;
  }
  return debounce(func, wait, {
    "leading": leading,
    "maxWait": wait,
    "trailing": trailing
  });
}
const DESCRIPTION_ITEM_FLAG = "DESCRIPTION_ITEM_FLAG";
function isDescriptionsItem(vNode) {
  if (typeof vNode === "object" && vNode && !Array.isArray(vNode)) {
    return vNode.type && vNode.type[DESCRIPTION_ITEM_FLAG];
  }
  return false;
}
const style$1 = c$1([cB("descriptions", {
  fontSize: "var(--n-font-size)"
}, [cB("descriptions-separator", `
 display: inline-block;
 margin: 0 8px 0 2px;
 `), cB("descriptions-table-wrapper", [cB("descriptions-table", [cB("descriptions-table-row", [cB("descriptions-table-header", {
  padding: "var(--n-th-padding)"
}), cB("descriptions-table-content", {
  padding: "var(--n-td-padding)"
})])])]), cNotM("bordered", [cB("descriptions-table-wrapper", [cB("descriptions-table", [cB("descriptions-table-row", [c$1("&:last-child", [cB("descriptions-table-content", {
  paddingBottom: 0
})])])])])]), cM("left-label-placement", [cB("descriptions-table-content", [c$1("> *", {
  verticalAlign: "top"
})])]), cM("left-label-align", [c$1("th", {
  textAlign: "left"
})]), cM("center-label-align", [c$1("th", {
  textAlign: "center"
})]), cM("right-label-align", [c$1("th", {
  textAlign: "right"
})]), cM("bordered", [cB("descriptions-table-wrapper", `
 border-radius: var(--n-border-radius);
 overflow: hidden;
 background: var(--n-merged-td-color);
 border: 1px solid var(--n-merged-border-color);
 `, [cB("descriptions-table", [cB("descriptions-table-row", [c$1("&:not(:last-child)", [cB("descriptions-table-content", {
  borderBottom: "1px solid var(--n-merged-border-color)"
}), cB("descriptions-table-header", {
  borderBottom: "1px solid var(--n-merged-border-color)"
})]), cB("descriptions-table-header", `
 font-weight: 400;
 background-clip: padding-box;
 background-color: var(--n-merged-th-color);
 `, [c$1("&:not(:last-child)", {
  borderRight: "1px solid var(--n-merged-border-color)"
})]), cB("descriptions-table-content", [c$1("&:not(:last-child)", {
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
    const themeRef = useTheme("Descriptions", "-descriptions", style$1, descriptionsLight, props, mergedClsPrefixRef);
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
const tabsInjectionKey = createInjectionKey("n-tabs");
const tabPaneProps = {
  tab: [String, Number, Object, Function],
  name: {
    type: [String, Number],
    required: true
  },
  disabled: Boolean,
  displayDirective: {
    type: String,
    default: "if"
  },
  closable: {
    type: Boolean,
    default: void 0
  },
  tabProps: Object,
  /** @deprecated */
  label: [String, Number, Object, Function]
};
const NTabPane = defineComponent({
  __TAB_PANE__: true,
  name: "TabPane",
  alias: ["TabPanel"],
  props: tabPaneProps,
  setup(props) {
    const NTab = inject(tabsInjectionKey, null);
    if (!NTab) {
      throwError("tab-pane", "`n-tab-pane` must be placed inside `n-tabs`.");
    }
    return {
      style: NTab.paneStyleRef,
      class: NTab.paneClassRef,
      mergedClsPrefix: NTab.mergedClsPrefixRef
    };
  },
  render() {
    return h("div", { class: [`${this.mergedClsPrefix}-tab-pane`, this.class], style: this.style }, this.$slots);
  }
});
const tabProps = Object.assign({ internalLeftPadded: Boolean, internalAddable: Boolean, internalCreatedByPane: Boolean }, omit(tabPaneProps, ["displayDirective"]));
const Tab = defineComponent({
  __TAB__: true,
  inheritAttrs: false,
  name: "Tab",
  props: tabProps,
  setup(props) {
    const {
      mergedClsPrefixRef,
      valueRef,
      typeRef,
      closableRef,
      tabStyleRef,
      tabChangeIdRef,
      onBeforeLeaveRef,
      triggerRef,
      handleAdd,
      activateTab,
      handleClose
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    } = inject(tabsInjectionKey);
    return {
      trigger: triggerRef,
      mergedClosable: computed(() => {
        if (props.internalAddable)
          return false;
        const { closable } = props;
        if (closable === void 0)
          return closableRef.value;
        return closable;
      }),
      style: tabStyleRef,
      clsPrefix: mergedClsPrefixRef,
      value: valueRef,
      type: typeRef,
      handleClose(e) {
        e.stopPropagation();
        if (props.disabled)
          return;
        handleClose(props.name);
      },
      activateTab() {
        if (props.disabled)
          return;
        if (props.internalAddable) {
          handleAdd();
          return;
        }
        const { name: nameProp } = props;
        const id = ++tabChangeIdRef.id;
        if (nameProp !== valueRef.value) {
          const { value: onBeforeLeave } = onBeforeLeaveRef;
          if (!onBeforeLeave) {
            activateTab(nameProp);
          } else {
            void Promise.resolve(onBeforeLeave(props.name, valueRef.value)).then((allowLeave) => {
              if (allowLeave && tabChangeIdRef.id === id) {
                activateTab(nameProp);
              }
            });
          }
        }
      }
    };
  },
  render() {
    const { internalAddable, clsPrefix, name, disabled, label, tab, value, mergedClosable, style: style2, trigger, $slots: { default: defaultSlot } } = this;
    const mergedTab = label !== null && label !== void 0 ? label : tab;
    return h(
      "div",
      { class: `${clsPrefix}-tabs-tab-wrapper` },
      this.internalLeftPadded ? h("div", { class: `${clsPrefix}-tabs-tab-pad` }) : null,
      h(
        "div",
        Object.assign({ key: name, "data-name": name, "data-disabled": disabled ? true : void 0 }, mergeProps({
          class: [
            `${clsPrefix}-tabs-tab`,
            value === name && `${clsPrefix}-tabs-tab--active`,
            disabled && `${clsPrefix}-tabs-tab--disabled`,
            mergedClosable && `${clsPrefix}-tabs-tab--closable`,
            internalAddable && `${clsPrefix}-tabs-tab--addable`
          ],
          onClick: trigger === "click" ? this.activateTab : void 0,
          onMouseenter: trigger === "hover" ? this.activateTab : void 0,
          style: internalAddable ? void 0 : style2
        }, this.internalCreatedByPane ? this.tabProps || {} : this.$attrs)),
        h("span", { class: `${clsPrefix}-tabs-tab__label` }, internalAddable ? h(
          Fragment,
          null,
          h("div", { class: `${clsPrefix}-tabs-tab__height-placeholder` }, " "),
          h(NBaseIcon, { clsPrefix }, {
            default: () => h(AddIcon, null)
          })
        ) : defaultSlot ? defaultSlot() : typeof mergedTab === "object" ? mergedTab : render(mergedTab !== null && mergedTab !== void 0 ? mergedTab : name)),
        mergedClosable && this.type === "card" ? h(NBaseClose, { clsPrefix, class: `${clsPrefix}-tabs-tab__close`, onClick: this.handleClose, disabled }) : null
      )
    );
  }
});
const style = cB("tabs", `
 box-sizing: border-box;
 width: 100%;
 display: flex;
 flex-direction: column;
 transition:
 background-color .3s var(--n-bezier),
 border-color .3s var(--n-bezier);
`, [cM("segment-type", [cB("tabs-rail", [c$1("&.transition-disabled", "color: red;", [cB("tabs-tab", `
 transition: none;
 `)])])]), cM("left, right", `
 flex-direction: row;
 `, [cB("tabs-bar", `
 width: 2px;
 right: 0;
 transition:
 top .2s var(--n-bezier),
 max-height .2s var(--n-bezier),
 background-color .3s var(--n-bezier);
 `), cB("tabs-tab", `
 padding: var(--n-tab-padding-vertical); 
 `)]), cM("right", `
 flex-direction: row-reverse;
 `, [cB("tabs-bar", `
 left: 0;
 `)]), cM("bottom", `
 flex-direction: column-reverse;
 justify-content: flex-end;
 `, [cB("tabs-bar", `
 top: 0;
 `)]), cB("tabs-rail", `
 padding: 3px;
 border-radius: var(--n-tab-border-radius);
 width: 100%;
 background-color: var(--n-color-segment);
 transition: background-color .3s var(--n-bezier);
 display: flex;
 align-items: center;
 `, [cB("tabs-tab-wrapper", `
 flex-basis: 0;
 flex-grow: 1;
 display: flex;
 align-items: center;
 justify-content: center;
 `, [cB("tabs-tab", `
 overflow: hidden;
 border-radius: var(--n-tab-border-radius);
 width: 100%;
 display: flex;
 align-items: center;
 justify-content: center;
 `, [cM("active", `
 font-weight: var(--n-font-weight-strong);
 color: var(--n-tab-text-color-active);
 background-color: var(--n-tab-color-segment);
 box-shadow: 0 1px 3px 0 rgba(0, 0, 0, .08);
 `), c$1("&:hover", `
 color: var(--n-tab-text-color-hover);
 `)])])]), cM("flex", [cB("tabs-nav", {
  width: "100%"
}, [cB("tabs-wrapper", {
  width: "100%"
}, [cB("tabs-tab", {
  marginRight: 0
})])])]), cB("tabs-nav", `
 box-sizing: border-box;
 line-height: 1.5;
 display: flex;
 transition: border-color .3s var(--n-bezier);
 `, [cE("prefix, suffix", `
 display: flex;
 align-items: center;
 `), cE("prefix", "padding-right: 16px;"), cE("suffix", "padding-left: 16px;")]), cB("tabs-nav-scroll-wrapper", `
 flex: 1;
 position: relative;
 overflow: hidden;
 `, [cM("shadow-before", [c$1("&::before", `
 box-shadow: inset 10px 0 8px -8px rgba(0, 0, 0, .12);
 `)]), cM("shadow-after", [c$1("&::after", `
 box-shadow: inset -10px 0 8px -8px rgba(0, 0, 0, .12);
 `)]), cB("tabs-nav-y-scroll", `
 height: 100%;
 width: 100%;
 overflow-y: auto; 
 scrollbar-width: none;
 `, [c$1("&::-webkit-scrollbar", `
 width: 0;
 height: 0;
 `)]), c$1("&::before, &::after", `
 transition: box-shadow .3s var(--n-bezier);
 pointer-events: none;
 content: "";
 position: absolute;
 top: 0;
 bottom: 0;
 width: 20px;
 z-index: 1;
 `), c$1("&::before", `
 left: 0;
 `), c$1("&::after", `
 right: 0;
 `)]), cB("tabs-nav-scroll-content", `
 display: flex;
 position: relative;
 min-width: 100%;
 width: fit-content;
 `), cB("tabs-wrapper", `
 display: inline-flex;
 flex-wrap: nowrap;
 position: relative;
 `), cB("tabs-tab-wrapper", `
 display: flex;
 flex-wrap: nowrap;
 flex-shrink: 0;
 flex-grow: 0;
 `), cB("tabs-tab", `
 cursor: pointer;
 white-space: nowrap;
 flex-wrap: nowrap;
 display: inline-flex;
 align-items: center;
 color: var(--n-tab-text-color);
 font-size: var(--n-tab-font-size);
 background-clip: padding-box;
 padding: var(--n-tab-padding);
 transition:
 box-shadow .3s var(--n-bezier),
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier),
 border-color .3s var(--n-bezier);
 `, [cM("disabled", {
  cursor: "not-allowed"
}), cE("close", `
 margin-left: 6px;
 transition:
 background-color .3s var(--n-bezier),
 color .3s var(--n-bezier);
 `), cE("label", `
 display: flex;
 align-items: center;
 `)]), cB("tabs-bar", `
 position: absolute;
 bottom: 0;
 height: 2px;
 border-radius: 1px;
 background-color: var(--n-bar-color);
 transition:
 left .2s var(--n-bezier),
 max-width .2s var(--n-bezier),
 background-color .3s var(--n-bezier);
 `, [c$1("&.transition-disabled", `
 transition: none;
 `), cM("disabled", `
 background-color: var(--n-tab-text-color-disabled)
 `)]), cB("tabs-pane-wrapper", `
 position: relative;
 overflow: hidden;
 transition: max-height .2s var(--n-bezier);
 `), cB("tab-pane", `
 color: var(--n-pane-text-color);
 width: 100%;
 padding: var(--n-pane-padding);
 transition:
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier),
 opacity .2s var(--n-bezier);
 left: 0;
 right: 0;
 top: 0;
 `, [c$1("&.next-transition-leave-active, &.prev-transition-leave-active, &.next-transition-enter-active, &.prev-transition-enter-active", `
 transition:
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier),
 transform .2s var(--n-bezier),
 opacity .2s var(--n-bezier);
 `), c$1("&.next-transition-leave-active, &.prev-transition-leave-active", `
 position: absolute;
 `), c$1("&.next-transition-enter-from, &.prev-transition-leave-to", `
 transform: translateX(32px);
 opacity: 0;
 `), c$1("&.next-transition-leave-to, &.prev-transition-enter-from", `
 transform: translateX(-32px);
 opacity: 0;
 `), c$1("&.next-transition-leave-from, &.next-transition-enter-to, &.prev-transition-leave-from, &.prev-transition-enter-to", `
 transform: translateX(0);
 opacity: 1;
 `)]), cB("tabs-tab-pad", `
 width: var(--n-tab-gap);
 flex-grow: 0;
 flex-shrink: 0;
 `), cM("line-type, bar-type", [cB("tabs-tab", `
 font-weight: var(--n-tab-font-weight);
 box-sizing: border-box;
 vertical-align: bottom;
 `, [c$1("&:hover", {
  color: "var(--n-tab-text-color-hover)"
}), cM("active", `
 color: var(--n-tab-text-color-active);
 font-weight: var(--n-tab-font-weight-active);
 `), cM("disabled", {
  color: "var(--n-tab-text-color-disabled)"
})])]), cB("tabs-nav", [cM("line-type", [cE("prefix, suffix", `
 transition: border-color .3s var(--n-bezier);
 border-bottom: 1px solid var(--n-tab-border-color);
 `), cB("tabs-nav-scroll-content", `
 transition: border-color .3s var(--n-bezier);
 border-bottom: 1px solid var(--n-tab-border-color);
 `), cB("tabs-bar", `
 border-radius: 0;
 bottom: -1px;
 `)]), cM("card-type", [cE("prefix, suffix", `
 transition: border-color .3s var(--n-bezier);
 border-bottom: 1px solid var(--n-tab-border-color);
 `), cB("tabs-pad", `
 flex-grow: 1;
 transition: border-color .3s var(--n-bezier);
 border-bottom: 1px solid var(--n-tab-border-color);
 `), cB("tabs-tab-pad", `
 transition: border-color .3s var(--n-bezier);
 border-bottom: 1px solid var(--n-tab-border-color);
 `), cB("tabs-tab", `
 font-weight: var(--n-tab-font-weight);
 border: 1px solid var(--n-tab-border-color);
 border-top-left-radius: var(--n-tab-border-radius);
 border-top-right-radius: var(--n-tab-border-radius);
 background-color: var(--n-tab-color);
 box-sizing: border-box;
 position: relative;
 vertical-align: bottom;
 display: flex;
 justify-content: space-between;
 font-size: var(--n-tab-font-size);
 color: var(--n-tab-text-color);
 `, [cM("addable", `
 padding-left: 8px;
 padding-right: 8px;
 font-size: 16px;
 `, [cE("height-placeholder", `
 width: 0;
 font-size: var(--n-tab-font-size);
 `), cNotM("disabled", [c$1("&:hover", `
 color: var(--n-tab-text-color-hover);
 `)])]), cM("closable", "padding-right: 6px;"), cM("active", `
 border-bottom: 1px solid #0000;
 background-color: #0000;
 font-weight: var(--n-tab-font-weight-active);
 color: var(--n-tab-text-color-active);
 `), cM("disabled", "color: var(--n-tab-text-color-disabled);")]), cB("tabs-scroll-padding", "border-bottom: 1px solid var(--n-tab-border-color);")]), cM("left, right", [cB("tabs-wrapper", `
 flex-direction: column;
 `, [cB("tabs-tab-wrapper", `
 flex-direction: column;
 `, [cB("tabs-tab-pad", `
 height: var(--n-tab-gap);
 width: 100%;
 `)])]), cB("tabs-nav-scroll-content", `
 border-bottom: none;
 `)]), cM("left", [cB("tabs-nav-scroll-content", `
 box-sizing: border-box;
 border-right: 1px solid var(--n-tab-border-color);
 `)]), cM("right", [cB("tabs-nav-scroll-content", `
 border-left: 1px solid var(--n-tab-border-color);
 `)]), cM("bottom", [cB("tabs-nav-scroll-content", `
 border-top: 1px solid var(--n-tab-border-color);
 border-bottom: none;
 `)])])]);
const tabsProps = Object.assign(Object.assign({}, useTheme.props), {
  value: [String, Number],
  defaultValue: [String, Number],
  trigger: {
    type: String,
    default: "click"
  },
  type: {
    type: String,
    default: "bar"
  },
  closable: Boolean,
  justifyContent: String,
  size: {
    type: String,
    default: "medium"
  },
  placement: {
    type: String,
    default: "top"
  },
  tabStyle: [String, Object],
  barWidth: Number,
  paneClass: String,
  paneStyle: [String, Object],
  addable: [Boolean, Object],
  tabsPadding: {
    type: Number,
    default: 0
  },
  animated: Boolean,
  onBeforeLeave: Function,
  onAdd: Function,
  "onUpdate:value": [Function, Array],
  onUpdateValue: [Function, Array],
  onClose: [Function, Array],
  // deprecated
  labelSize: String,
  activeName: [String, Number],
  onActiveNameChange: [Function, Array]
});
const NTabs = defineComponent({
  name: "Tabs",
  props: tabsProps,
  setup(props, { slots }) {
    var _a, _b, _c, _d;
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const themeRef = useTheme("Tabs", "-tabs", style, tabsLight, props, mergedClsPrefixRef);
    const tabsElRef = ref(null);
    const barElRef = ref(null);
    const scrollWrapperElRef = ref(null);
    const addTabInstRef = ref(null);
    const xScrollInstRef = ref(null);
    const leftReachedRef = ref(true);
    const rightReachedRef = ref(true);
    const compitableSizeRef = useCompitable(props, ["labelSize", "size"]);
    const compitableValueRef = useCompitable(props, ["activeName", "value"]);
    const uncontrolledValueRef = ref((_b = (_a = compitableValueRef.value) !== null && _a !== void 0 ? _a : props.defaultValue) !== null && _b !== void 0 ? _b : slots.default ? (_d = (_c = flatten(slots.default())[0]) === null || _c === void 0 ? void 0 : _c.props) === null || _d === void 0 ? void 0 : _d.name : null);
    const mergedValueRef = useMergedState(compitableValueRef, uncontrolledValueRef);
    const tabChangeIdRef = { id: 0 };
    const tabWrapperStyleRef = computed(() => {
      if (!props.justifyContent || props.type === "card")
        return void 0;
      return {
        display: "flex",
        justifyContent: props.justifyContent
      };
    });
    watch(mergedValueRef, () => {
      tabChangeIdRef.id = 0;
      updateCurrentBarStyle();
      updateCurrentScrollPosition();
    });
    function getCurrentEl() {
      var _a2;
      const { value } = mergedValueRef;
      if (value === null)
        return null;
      const tabEl = (_a2 = tabsElRef.value) === null || _a2 === void 0 ? void 0 : _a2.querySelector(`[data-name="${value}"]`);
      return tabEl;
    }
    function updateBarStyle(tabEl) {
      if (props.type === "card")
        return;
      const { value: barEl } = barElRef;
      if (!barEl)
        return;
      if (tabEl) {
        const disabledClassName = `${mergedClsPrefixRef.value}-tabs-bar--disabled`;
        const { barWidth, placement } = props;
        if (tabEl.dataset.disabled === "true") {
          barEl.classList.add(disabledClassName);
        } else {
          barEl.classList.remove(disabledClassName);
        }
        if (["top", "bottom"].includes(placement)) {
          clearBarStyle(["top", "maxHeight", "height"]);
          if (typeof barWidth === "number" && tabEl.offsetWidth >= barWidth) {
            const offsetDiffLeft = Math.floor((tabEl.offsetWidth - barWidth) / 2) + tabEl.offsetLeft;
            barEl.style.left = `${offsetDiffLeft}px`;
            barEl.style.maxWidth = `${barWidth}px`;
          } else {
            barEl.style.left = `${tabEl.offsetLeft}px`;
            barEl.style.maxWidth = `${tabEl.offsetWidth}px`;
          }
          barEl.style.width = "8192px";
          void barEl.offsetWidth;
        } else {
          clearBarStyle(["left", "maxWidth", "width"]);
          if (typeof barWidth === "number" && tabEl.offsetHeight >= barWidth) {
            const offsetDiffTop = Math.floor((tabEl.offsetHeight - barWidth) / 2) + tabEl.offsetTop;
            barEl.style.top = `${offsetDiffTop}px`;
            barEl.style.maxHeight = `${barWidth}px`;
          } else {
            barEl.style.top = `${tabEl.offsetTop}px`;
            barEl.style.maxHeight = `${tabEl.offsetHeight}px`;
          }
          barEl.style.height = "8192px";
          void barEl.offsetHeight;
        }
      }
    }
    function clearBarStyle(styleProps) {
      const { value: barEl } = barElRef;
      if (!barEl)
        return;
      for (const prop of styleProps) {
        barEl.style[prop] = "";
      }
    }
    function updateCurrentBarStyle() {
      if (props.type === "card")
        return;
      const tabEl = getCurrentEl();
      if (tabEl) {
        updateBarStyle(tabEl);
      }
    }
    function updateCurrentScrollPosition(smooth) {
      var _a2;
      const scrollWrapperEl = (_a2 = xScrollInstRef.value) === null || _a2 === void 0 ? void 0 : _a2.$el;
      if (!scrollWrapperEl)
        return;
      const tabEl = getCurrentEl();
      if (!tabEl)
        return;
      const { scrollLeft: scrollWrapperElScrollLeft, offsetWidth: scrollWrapperElOffsetWidth } = scrollWrapperEl;
      const { offsetLeft: tabElOffsetLeft, offsetWidth: tabElOffsetWidth } = tabEl;
      if (scrollWrapperElScrollLeft > tabElOffsetLeft) {
        scrollWrapperEl.scrollTo({
          top: 0,
          left: tabElOffsetLeft,
          behavior: "smooth"
        });
      } else if (tabElOffsetLeft + tabElOffsetWidth > scrollWrapperElScrollLeft + scrollWrapperElOffsetWidth) {
        scrollWrapperEl.scrollTo({
          top: 0,
          left: tabElOffsetLeft + tabElOffsetWidth - scrollWrapperElOffsetWidth,
          behavior: "smooth"
        });
      }
    }
    const tabsPaneWrapperRef = ref(null);
    let fromHeight = 0;
    let hangingTransition = null;
    function onAnimationBeforeLeave(el) {
      const tabsPaneWrapperEl = tabsPaneWrapperRef.value;
      if (tabsPaneWrapperEl) {
        fromHeight = el.getBoundingClientRect().height;
        const fromHeightPx = `${fromHeight}px`;
        const applyFromStyle = () => {
          tabsPaneWrapperEl.style.height = fromHeightPx;
          tabsPaneWrapperEl.style.maxHeight = fromHeightPx;
        };
        if (!hangingTransition) {
          hangingTransition = applyFromStyle;
        } else {
          applyFromStyle();
          hangingTransition();
          hangingTransition = null;
        }
      }
    }
    function onAnimationEnter(el) {
      const tabsPaneWrapperEl = tabsPaneWrapperRef.value;
      if (tabsPaneWrapperEl) {
        const targetHeight = el.getBoundingClientRect().height;
        const applyTargetStyle = () => {
          void document.body.offsetHeight;
          tabsPaneWrapperEl.style.maxHeight = `${targetHeight}px`;
          tabsPaneWrapperEl.style.height = `${Math.max(fromHeight, targetHeight)}px`;
        };
        if (!hangingTransition) {
          hangingTransition = applyTargetStyle;
        } else {
          hangingTransition();
          hangingTransition = null;
          applyTargetStyle();
        }
      }
    }
    function onAnimationAfterEnter() {
      const tabsPaneWrapperEl = tabsPaneWrapperRef.value;
      if (tabsPaneWrapperEl) {
        tabsPaneWrapperEl.style.maxHeight = "";
        tabsPaneWrapperEl.style.height = "";
      }
    }
    const renderNameListRef = { value: [] };
    const animationDirectionRef = ref("next");
    function activateTab(panelName) {
      const currentValue = mergedValueRef.value;
      let dir = "next";
      for (const name of renderNameListRef.value) {
        if (name === currentValue) {
          break;
        }
        if (name === panelName) {
          dir = "prev";
          break;
        }
      }
      animationDirectionRef.value = dir;
      doUpdateValue(panelName);
    }
    function doUpdateValue(panelName) {
      const { onActiveNameChange, onUpdateValue, "onUpdate:value": _onUpdateValue } = props;
      if (onActiveNameChange) {
        call(onActiveNameChange, panelName);
      }
      if (onUpdateValue)
        call(onUpdateValue, panelName);
      if (_onUpdateValue)
        call(_onUpdateValue, panelName);
      uncontrolledValueRef.value = panelName;
    }
    function handleClose(panelName) {
      const { onClose } = props;
      if (onClose)
        call(onClose, panelName);
    }
    function updateBarPositionInstantly() {
      const { value: barEl } = barElRef;
      if (!barEl)
        return;
      const disableTransitionClassName = "transition-disabled";
      barEl.classList.add(disableTransitionClassName);
      updateCurrentBarStyle();
      barEl.classList.remove(disableTransitionClassName);
    }
    let memorizedWidth = 0;
    function _handleNavResize(entry) {
      var _b2;
      if (entry.contentRect.width === 0 && entry.contentRect.height === 0) {
        return;
      }
      if (memorizedWidth === entry.contentRect.width) {
        return;
      }
      memorizedWidth = entry.contentRect.width;
      const { type } = props;
      if (type === "line" || type === "bar") {
        {
          updateBarPositionInstantly();
        }
      }
      if (type !== "segment") {
        deriveScrollShadow((_b2 = xScrollInstRef.value) === null || _b2 === void 0 ? void 0 : _b2.$el);
      }
    }
    const handleNavResize = throttle(_handleNavResize, 64);
    watch([() => props.justifyContent, () => props.size], () => {
      void nextTick(() => {
        const { type } = props;
        if (type === "line" || type === "bar") {
          updateBarPositionInstantly();
        }
      });
    });
    const addTabFixedRef = ref(false);
    function _handleTabsResize(entry) {
      var _a2;
      const { target, contentRect: { width } } = entry;
      const containerWidth = target.parentElement.offsetWidth;
      if (!addTabFixedRef.value) {
        if (containerWidth < width) {
          addTabFixedRef.value = true;
        }
      } else {
        const { value: addTabInst } = addTabInstRef;
        if (!addTabInst)
          return;
        if (containerWidth - width > addTabInst.$el.offsetWidth) {
          addTabFixedRef.value = false;
        }
      }
      deriveScrollShadow((_a2 = xScrollInstRef.value) === null || _a2 === void 0 ? void 0 : _a2.$el);
    }
    const handleTabsResize = throttle(_handleTabsResize, 64);
    function handleAdd() {
      const { onAdd } = props;
      if (onAdd)
        onAdd();
      void nextTick(() => {
        const currentEl = getCurrentEl();
        const { value: xScrollInst } = xScrollInstRef;
        if (!currentEl || !xScrollInst)
          return;
        xScrollInst.scrollTo({
          left: currentEl.offsetLeft,
          top: 0,
          behavior: "smooth"
        });
      });
    }
    function deriveScrollShadow(el) {
      if (!el)
        return;
      const { scrollLeft, scrollWidth, offsetWidth } = el;
      leftReachedRef.value = scrollLeft <= 0;
      rightReachedRef.value = scrollLeft + offsetWidth >= scrollWidth;
    }
    const handleScroll = throttle((e) => {
      deriveScrollShadow(e.target);
    }, 64);
    provide(tabsInjectionKey, {
      triggerRef: toRef(props, "trigger"),
      tabStyleRef: toRef(props, "tabStyle"),
      paneClassRef: toRef(props, "paneClass"),
      paneStyleRef: toRef(props, "paneStyle"),
      mergedClsPrefixRef,
      typeRef: toRef(props, "type"),
      closableRef: toRef(props, "closable"),
      valueRef: mergedValueRef,
      tabChangeIdRef,
      onBeforeLeaveRef: toRef(props, "onBeforeLeave"),
      activateTab,
      handleClose,
      handleAdd
    });
    onFontsReady(() => {
      updateCurrentBarStyle();
      updateCurrentScrollPosition();
    });
    watchEffect(() => {
      const { value: el } = scrollWrapperElRef;
      if (!el || ["left", "right"].includes(props.placement))
        return;
      const { value: clsPrefix } = mergedClsPrefixRef;
      const shadowBeforeClass = `${clsPrefix}-tabs-nav-scroll-wrapper--shadow-before`;
      const shadowAfterClass = `${clsPrefix}-tabs-nav-scroll-wrapper--shadow-after`;
      if (leftReachedRef.value) {
        el.classList.remove(shadowBeforeClass);
      } else {
        el.classList.add(shadowBeforeClass);
      }
      if (rightReachedRef.value) {
        el.classList.remove(shadowAfterClass);
      } else {
        el.classList.add(shadowAfterClass);
      }
    });
    const tabsRailElRef = ref(null);
    watch(mergedValueRef, () => {
      if (props.type === "segment") {
        const tabsRailEl = tabsRailElRef.value;
        if (tabsRailEl) {
          void nextTick(() => {
            tabsRailEl.classList.add("transition-disabled");
            void tabsRailEl.offsetWidth;
            tabsRailEl.classList.remove("transition-disabled");
          });
        }
      }
    });
    const exposedMethods = {
      syncBarPosition: () => {
        updateCurrentBarStyle();
      }
    };
    const cssVarsRef = computed(() => {
      const { value: size } = compitableSizeRef;
      const { type } = props;
      const typeSuffix = {
        card: "Card",
        bar: "Bar",
        line: "Line",
        segment: "Segment"
      }[type];
      const sizeType = `${size}${typeSuffix}`;
      const { self: { barColor, closeIconColor, closeIconColorHover, closeIconColorPressed, tabColor, tabBorderColor, paneTextColor, tabFontWeight, tabBorderRadius, tabFontWeightActive, colorSegment, fontWeightStrong, tabColorSegment, closeSize, closeIconSize, closeColorHover, closeColorPressed, closeBorderRadius, [createKey("panePadding", size)]: panePadding, [createKey("tabPadding", sizeType)]: tabPadding, [createKey("tabPaddingVertical", sizeType)]: tabPaddingVertical, [createKey("tabGap", sizeType)]: tabGap, [createKey("tabTextColor", type)]: tabTextColor, [createKey("tabTextColorActive", type)]: tabTextColorActive, [createKey("tabTextColorHover", type)]: tabTextColorHover, [createKey("tabTextColorDisabled", type)]: tabTextColorDisabled, [createKey("tabFontSize", size)]: tabFontSize }, common: { cubicBezierEaseInOut } } = themeRef.value;
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-color-segment": colorSegment,
        "--n-bar-color": barColor,
        "--n-tab-font-size": tabFontSize,
        "--n-tab-text-color": tabTextColor,
        "--n-tab-text-color-active": tabTextColorActive,
        "--n-tab-text-color-disabled": tabTextColorDisabled,
        "--n-tab-text-color-hover": tabTextColorHover,
        "--n-pane-text-color": paneTextColor,
        "--n-tab-border-color": tabBorderColor,
        "--n-tab-border-radius": tabBorderRadius,
        "--n-close-size": closeSize,
        "--n-close-icon-size": closeIconSize,
        "--n-close-color-hover": closeColorHover,
        "--n-close-color-pressed": closeColorPressed,
        "--n-close-border-radius": closeBorderRadius,
        "--n-close-icon-color": closeIconColor,
        "--n-close-icon-color-hover": closeIconColorHover,
        "--n-close-icon-color-pressed": closeIconColorPressed,
        "--n-tab-color": tabColor,
        "--n-tab-font-weight": tabFontWeight,
        "--n-tab-font-weight-active": tabFontWeightActive,
        "--n-tab-padding": tabPadding,
        "--n-tab-padding-vertical": tabPaddingVertical,
        "--n-tab-gap": tabGap,
        "--n-pane-padding": panePadding,
        "--n-font-weight-strong": fontWeightStrong,
        "--n-tab-color-segment": tabColorSegment
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("tabs", computed(() => {
      return `${compitableSizeRef.value[0]}${props.type[0]}`;
    }), cssVarsRef, props) : void 0;
    return Object.assign({
      mergedClsPrefix: mergedClsPrefixRef,
      mergedValue: mergedValueRef,
      renderedNames: /* @__PURE__ */ new Set(),
      tabsRailElRef,
      tabsPaneWrapperRef,
      tabsElRef,
      barElRef,
      addTabInstRef,
      xScrollInstRef,
      scrollWrapperElRef,
      addTabFixed: addTabFixedRef,
      tabWrapperStyle: tabWrapperStyleRef,
      handleNavResize,
      mergedSize: compitableSizeRef,
      handleScroll,
      handleTabsResize,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      animationDirection: animationDirectionRef,
      renderNameListRef,
      onAnimationBeforeLeave,
      onAnimationEnter,
      onAnimationAfterEnter,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    }, exposedMethods);
  },
  render() {
    const { mergedClsPrefix, type, placement, addTabFixed, addable, mergedSize, renderNameListRef, onRender, $slots: { default: defaultSlot, prefix: prefixSlot, suffix: suffixSlot } } = this;
    onRender === null || onRender === void 0 ? void 0 : onRender();
    const tabPaneChildren = defaultSlot ? flatten(defaultSlot()).filter((v) => {
      return v.type.__TAB_PANE__ === true;
    }) : [];
    const tabChildren = defaultSlot ? flatten(defaultSlot()).filter((v) => {
      return v.type.__TAB__ === true;
    }) : [];
    const showPane = !tabChildren.length;
    const isCard = type === "card";
    const isSegment = type === "segment";
    const mergedJustifyContent = !isCard && !isSegment && this.justifyContent;
    renderNameListRef.value = [];
    const scrollContent = () => {
      const tabs = h(
        "div",
        { style: this.tabWrapperStyle, class: [`${mergedClsPrefix}-tabs-wrapper`] },
        mergedJustifyContent ? null : h("div", { class: `${mergedClsPrefix}-tabs-scroll-padding`, style: { width: `${this.tabsPadding}px` } }),
        showPane ? tabPaneChildren.map((tabPaneVNode, index) => {
          renderNameListRef.value.push(tabPaneVNode.props.name);
          return justifyTabDynamicProps(h(Tab, Object.assign({}, tabPaneVNode.props, { internalCreatedByPane: true, internalLeftPadded: index !== 0 && (!mergedJustifyContent || mergedJustifyContent === "center" || mergedJustifyContent === "start" || mergedJustifyContent === "end") }), tabPaneVNode.children ? {
            default: tabPaneVNode.children.tab
          } : void 0));
        }) : tabChildren.map((tabVNode, index) => {
          renderNameListRef.value.push(tabVNode.props.name);
          if (index !== 0 && !mergedJustifyContent) {
            return justifyTabDynamicProps(createLeftPaddedTabVNode(tabVNode));
          } else {
            return justifyTabDynamicProps(tabVNode);
          }
        }),
        !addTabFixed && addable && isCard ? createAddTag(addable, (showPane ? tabPaneChildren.length : tabChildren.length) !== 0) : null,
        mergedJustifyContent ? null : h("div", { class: `${mergedClsPrefix}-tabs-scroll-padding`, style: { width: `${this.tabsPadding}px` } })
      );
      return h(
        "div",
        { ref: "tabsElRef", class: `${mergedClsPrefix}-tabs-nav-scroll-content` },
        isCard && addable ? h(VResizeObserver, { onResize: this.handleTabsResize }, {
          default: () => tabs
        }) : tabs,
        isCard ? h("div", { class: `${mergedClsPrefix}-tabs-pad` }) : null,
        isCard ? null : h("div", { ref: "barElRef", class: `${mergedClsPrefix}-tabs-bar` })
      );
    };
    return h(
      "div",
      { class: [
        `${mergedClsPrefix}-tabs`,
        this.themeClass,
        `${mergedClsPrefix}-tabs--${type}-type`,
        `${mergedClsPrefix}-tabs--${mergedSize}-size`,
        mergedJustifyContent && `${mergedClsPrefix}-tabs--flex`,
        `${mergedClsPrefix}-tabs--${placement}`
      ], style: this.cssVars },
      h(
        "div",
        { class: [
          // the class should be applied here since it's possible
          // to make tabs nested in tabs, style may influence each
          // other. adding a class will make it easy to write the
          // style.
          `${mergedClsPrefix}-tabs-nav--${type}-type`,
          `${mergedClsPrefix}-tabs-nav--${placement}`,
          `${mergedClsPrefix}-tabs-nav`
        ] },
        resolveWrappedSlot(prefixSlot, (children) => children && h("div", { class: `${mergedClsPrefix}-tabs-nav__prefix` }, children)),
        isSegment ? h("div", { class: `${mergedClsPrefix}-tabs-rail`, ref: "tabsRailElRef" }, showPane ? tabPaneChildren.map((tabPaneVNode, index) => {
          renderNameListRef.value.push(tabPaneVNode.props.name);
          return h(Tab, Object.assign({}, tabPaneVNode.props, { internalCreatedByPane: true, internalLeftPadded: index !== 0 }), tabPaneVNode.children ? {
            default: tabPaneVNode.children.tab
          } : void 0);
        }) : tabChildren.map((tabVNode, index) => {
          renderNameListRef.value.push(tabVNode.props.name);
          if (index === 0) {
            return tabVNode;
          } else {
            return createLeftPaddedTabVNode(tabVNode);
          }
        })) : h(VResizeObserver, { onResize: this.handleNavResize }, {
          default: () => h("div", { class: `${mergedClsPrefix}-tabs-nav-scroll-wrapper`, ref: "scrollWrapperElRef" }, ["top", "bottom"].includes(placement) ? h(VXScroll, { ref: "xScrollInstRef", onScroll: this.handleScroll }, {
            default: scrollContent
          }) : h("div", { class: `${mergedClsPrefix}-tabs-nav-y-scroll` }, scrollContent()))
        }),
        addTabFixed && addable && isCard ? createAddTag(addable, true) : null,
        resolveWrappedSlot(suffixSlot, (children) => children && h("div", { class: `${mergedClsPrefix}-tabs-nav__suffix` }, children))
      ),
      showPane && (this.animated ? h("div", { ref: "tabsPaneWrapperRef", class: `${mergedClsPrefix}-tabs-pane-wrapper` }, filterMapTabPanes(tabPaneChildren, this.mergedValue, this.renderedNames, this.onAnimationBeforeLeave, this.onAnimationEnter, this.onAnimationAfterEnter, this.animationDirection)) : filterMapTabPanes(tabPaneChildren, this.mergedValue, this.renderedNames))
    );
  }
});
function filterMapTabPanes(tabPaneVNodes, value, renderedNames, onBeforeLeave, onEnter, onAfterEnter, animationDirection) {
  const children = [];
  tabPaneVNodes.forEach((vNode) => {
    const { name, displayDirective, "display-directive": _displayDirective } = vNode.props;
    const matchDisplayDirective = (directive) => displayDirective === directive || _displayDirective === directive;
    const show = value === name;
    if (vNode.key !== void 0) {
      vNode.key = name;
    }
    if (show || matchDisplayDirective("show") || matchDisplayDirective("show:lazy") && renderedNames.has(name)) {
      if (!renderedNames.has(name)) {
        renderedNames.add(name);
      }
      const useVShow = !matchDisplayDirective("if");
      children.push(useVShow ? withDirectives(vNode, [[vShow, show]]) : vNode);
    }
  });
  if (!animationDirection) {
    return children;
  }
  return h(TransitionGroup, { name: `${animationDirection}-transition`, onBeforeLeave, onEnter, onAfterEnter }, { default: () => children });
}
function createAddTag(addable, internalLeftPadded) {
  return h(Tab, { ref: "addTabInstRef", key: "__addable", name: "__addable", internalCreatedByPane: true, internalAddable: true, internalLeftPadded, disabled: typeof addable === "object" && addable.disabled });
}
function createLeftPaddedTabVNode(tabVNode) {
  const modifiedVNode = cloneVNode(tabVNode);
  if (modifiedVNode.props) {
    modifiedVNode.props.internalLeftPadded = true;
  } else {
    modifiedVNode.props = {
      internalLeftPadded: true
    };
  }
  return modifiedVNode;
}
function justifyTabDynamicProps(tabVNode) {
  if (Array.isArray(tabVNode.dynamicProps)) {
    if (!tabVNode.dynamicProps.includes("internalLeftPadded")) {
      tabVNode.dynamicProps.push("internalLeftPadded");
    }
  } else {
    tabVNode.dynamicProps = ["internalLeftPadded"];
  }
  return tabVNode;
}
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
