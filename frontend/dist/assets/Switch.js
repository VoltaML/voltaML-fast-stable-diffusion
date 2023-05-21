import { Y as cB, Z as cE, a0 as iconSwitchTransition, X as c, $ as cM, aq as cNotM, d as defineComponent, Q as useConfig, a5 as useTheme, R as useFormItem, E as ref, U as toRef, S as useMergedState, c as computed, a9 as useThemeClass, bC as isSlotEmpty, D as h, aB as resolveWrappedSlot, bD as switchLight, a8 as createKey, aL as pxfy, aE as depx, ab as NIconSwitchTransition, aN as NBaseLoading, W as call } from "./index.js";
const style = cB("switch", `
 height: var(--n-height);
 min-width: var(--n-width);
 vertical-align: middle;
 user-select: none;
 -webkit-user-select: none;
 display: inline-flex;
 outline: none;
 justify-content: center;
 align-items: center;
`, [cE("children-placeholder", `
 height: var(--n-rail-height);
 display: flex;
 flex-direction: column;
 overflow: hidden;
 pointer-events: none;
 visibility: hidden;
 `), cE("rail-placeholder", `
 display: flex;
 flex-wrap: none;
 `), cE("button-placeholder", `
 width: calc(1.75 * var(--n-rail-height));
 height: var(--n-rail-height);
 `), cB("base-loading", `
 position: absolute;
 top: 50%;
 left: 50%;
 transform: translateX(-50%) translateY(-50%);
 font-size: calc(var(--n-button-width) - 4px);
 color: var(--n-loading-color);
 transition: color .3s var(--n-bezier);
 `, [iconSwitchTransition({
  left: "50%",
  top: "50%",
  originalTransform: "translateX(-50%) translateY(-50%)"
})]), cE("checked, unchecked", `
 transition: color .3s var(--n-bezier);
 color: var(--n-text-color);
 box-sizing: border-box;
 position: absolute;
 white-space: nowrap;
 top: 0;
 bottom: 0;
 display: flex;
 align-items: center;
 line-height: 1;
 `), cE("checked", `
 right: 0;
 padding-right: calc(1.25 * var(--n-rail-height) - var(--n-offset));
 `), cE("unchecked", `
 left: 0;
 justify-content: flex-end;
 padding-left: calc(1.25 * var(--n-rail-height) - var(--n-offset));
 `), c("&:focus", [cE("rail", `
 box-shadow: var(--n-box-shadow-focus);
 `)]), cM("round", [cE("rail", "border-radius: calc(var(--n-rail-height) / 2);", [cE("button", "border-radius: calc(var(--n-button-height) / 2);")])]), cNotM("disabled", [cNotM("icon", [cM("rubber-band", [cM("pressed", [cE("rail", [cE("button", "max-width: var(--n-button-width-pressed);")])]), cE("rail", [c("&:active", [cE("button", "max-width: var(--n-button-width-pressed);")])]), cM("active", [cM("pressed", [cE("rail", [cE("button", "left: calc(100% - var(--n-offset) - var(--n-button-width-pressed));")])]), cE("rail", [c("&:active", [cE("button", "left: calc(100% - var(--n-offset) - var(--n-button-width-pressed));")])])])])])]), cM("active", [cE("rail", [cE("button", "left: calc(100% - var(--n-button-width) - var(--n-offset))")])]), cE("rail", `
 overflow: hidden;
 height: var(--n-rail-height);
 min-width: var(--n-rail-width);
 border-radius: var(--n-rail-border-radius);
 cursor: pointer;
 position: relative;
 transition:
 opacity .3s var(--n-bezier),
 background .3s var(--n-bezier),
 box-shadow .3s var(--n-bezier);
 background-color: var(--n-rail-color);
 `, [cE("button-icon", `
 color: var(--n-icon-color);
 transition: color .3s var(--n-bezier);
 font-size: calc(var(--n-button-height) - 4px);
 position: absolute;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 display: flex;
 justify-content: center;
 align-items: center;
 line-height: 1;
 `, [iconSwitchTransition()]), cE("button", `
 align-items: center; 
 top: var(--n-offset);
 left: var(--n-offset);
 height: var(--n-button-height);
 width: var(--n-button-width-pressed);
 max-width: var(--n-button-width);
 border-radius: var(--n-button-border-radius);
 background-color: var(--n-button-color);
 box-shadow: var(--n-button-box-shadow);
 box-sizing: border-box;
 cursor: inherit;
 content: "";
 position: absolute;
 transition:
 background-color .3s var(--n-bezier),
 left .3s var(--n-bezier),
 opacity .3s var(--n-bezier),
 max-width .3s var(--n-bezier),
 box-shadow .3s var(--n-bezier);
 `)]), cM("active", [cE("rail", "background-color: var(--n-rail-color-active);")]), cM("loading", [cE("rail", `
 cursor: wait;
 `)]), cM("disabled", [cE("rail", `
 cursor: not-allowed;
 opacity: .5;
 `)])]);
const switchProps = Object.assign(Object.assign({}, useTheme.props), {
  size: {
    type: String,
    default: "medium"
  },
  value: {
    type: [String, Number, Boolean],
    default: void 0
  },
  loading: Boolean,
  defaultValue: {
    type: [String, Number, Boolean],
    default: false
  },
  disabled: {
    type: Boolean,
    default: void 0
  },
  round: {
    type: Boolean,
    default: true
  },
  "onUpdate:value": [Function, Array],
  onUpdateValue: [Function, Array],
  checkedValue: {
    type: [String, Number, Boolean],
    default: true
  },
  uncheckedValue: {
    type: [String, Number, Boolean],
    default: false
  },
  railStyle: Function,
  rubberBand: {
    type: Boolean,
    default: true
  },
  /** @deprecated */
  onChange: [Function, Array]
});
let supportCssMax;
const NSwitch = defineComponent({
  name: "Switch",
  props: switchProps,
  setup(props) {
    if (supportCssMax === void 0) {
      if (typeof CSS !== "undefined") {
        if (typeof CSS.supports !== "undefined") {
          supportCssMax = CSS.supports("width", "max(1px)");
        } else {
          supportCssMax = false;
        }
      } else {
        supportCssMax = true;
      }
    }
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const themeRef = useTheme("Switch", "-switch", style, switchLight, props, mergedClsPrefixRef);
    const formItem = useFormItem(props);
    const { mergedSizeRef, mergedDisabledRef } = formItem;
    const uncontrolledValueRef = ref(props.defaultValue);
    const controlledValueRef = toRef(props, "value");
    const mergedValueRef = useMergedState(controlledValueRef, uncontrolledValueRef);
    const checkedRef = computed(() => {
      return mergedValueRef.value === props.checkedValue;
    });
    const pressedRef = ref(false);
    const focusedRef = ref(false);
    const mergedRailStyleRef = computed(() => {
      const { railStyle } = props;
      if (!railStyle)
        return void 0;
      return railStyle({ focused: focusedRef.value, checked: checkedRef.value });
    });
    function doUpdateValue(value) {
      const { "onUpdate:value": _onUpdateValue, onChange, onUpdateValue } = props;
      const { nTriggerFormInput, nTriggerFormChange } = formItem;
      if (_onUpdateValue)
        call(_onUpdateValue, value);
      if (onUpdateValue)
        call(onUpdateValue, value);
      if (onChange)
        call(onChange, value);
      uncontrolledValueRef.value = value;
      nTriggerFormInput();
      nTriggerFormChange();
    }
    function doFocus() {
      const { nTriggerFormFocus } = formItem;
      nTriggerFormFocus();
    }
    function doBlur() {
      const { nTriggerFormBlur } = formItem;
      nTriggerFormBlur();
    }
    function handleClick() {
      if (props.loading || mergedDisabledRef.value)
        return;
      if (mergedValueRef.value !== props.checkedValue) {
        doUpdateValue(props.checkedValue);
      } else {
        doUpdateValue(props.uncheckedValue);
      }
    }
    function handleFocus() {
      focusedRef.value = true;
      doFocus();
    }
    function handleBlur() {
      focusedRef.value = false;
      doBlur();
      pressedRef.value = false;
    }
    function handleKeyup(e) {
      if (props.loading || mergedDisabledRef.value)
        return;
      if (e.key === " ") {
        if (mergedValueRef.value !== props.checkedValue) {
          doUpdateValue(props.checkedValue);
        } else {
          doUpdateValue(props.uncheckedValue);
        }
        pressedRef.value = false;
      }
    }
    function handleKeydown(e) {
      if (props.loading || mergedDisabledRef.value)
        return;
      if (e.key === " ") {
        e.preventDefault();
        pressedRef.value = true;
      }
    }
    const cssVarsRef = computed(() => {
      const { value: size } = mergedSizeRef;
      const { self: { opacityDisabled, railColor, railColorActive, buttonBoxShadow, buttonColor, boxShadowFocus, loadingColor, textColor, iconColor, [createKey("buttonHeight", size)]: buttonHeight, [createKey("buttonWidth", size)]: buttonWidth, [createKey("buttonWidthPressed", size)]: buttonWidthPressed, [createKey("railHeight", size)]: railHeight, [createKey("railWidth", size)]: railWidth, [createKey("railBorderRadius", size)]: railBorderRadius, [createKey("buttonBorderRadius", size)]: buttonBorderRadius }, common: { cubicBezierEaseInOut } } = themeRef.value;
      let offset;
      let height;
      let width;
      if (supportCssMax) {
        offset = `calc((${railHeight} - ${buttonHeight}) / 2)`;
        height = `max(${railHeight}, ${buttonHeight})`;
        width = `max(${railWidth}, calc(${railWidth} + ${buttonHeight} - ${railHeight}))`;
      } else {
        offset = pxfy((depx(railHeight) - depx(buttonHeight)) / 2);
        height = pxfy(Math.max(depx(railHeight), depx(buttonHeight)));
        width = depx(railHeight) > depx(buttonHeight) ? railWidth : pxfy(depx(railWidth) + depx(buttonHeight) - depx(railHeight));
      }
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-button-border-radius": buttonBorderRadius,
        "--n-button-box-shadow": buttonBoxShadow,
        "--n-button-color": buttonColor,
        "--n-button-width": buttonWidth,
        "--n-button-width-pressed": buttonWidthPressed,
        "--n-button-height": buttonHeight,
        "--n-height": height,
        "--n-offset": offset,
        "--n-opacity-disabled": opacityDisabled,
        "--n-rail-border-radius": railBorderRadius,
        "--n-rail-color": railColor,
        "--n-rail-color-active": railColorActive,
        "--n-rail-height": railHeight,
        "--n-rail-width": railWidth,
        "--n-width": width,
        "--n-box-shadow-focus": boxShadowFocus,
        "--n-loading-color": loadingColor,
        "--n-text-color": textColor,
        "--n-icon-color": iconColor
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("switch", computed(() => {
      return mergedSizeRef.value[0];
    }), cssVarsRef, props) : void 0;
    return {
      handleClick,
      handleBlur,
      handleFocus,
      handleKeyup,
      handleKeydown,
      mergedRailStyle: mergedRailStyleRef,
      pressed: pressedRef,
      mergedClsPrefix: mergedClsPrefixRef,
      mergedValue: mergedValueRef,
      checked: checkedRef,
      mergedDisabled: mergedDisabledRef,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    const { mergedClsPrefix, mergedDisabled, checked, mergedRailStyle, onRender, $slots } = this;
    onRender === null || onRender === void 0 ? void 0 : onRender();
    const { checked: checkedSlot, unchecked: uncheckedSlot, icon: iconSlot, "checked-icon": checkedIconSlot, "unchecked-icon": uncheckedIconSlot } = $slots;
    const hasIcon = !(isSlotEmpty(iconSlot) && isSlotEmpty(checkedIconSlot) && isSlotEmpty(uncheckedIconSlot));
    return h(
      "div",
      { role: "switch", "aria-checked": checked, class: [
        `${mergedClsPrefix}-switch`,
        this.themeClass,
        hasIcon && `${mergedClsPrefix}-switch--icon`,
        checked && `${mergedClsPrefix}-switch--active`,
        mergedDisabled && `${mergedClsPrefix}-switch--disabled`,
        this.round && `${mergedClsPrefix}-switch--round`,
        this.loading && `${mergedClsPrefix}-switch--loading`,
        this.pressed && `${mergedClsPrefix}-switch--pressed`,
        this.rubberBand && `${mergedClsPrefix}-switch--rubber-band`
      ], tabindex: !this.mergedDisabled ? 0 : void 0, style: this.cssVars, onClick: this.handleClick, onFocus: this.handleFocus, onBlur: this.handleBlur, onKeyup: this.handleKeyup, onKeydown: this.handleKeydown },
      h(
        "div",
        { class: `${mergedClsPrefix}-switch__rail`, "aria-hidden": "true", style: mergedRailStyle },
        resolveWrappedSlot(checkedSlot, (checkedSlotChildren) => resolveWrappedSlot(uncheckedSlot, (uncheckedSlotChildren) => {
          if (checkedSlotChildren || uncheckedSlotChildren) {
            return h(
              "div",
              { "aria-hidden": true, class: `${mergedClsPrefix}-switch__children-placeholder` },
              h(
                "div",
                { class: `${mergedClsPrefix}-switch__rail-placeholder` },
                h("div", { class: `${mergedClsPrefix}-switch__button-placeholder` }),
                checkedSlotChildren
              ),
              h(
                "div",
                { class: `${mergedClsPrefix}-switch__rail-placeholder` },
                h("div", { class: `${mergedClsPrefix}-switch__button-placeholder` }),
                uncheckedSlotChildren
              )
            );
          }
          return null;
        })),
        h(
          "div",
          { class: `${mergedClsPrefix}-switch__button` },
          resolveWrappedSlot(iconSlot, (icon) => resolveWrappedSlot(checkedIconSlot, (checkedIcon) => resolveWrappedSlot(uncheckedIconSlot, (uncheckedIcon) => {
            return h(NIconSwitchTransition, null, {
              default: () => this.loading ? h(NBaseLoading, { key: "loading", clsPrefix: mergedClsPrefix, strokeWidth: 20 }) : this.checked && (checkedIcon || icon) ? h("div", { class: `${mergedClsPrefix}-switch__button-icon`, key: checkedIcon ? "checked-icon" : "icon" }, checkedIcon || icon) : !this.checked && (uncheckedIcon || icon) ? h("div", { class: `${mergedClsPrefix}-switch__button-icon`, key: uncheckedIcon ? "unchecked-icon" : "icon" }, uncheckedIcon || icon) : null
            });
          }))),
          resolveWrappedSlot(checkedSlot, (children) => children && h("div", { key: "checked", class: `${mergedClsPrefix}-switch__checked` }, children)),
          resolveWrappedSlot(uncheckedSlot, (children) => children && h("div", { key: "unchecked", class: `${mergedClsPrefix}-switch__unchecked` }, children))
        )
      )
    );
  }
});
export {
  NSwitch as N
};
