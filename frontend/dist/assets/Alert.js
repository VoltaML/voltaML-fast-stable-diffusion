import { am as commonLight, an as commonVars, ao as composite, ap as changeColor, X as cB, $ as cE, Y as cM, aq as fadeInHeightExpandTransition, Z as c, d as defineComponent, a1 as useConfig, a2 as useTheme, ar as useRtl, Q as computed, ab as useThemeClass, r as ref, G as h, R as mergeProps, V as NBaseClose, as as resolveSlot, ac as resolveWrappedSlot, at as NFadeInExpandTransition, au as getMargin, ah as createKey, S as NBaseIcon, av as ErrorIcon, aw as WarningIcon, ax as InfoIcon, ay as SuccessIcon } from "./index.js";
const self = (vars) => {
  const { lineHeight, borderRadius, fontWeightStrong, baseColor, dividerColor, actionColor, textColor1, textColor2, closeColorHover, closeColorPressed, closeIconColor, closeIconColorHover, closeIconColorPressed, infoColor, successColor, warningColor, errorColor, fontSize } = vars;
  return Object.assign(Object.assign({}, commonVars), {
    fontSize,
    lineHeight,
    titleFontWeight: fontWeightStrong,
    borderRadius,
    border: `1px solid ${dividerColor}`,
    color: actionColor,
    titleTextColor: textColor1,
    iconColor: textColor2,
    contentTextColor: textColor2,
    closeBorderRadius: borderRadius,
    closeColorHover,
    closeColorPressed,
    closeIconColor,
    closeIconColorHover,
    closeIconColorPressed,
    borderInfo: `1px solid ${composite(baseColor, changeColor(infoColor, { alpha: 0.25 }))}`,
    colorInfo: composite(baseColor, changeColor(infoColor, { alpha: 0.08 })),
    titleTextColorInfo: textColor1,
    iconColorInfo: infoColor,
    contentTextColorInfo: textColor2,
    closeColorHoverInfo: closeColorHover,
    closeColorPressedInfo: closeColorPressed,
    closeIconColorInfo: closeIconColor,
    closeIconColorHoverInfo: closeIconColorHover,
    closeIconColorPressedInfo: closeIconColorPressed,
    borderSuccess: `1px solid ${composite(baseColor, changeColor(successColor, { alpha: 0.25 }))}`,
    colorSuccess: composite(baseColor, changeColor(successColor, { alpha: 0.08 })),
    titleTextColorSuccess: textColor1,
    iconColorSuccess: successColor,
    contentTextColorSuccess: textColor2,
    closeColorHoverSuccess: closeColorHover,
    closeColorPressedSuccess: closeColorPressed,
    closeIconColorSuccess: closeIconColor,
    closeIconColorHoverSuccess: closeIconColorHover,
    closeIconColorPressedSuccess: closeIconColorPressed,
    borderWarning: `1px solid ${composite(baseColor, changeColor(warningColor, { alpha: 0.33 }))}`,
    colorWarning: composite(baseColor, changeColor(warningColor, { alpha: 0.08 })),
    titleTextColorWarning: textColor1,
    iconColorWarning: warningColor,
    contentTextColorWarning: textColor2,
    closeColorHoverWarning: closeColorHover,
    closeColorPressedWarning: closeColorPressed,
    closeIconColorWarning: closeIconColor,
    closeIconColorHoverWarning: closeIconColorHover,
    closeIconColorPressedWarning: closeIconColorPressed,
    borderError: `1px solid ${composite(baseColor, changeColor(errorColor, { alpha: 0.25 }))}`,
    colorError: composite(baseColor, changeColor(errorColor, { alpha: 0.08 })),
    titleTextColorError: textColor1,
    iconColorError: errorColor,
    contentTextColorError: textColor2,
    closeColorHoverError: closeColorHover,
    closeColorPressedError: closeColorPressed,
    closeIconColorError: closeIconColor,
    closeIconColorHoverError: closeIconColorHover,
    closeIconColorPressedError: closeIconColorPressed
  });
};
const alertLight = {
  name: "Alert",
  common: commonLight,
  self
};
const alertLight$1 = alertLight;
const style = cB("alert", `
 line-height: var(--n-line-height);
 border-radius: var(--n-border-radius);
 position: relative;
 transition: background-color .3s var(--n-bezier);
 background-color: var(--n-color);
 text-align: start;
 word-break: break-word;
`, [cE("border", `
 border-radius: inherit;
 position: absolute;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 transition: border-color .3s var(--n-bezier);
 border: var(--n-border);
 pointer-events: none;
 `), cM("closable", [cB("alert-body", [cE("title", `
 padding-right: 24px;
 `)])]), cE("icon", {
  color: "var(--n-icon-color)"
}), cB("alert-body", {
  padding: "var(--n-padding)"
}, [cE("title", {
  color: "var(--n-title-text-color)"
}), cE("content", {
  color: "var(--n-content-text-color)"
})]), fadeInHeightExpandTransition({
  originalTransition: "transform .3s var(--n-bezier)",
  enterToProps: {
    transform: "scale(1)"
  },
  leaveToProps: {
    transform: "scale(0.9)"
  }
}), cE("icon", `
 position: absolute;
 left: 0;
 top: 0;
 align-items: center;
 justify-content: center;
 display: flex;
 width: var(--n-icon-size);
 height: var(--n-icon-size);
 font-size: var(--n-icon-size);
 margin: var(--n-icon-margin);
 `), cE("close", `
 transition:
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 position: absolute;
 right: 0;
 top: 0;
 margin: var(--n-close-margin);
 `), cM("show-icon", [cB("alert-body", {
  paddingLeft: "calc(var(--n-icon-margin-left) + var(--n-icon-size) + var(--n-icon-margin-right))"
})]), cB("alert-body", `
 border-radius: var(--n-border-radius);
 transition: border-color .3s var(--n-bezier);
 `, [cE("title", `
 transition: color .3s var(--n-bezier);
 font-size: 16px;
 line-height: 19px;
 font-weight: var(--n-title-font-weight);
 `, [c("& +", [cE("content", {
  marginTop: "9px"
})])]), cE("content", {
  transition: "color .3s var(--n-bezier)",
  fontSize: "var(--n-font-size)"
})]), cE("icon", {
  transition: "color .3s var(--n-bezier)"
})]);
const alertProps = Object.assign(Object.assign({}, useTheme.props), {
  title: String,
  showIcon: {
    type: Boolean,
    default: true
  },
  type: {
    type: String,
    default: "default"
  },
  bordered: {
    type: Boolean,
    default: true
  },
  closable: Boolean,
  onClose: Function,
  onAfterLeave: Function,
  /** @deprecated */
  onAfterHide: Function
});
const NAlert = defineComponent({
  name: "Alert",
  inheritAttrs: false,
  props: alertProps,
  setup(props) {
    const { mergedClsPrefixRef, mergedBorderedRef, inlineThemeDisabled, mergedRtlRef } = useConfig(props);
    const themeRef = useTheme("Alert", "-alert", style, alertLight$1, props, mergedClsPrefixRef);
    const rtlEnabledRef = useRtl("Alert", mergedRtlRef, mergedClsPrefixRef);
    const cssVarsRef = computed(() => {
      const { common: { cubicBezierEaseInOut }, self: self2 } = themeRef.value;
      const { fontSize, borderRadius, titleFontWeight, lineHeight, iconSize, iconMargin, iconMarginRtl, closeIconSize, closeBorderRadius, closeSize, closeMargin, closeMarginRtl, padding } = self2;
      const { type } = props;
      const { left, right } = getMargin(iconMargin);
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-color": self2[createKey("color", type)],
        "--n-close-icon-size": closeIconSize,
        "--n-close-border-radius": closeBorderRadius,
        "--n-close-color-hover": self2[createKey("closeColorHover", type)],
        "--n-close-color-pressed": self2[createKey("closeColorPressed", type)],
        "--n-close-icon-color": self2[createKey("closeIconColor", type)],
        "--n-close-icon-color-hover": self2[createKey("closeIconColorHover", type)],
        "--n-close-icon-color-pressed": self2[createKey("closeIconColorPressed", type)],
        "--n-icon-color": self2[createKey("iconColor", type)],
        "--n-border": self2[createKey("border", type)],
        "--n-title-text-color": self2[createKey("titleTextColor", type)],
        "--n-content-text-color": self2[createKey("contentTextColor", type)],
        "--n-line-height": lineHeight,
        "--n-border-radius": borderRadius,
        "--n-font-size": fontSize,
        "--n-title-font-weight": titleFontWeight,
        "--n-icon-size": iconSize,
        "--n-icon-margin": iconMargin,
        "--n-icon-margin-rtl": iconMarginRtl,
        "--n-close-size": closeSize,
        "--n-close-margin": closeMargin,
        "--n-close-margin-rtl": closeMarginRtl,
        "--n-padding": padding,
        "--n-icon-margin-left": left,
        "--n-icon-margin-right": right
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("alert", computed(() => {
      return props.type[0];
    }), cssVarsRef, props) : void 0;
    const visibleRef = ref(true);
    const doAfterLeave = () => {
      const {
        onAfterLeave,
        onAfterHide
        // deprecated
      } = props;
      if (onAfterLeave)
        onAfterLeave();
      if (onAfterHide)
        onAfterHide();
    };
    const handleCloseClick = () => {
      var _a;
      void Promise.resolve((_a = props.onClose) === null || _a === void 0 ? void 0 : _a.call(props)).then((result) => {
        if (result === false)
          return;
        visibleRef.value = false;
      });
    };
    const handleAfterLeave = () => {
      doAfterLeave();
    };
    return {
      rtlEnabled: rtlEnabledRef,
      mergedClsPrefix: mergedClsPrefixRef,
      mergedBordered: mergedBorderedRef,
      visible: visibleRef,
      handleCloseClick,
      handleAfterLeave,
      mergedTheme: themeRef,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    var _a;
    (_a = this.onRender) === null || _a === void 0 ? void 0 : _a.call(this);
    return h(NFadeInExpandTransition, { onAfterLeave: this.handleAfterLeave }, {
      default: () => {
        const { mergedClsPrefix, $slots } = this;
        const attrs = {
          class: [
            `${mergedClsPrefix}-alert`,
            this.themeClass,
            this.closable && `${mergedClsPrefix}-alert--closable`,
            this.showIcon && `${mergedClsPrefix}-alert--show-icon`,
            this.rtlEnabled && `${mergedClsPrefix}-alert--rtl`
          ],
          style: this.cssVars,
          role: "alert"
        };
        return this.visible ? h(
          "div",
          Object.assign({}, mergeProps(this.$attrs, attrs)),
          this.closable && h(NBaseClose, { clsPrefix: mergedClsPrefix, class: `${mergedClsPrefix}-alert__close`, onClick: this.handleCloseClick }),
          this.bordered && h("div", { class: `${mergedClsPrefix}-alert__border` }),
          this.showIcon && h("div", { class: `${mergedClsPrefix}-alert__icon`, "aria-hidden": "true" }, resolveSlot($slots.icon, () => [
            h(NBaseIcon, { clsPrefix: mergedClsPrefix }, {
              default: () => {
                switch (this.type) {
                  case "success":
                    return h(SuccessIcon, null);
                  case "info":
                    return h(InfoIcon, null);
                  case "warning":
                    return h(WarningIcon, null);
                  case "error":
                    return h(ErrorIcon, null);
                  default:
                    return null;
                }
              }
            })
          ])),
          h(
            "div",
            { class: [
              `${mergedClsPrefix}-alert-body`,
              this.mergedBordered && `${mergedClsPrefix}-alert-body--bordered`
            ] },
            resolveWrappedSlot($slots.header, (children) => {
              const mergedChildren = children || this.title;
              return mergedChildren ? h("div", { class: `${mergedClsPrefix}-alert-body__title` }, mergedChildren) : null;
            }),
            $slots.default && h("div", { class: `${mergedClsPrefix}-alert-body__content` }, $slots)
          )
        ) : null;
      }
    });
  }
});
export {
  NAlert as N
};
