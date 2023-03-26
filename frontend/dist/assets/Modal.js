import { ba as isBrowser, bb as readonly, r as ref, ak as on, bc as hasInstance, bd as onBeforeMount, aM as onBeforeUnmount, aN as off, ao as keysOf, i as c, e as cB, f as cE, g as cM, ad as insideModal, be as asModal, j as defineComponent, k as useTheme, u as useConfig, m as computed, bf as dialogLight, w as createKey, n as useThemeClass, o as h, s as resolveWrappedSlot, bg as render, x as NBaseIcon, a2 as NButton, q as resolveSlot, N as NBaseClose, I as InfoIcon, S as SuccessIcon, W as WarningIcon, E as ErrorIcon, aa as createInjectionKey, bh as cardBaseProps, am as watch, a9 as toRef, bi as useLockHtmlScroll, bj as getFirstSlotVNode, aX as warn, bk as cloneVNode, p as mergeProps, bl as withDirectives, bm as vShow, aL as NScrollbar, bn as FocusTrap, b1 as Transition, bo as clickoutside, at as keep, K as NCard, bp as cardBasePropKeys, af as inject, bq as modalInjectionKey, ar as nextTick, a8 as provide, br as modalBodyInjectionKey, bs as drawerBodyInjectionKey, bt as popoverBodyInjectionKey, bu as fadeInTransition, b0 as fadeInScaleUpTransition, bv as isMounted, bw as useIsComposing, bx as zindexable, by as LazyTeleport, bz as modalLight, ab as call, bA as getPreciseEventTarget, bB as eventEffectNotPerformed } from "./index.js";
const mousePositionRef = ref(null);
function clickHandler(e) {
  if (e.clientX > 0 || e.clientY > 0) {
    mousePositionRef.value = {
      x: e.clientX,
      y: e.clientY
    };
  } else {
    const { target } = e;
    if (target instanceof Element) {
      const { left, top, width, height } = target.getBoundingClientRect();
      if (left > 0 || top > 0) {
        mousePositionRef.value = {
          x: left + width / 2,
          y: top + height / 2
        };
      } else {
        mousePositionRef.value = { x: 0, y: 0 };
      }
    } else {
      mousePositionRef.value = null;
    }
  }
}
let usedCount$1 = 0;
let managable$1 = true;
function useClickPosition() {
  if (!isBrowser)
    return readonly(ref(null));
  if (usedCount$1 === 0)
    on("click", document, clickHandler, true);
  const setup = () => {
    usedCount$1 += 1;
  };
  if (managable$1 && (managable$1 = hasInstance())) {
    onBeforeMount(setup);
    onBeforeUnmount(() => {
      usedCount$1 -= 1;
      if (usedCount$1 === 0)
        off("click", document, clickHandler, true);
    });
  } else {
    setup();
  }
  return readonly(mousePositionRef);
}
const clickedTimeRef = ref(void 0);
let usedCount = 0;
function handleClick() {
  clickedTimeRef.value = Date.now();
}
let managable = true;
function useClicked(timeout) {
  if (!isBrowser)
    return readonly(ref(false));
  const clickedRef = ref(false);
  let timerId = null;
  function clearTimer() {
    if (timerId !== null)
      window.clearTimeout(timerId);
  }
  function clickedHandler() {
    clearTimer();
    clickedRef.value = true;
    timerId = window.setTimeout(() => {
      clickedRef.value = false;
    }, timeout);
  }
  if (usedCount === 0) {
    on("click", window, handleClick, true);
  }
  const setup = () => {
    usedCount += 1;
    on("click", window, clickedHandler, true);
  };
  if (managable && (managable = hasInstance())) {
    onBeforeMount(setup);
    onBeforeUnmount(() => {
      usedCount -= 1;
      if (usedCount === 0) {
        off("click", window, handleClick, true);
      }
      off("click", window, clickedHandler, true);
      clearTimer();
    });
  } else {
    setup();
  }
  return readonly(clickedRef);
}
const dialogProps = {
  icon: Function,
  type: {
    type: String,
    default: "default"
  },
  title: [String, Function],
  closable: {
    type: Boolean,
    default: true
  },
  negativeText: String,
  positiveText: String,
  positiveButtonProps: Object,
  negativeButtonProps: Object,
  content: [String, Function],
  action: Function,
  showIcon: {
    type: Boolean,
    default: true
  },
  loading: Boolean,
  bordered: Boolean,
  iconPlacement: String,
  onPositiveClick: Function,
  onNegativeClick: Function,
  onClose: Function
};
const dialogPropKeys = keysOf(dialogProps);
const style$1 = c([cB("dialog", `
 word-break: break-word;
 line-height: var(--n-line-height);
 position: relative;
 background: var(--n-color);
 color: var(--n-text-color);
 box-sizing: border-box;
 margin: auto;
 border-radius: var(--n-border-radius);
 padding: var(--n-padding);
 transition: 
 border-color .3s var(--n-bezier),
 background-color .3s var(--n-bezier),
 color .3s var(--n-bezier);
 `, [cE("icon", {
  color: "var(--n-icon-color)"
}), cM("bordered", {
  border: "var(--n-border)"
}), cM("icon-top", [cE("close", {
  margin: "var(--n-close-margin)"
}), cE("icon", {
  margin: "var(--n-icon-margin)"
}), cE("content", {
  textAlign: "center"
}), cE("title", {
  justifyContent: "center"
}), cE("action", {
  justifyContent: "center"
})]), cM("icon-left", [cE("icon", {
  margin: "var(--n-icon-margin)"
}), cM("closable", [cE("title", `
 padding-right: calc(var(--n-close-size) + 6px);
 `)])]), cE("close", `
 position: absolute;
 right: 0;
 top: 0;
 margin: var(--n-close-margin);
 transition:
 background-color .3s var(--n-bezier),
 color .3s var(--n-bezier);
 z-index: 1;
 `), cE("content", `
 font-size: var(--n-font-size);
 margin: var(--n-content-margin);
 position: relative;
 word-break: break-word;
 `, [cM("last", "margin-bottom: 0;")]), cE("action", `
 display: flex;
 justify-content: flex-end;
 `, [c("> *:not(:last-child)", {
  marginRight: "var(--n-action-space)"
})]), cE("icon", {
  fontSize: "var(--n-icon-size)",
  transition: "color .3s var(--n-bezier)"
}), cE("title", `
 transition: color .3s var(--n-bezier);
 display: flex;
 align-items: center;
 font-size: var(--n-title-font-size);
 font-weight: var(--n-title-font-weight);
 color: var(--n-title-text-color);
 `), cB("dialog-icon-container", {
  display: "flex",
  justifyContent: "center"
})]), insideModal(cB("dialog", `
 width: 446px;
 max-width: calc(100vw - 32px);
 `)), cB("dialog", [asModal(`
 width: 446px;
 max-width: calc(100vw - 32px);
 `)])]);
const iconRenderMap = {
  default: () => h(InfoIcon, null),
  info: () => h(InfoIcon, null),
  success: () => h(SuccessIcon, null),
  warning: () => h(WarningIcon, null),
  error: () => h(ErrorIcon, null)
};
const NDialog = defineComponent({
  name: "Dialog",
  alias: [
    "NimbusConfirmCard",
    "Confirm"
    // deprecated
  ],
  props: Object.assign(Object.assign({}, useTheme.props), dialogProps),
  setup(props) {
    const { mergedComponentPropsRef, mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const mergedIconPlacementRef = computed(() => {
      var _a, _b;
      const { iconPlacement } = props;
      return iconPlacement || ((_b = (_a = mergedComponentPropsRef === null || mergedComponentPropsRef === void 0 ? void 0 : mergedComponentPropsRef.value) === null || _a === void 0 ? void 0 : _a.Dialog) === null || _b === void 0 ? void 0 : _b.iconPlacement) || "left";
    });
    function handlePositiveClick(e) {
      const { onPositiveClick } = props;
      if (onPositiveClick)
        onPositiveClick(e);
    }
    function handleNegativeClick(e) {
      const { onNegativeClick } = props;
      if (onNegativeClick)
        onNegativeClick(e);
    }
    function handleCloseClick() {
      const { onClose } = props;
      if (onClose)
        onClose();
    }
    const themeRef = useTheme("Dialog", "-dialog", style$1, dialogLight, props, mergedClsPrefixRef);
    const cssVarsRef = computed(() => {
      const { type } = props;
      const iconPlacement = mergedIconPlacementRef.value;
      const { common: { cubicBezierEaseInOut }, self: { fontSize, lineHeight, border, titleTextColor, textColor, color, closeBorderRadius, closeColorHover, closeColorPressed, closeIconColor, closeIconColorHover, closeIconColorPressed, closeIconSize, borderRadius, titleFontWeight, titleFontSize, padding, iconSize, actionSpace, contentMargin, closeSize, [iconPlacement === "top" ? "iconMarginIconTop" : "iconMargin"]: iconMargin, [iconPlacement === "top" ? "closeMarginIconTop" : "closeMargin"]: closeMargin, [createKey("iconColor", type)]: iconColor } } = themeRef.value;
      return {
        "--n-font-size": fontSize,
        "--n-icon-color": iconColor,
        "--n-bezier": cubicBezierEaseInOut,
        "--n-close-margin": closeMargin,
        "--n-icon-margin": iconMargin,
        "--n-icon-size": iconSize,
        "--n-close-size": closeSize,
        "--n-close-icon-size": closeIconSize,
        "--n-close-border-radius": closeBorderRadius,
        "--n-close-color-hover": closeColorHover,
        "--n-close-color-pressed": closeColorPressed,
        "--n-close-icon-color": closeIconColor,
        "--n-close-icon-color-hover": closeIconColorHover,
        "--n-close-icon-color-pressed": closeIconColorPressed,
        "--n-color": color,
        "--n-text-color": textColor,
        "--n-border-radius": borderRadius,
        "--n-padding": padding,
        "--n-line-height": lineHeight,
        "--n-border": border,
        "--n-content-margin": contentMargin,
        "--n-title-font-size": titleFontSize,
        "--n-title-font-weight": titleFontWeight,
        "--n-title-text-color": titleTextColor,
        "--n-action-space": actionSpace
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("dialog", computed(() => `${props.type[0]}${mergedIconPlacementRef.value[0]}`), cssVarsRef, props) : void 0;
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      mergedIconPlacement: mergedIconPlacementRef,
      mergedTheme: themeRef,
      handlePositiveClick,
      handleNegativeClick,
      handleCloseClick,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    var _a;
    const { bordered, mergedIconPlacement, cssVars, closable, showIcon, title, content, action, negativeText, positiveText, positiveButtonProps, negativeButtonProps, handlePositiveClick, handleNegativeClick, mergedTheme, loading, type, mergedClsPrefix } = this;
    (_a = this.onRender) === null || _a === void 0 ? void 0 : _a.call(this);
    const icon = showIcon ? h(NBaseIcon, { clsPrefix: mergedClsPrefix, class: `${mergedClsPrefix}-dialog__icon` }, {
      default: () => resolveWrappedSlot(this.$slots.icon, (children) => children || (this.icon ? render(this.icon) : iconRenderMap[this.type]()))
    }) : null;
    const actionNode = resolveWrappedSlot(this.$slots.action, (children) => children || positiveText || negativeText || action ? h("div", { class: `${mergedClsPrefix}-dialog__action` }, children || (action ? [render(action)] : [
      this.negativeText && h(NButton, Object.assign({ theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, ghost: true, size: "small", onClick: handleNegativeClick }, negativeButtonProps), {
        default: () => render(this.negativeText)
      }),
      this.positiveText && h(NButton, Object.assign({ theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, size: "small", type: type === "default" ? "primary" : type, disabled: loading, loading, onClick: handlePositiveClick }, positiveButtonProps), {
        default: () => render(this.positiveText)
      })
    ])) : null);
    return h(
      "div",
      { class: [
        `${mergedClsPrefix}-dialog`,
        this.themeClass,
        this.closable && `${mergedClsPrefix}-dialog--closable`,
        `${mergedClsPrefix}-dialog--icon-${mergedIconPlacement}`,
        bordered && `${mergedClsPrefix}-dialog--bordered`
      ], style: cssVars, role: "dialog" },
      closable ? h(NBaseClose, { clsPrefix: mergedClsPrefix, class: `${mergedClsPrefix}-dialog__close`, onClick: this.handleCloseClick }) : null,
      showIcon && mergedIconPlacement === "top" ? h("div", { class: `${mergedClsPrefix}-dialog-icon-container` }, icon) : null,
      h(
        "div",
        { class: `${mergedClsPrefix}-dialog__title` },
        showIcon && mergedIconPlacement === "left" ? icon : null,
        resolveSlot(this.$slots.header, () => [render(title)])
      ),
      h("div", { class: [
        `${mergedClsPrefix}-dialog__content`,
        actionNode ? "" : `${mergedClsPrefix}-dialog__content--last`
      ] }, resolveSlot(this.$slots.default, () => [render(content)])),
      actionNode
    );
  }
});
const dialogProviderInjectionKey = createInjectionKey("n-dialog-provider");
const presetProps = Object.assign(Object.assign({}, cardBaseProps), dialogProps);
const presetPropsKeys = keysOf(presetProps);
const NModalBodyWrapper = defineComponent({
  name: "ModalBody",
  inheritAttrs: false,
  props: Object.assign(Object.assign({ show: {
    type: Boolean,
    required: true
  }, preset: String, displayDirective: {
    type: String,
    required: true
  }, trapFocus: {
    type: Boolean,
    default: true
  }, autoFocus: {
    type: Boolean,
    default: true
  }, blockScroll: Boolean }, presetProps), {
    renderMask: Function,
    // events
    onClickoutside: Function,
    onBeforeLeave: {
      type: Function,
      required: true
    },
    onAfterLeave: {
      type: Function,
      required: true
    },
    onPositiveClick: {
      type: Function,
      required: true
    },
    onNegativeClick: {
      type: Function,
      required: true
    },
    onClose: {
      type: Function,
      required: true
    },
    onAfterEnter: Function,
    onEsc: Function
  }),
  setup(props) {
    const bodyRef = ref(null);
    const scrollbarRef = ref(null);
    const displayedRef = ref(props.show);
    const transformOriginXRef = ref(null);
    const transformOriginYRef = ref(null);
    watch(toRef(props, "show"), (value) => {
      if (value)
        displayedRef.value = true;
    });
    useLockHtmlScroll(computed(() => props.blockScroll && displayedRef.value));
    const NModal2 = inject(modalInjectionKey);
    function styleTransformOrigin() {
      if (NModal2.transformOriginRef.value === "center") {
        return "";
      }
      const { value: transformOriginX } = transformOriginXRef;
      const { value: transformOriginY } = transformOriginYRef;
      if (transformOriginX === null || transformOriginY === null) {
        return "";
      } else if (scrollbarRef.value) {
        const scrollTop = scrollbarRef.value.containerScrollTop;
        return `${transformOriginX}px ${transformOriginY + scrollTop}px`;
      }
      return "";
    }
    function syncTransformOrigin(el) {
      if (NModal2.transformOriginRef.value === "center") {
        return;
      }
      const mousePosition = NModal2.getMousePosition();
      if (!mousePosition) {
        return;
      }
      if (!scrollbarRef.value)
        return;
      const scrollTop = scrollbarRef.value.containerScrollTop;
      const { offsetLeft, offsetTop } = el;
      if (mousePosition) {
        const top = mousePosition.y;
        const left = mousePosition.x;
        transformOriginXRef.value = -(offsetLeft - left);
        transformOriginYRef.value = -(offsetTop - top - scrollTop);
      }
      el.style.transformOrigin = styleTransformOrigin();
    }
    function handleEnter(el) {
      void nextTick(() => {
        syncTransformOrigin(el);
      });
    }
    function handleBeforeLeave(el) {
      el.style.transformOrigin = styleTransformOrigin();
      props.onBeforeLeave();
    }
    function handleAfterLeave() {
      displayedRef.value = false;
      transformOriginXRef.value = null;
      transformOriginYRef.value = null;
      props.onAfterLeave();
    }
    function handleCloseClick() {
      const { onClose } = props;
      if (onClose) {
        onClose();
      }
    }
    function handleNegativeClick() {
      props.onNegativeClick();
    }
    function handlePositiveClick() {
      props.onPositiveClick();
    }
    const childNodeRef = ref(null);
    watch(childNodeRef, (node) => {
      if (node) {
        void nextTick(() => {
          const el = node.el;
          if (el && bodyRef.value !== el) {
            bodyRef.value = el;
          }
        });
      }
    });
    provide(modalBodyInjectionKey, bodyRef);
    provide(drawerBodyInjectionKey, null);
    provide(popoverBodyInjectionKey, null);
    return {
      mergedTheme: NModal2.mergedThemeRef,
      appear: NModal2.appearRef,
      isMounted: NModal2.isMountedRef,
      mergedClsPrefix: NModal2.mergedClsPrefixRef,
      bodyRef,
      scrollbarRef,
      displayed: displayedRef,
      childNodeRef,
      handlePositiveClick,
      handleNegativeClick,
      handleCloseClick,
      handleAfterLeave,
      handleBeforeLeave,
      handleEnter
    };
  },
  render() {
    const { $slots, $attrs, handleEnter, handleAfterLeave, handleBeforeLeave, preset, mergedClsPrefix } = this;
    let childNode = null;
    if (!preset) {
      childNode = getFirstSlotVNode($slots);
      if (!childNode) {
        warn("modal", "default slot is empty");
        return;
      }
      childNode = cloneVNode(childNode);
      childNode.props = mergeProps({
        class: `${mergedClsPrefix}-modal`
      }, $attrs, childNode.props || {});
    }
    return this.displayDirective === "show" || this.displayed || this.show ? withDirectives(h(
      "div",
      { role: "none", class: `${mergedClsPrefix}-modal-body-wrapper` },
      h(NScrollbar, { ref: "scrollbarRef", theme: this.mergedTheme.peers.Scrollbar, themeOverrides: this.mergedTheme.peerOverrides.Scrollbar, contentClass: `${mergedClsPrefix}-modal-scroll-content` }, {
        default: () => {
          var _a;
          return [
            (_a = this.renderMask) === null || _a === void 0 ? void 0 : _a.call(this),
            h(FocusTrap, { disabled: !this.trapFocus, active: this.show, onEsc: this.onEsc, autoFocus: this.autoFocus }, {
              default: () => {
                var _a2;
                return h(Transition, { name: "fade-in-scale-up-transition", appear: (_a2 = this.appear) !== null && _a2 !== void 0 ? _a2 : this.isMounted, onEnter: handleEnter, onAfterEnter: this.onAfterEnter, onAfterLeave: handleAfterLeave, onBeforeLeave: handleBeforeLeave }, {
                  default: () => {
                    const dirs = [
                      [vShow, this.show]
                    ];
                    const { onClickoutside } = this;
                    if (onClickoutside) {
                      dirs.push([
                        clickoutside,
                        this.onClickoutside,
                        void 0,
                        { capture: true }
                      ]);
                    }
                    return withDirectives(this.preset === "confirm" || this.preset === "dialog" ? h(NDialog, Object.assign({}, this.$attrs, { class: [
                      `${mergedClsPrefix}-modal`,
                      this.$attrs.class
                    ], ref: "bodyRef", theme: this.mergedTheme.peers.Dialog, themeOverrides: this.mergedTheme.peerOverrides.Dialog }, keep(this.$props, dialogPropKeys), { "aria-modal": "true" }), $slots) : this.preset === "card" ? h(NCard, Object.assign({}, this.$attrs, { ref: "bodyRef", class: [
                      `${mergedClsPrefix}-modal`,
                      this.$attrs.class
                    ], theme: this.mergedTheme.peers.Card, themeOverrides: this.mergedTheme.peerOverrides.Card }, keep(this.$props, cardBasePropKeys), { "aria-modal": "true", role: "dialog" }), $slots) : this.childNodeRef = childNode, dirs);
                  }
                });
              }
            })
          ];
        }
      })
    ), [
      [
        vShow,
        this.displayDirective === "if" || this.displayed || this.show
      ]
    ]) : null;
  }
});
const style = c([cB("modal-container", `
 position: fixed;
 left: 0;
 top: 0;
 height: 0;
 width: 0;
 display: flex;
 `), cB("modal-mask", `
 position: fixed;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 background-color: rgba(0, 0, 0, .4);
 `, [fadeInTransition({
  enterDuration: ".25s",
  leaveDuration: ".25s",
  enterCubicBezier: "var(--n-bezier-ease-out)",
  leaveCubicBezier: "var(--n-bezier-ease-out)"
})]), cB("modal-body-wrapper", `
 position: fixed;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 overflow: visible;
 `, [cB("modal-scroll-content", `
 min-height: 100%;
 display: flex;
 position: relative;
 `)]), cB("modal", `
 position: relative;
 align-self: center;
 color: var(--n-text-color);
 margin: auto;
 box-shadow: var(--n-box-shadow);
 `, [fadeInScaleUpTransition({
  duration: ".25s",
  enterScale: ".5"
})])]);
const modalProps = Object.assign(Object.assign(Object.assign(Object.assign({}, useTheme.props), { show: Boolean, unstableShowMask: {
  type: Boolean,
  default: true
}, maskClosable: {
  type: Boolean,
  default: true
}, preset: String, to: [String, Object], displayDirective: {
  type: String,
  default: "if"
}, transformOrigin: {
  type: String,
  default: "mouse"
}, zIndex: Number, autoFocus: {
  type: Boolean,
  default: true
}, trapFocus: {
  type: Boolean,
  default: true
}, closeOnEsc: {
  type: Boolean,
  default: true
}, blockScroll: { type: Boolean, default: true } }), presetProps), {
  // events
  onEsc: Function,
  "onUpdate:show": [Function, Array],
  onUpdateShow: [Function, Array],
  onAfterEnter: Function,
  onBeforeLeave: Function,
  onAfterLeave: Function,
  onClose: Function,
  onPositiveClick: Function,
  onNegativeClick: Function,
  onMaskClick: Function,
  // private
  internalDialog: Boolean,
  internalAppear: {
    type: Boolean,
    default: void 0
  },
  // deprecated
  overlayStyle: [String, Object],
  onBeforeHide: Function,
  onAfterHide: Function,
  onHide: Function
});
const NModal = defineComponent({
  name: "Modal",
  inheritAttrs: false,
  props: modalProps,
  setup(props) {
    const containerRef = ref(null);
    const { mergedClsPrefixRef, namespaceRef, inlineThemeDisabled } = useConfig(props);
    const themeRef = useTheme("Modal", "-modal", style, modalLight, props, mergedClsPrefixRef);
    const clickedRef = useClicked(64);
    const clickedPositionRef = useClickPosition();
    const isMountedRef = isMounted();
    const NDialogProvider = props.internalDialog ? inject(dialogProviderInjectionKey, null) : null;
    const isComposingRef = useIsComposing();
    function doUpdateShow(show) {
      const { onUpdateShow, "onUpdate:show": _onUpdateShow, onHide } = props;
      if (onUpdateShow)
        call(onUpdateShow, show);
      if (_onUpdateShow)
        call(_onUpdateShow, show);
      if (onHide && !show)
        onHide(show);
    }
    function handleCloseClick() {
      const { onClose } = props;
      if (onClose) {
        void Promise.resolve(onClose()).then((value) => {
          if (value === false)
            return;
          doUpdateShow(false);
        });
      } else {
        doUpdateShow(false);
      }
    }
    function handlePositiveClick() {
      const { onPositiveClick } = props;
      if (onPositiveClick) {
        void Promise.resolve(onPositiveClick()).then((value) => {
          if (value === false)
            return;
          doUpdateShow(false);
        });
      } else {
        doUpdateShow(false);
      }
    }
    function handleNegativeClick() {
      const { onNegativeClick } = props;
      if (onNegativeClick) {
        void Promise.resolve(onNegativeClick()).then((value) => {
          if (value === false)
            return;
          doUpdateShow(false);
        });
      } else {
        doUpdateShow(false);
      }
    }
    function handleBeforeLeave() {
      const { onBeforeLeave, onBeforeHide } = props;
      if (onBeforeLeave)
        call(onBeforeLeave);
      if (onBeforeHide)
        onBeforeHide();
    }
    function handleAfterLeave() {
      const { onAfterLeave, onAfterHide } = props;
      if (onAfterLeave)
        call(onAfterLeave);
      if (onAfterHide)
        onAfterHide();
    }
    function handleClickoutside(e) {
      var _a;
      const { onMaskClick } = props;
      if (onMaskClick) {
        onMaskClick(e);
      }
      if (props.maskClosable) {
        if ((_a = containerRef.value) === null || _a === void 0 ? void 0 : _a.contains(getPreciseEventTarget(e))) {
          doUpdateShow(false);
        }
      }
    }
    function handleEsc(e) {
      var _a;
      (_a = props.onEsc) === null || _a === void 0 ? void 0 : _a.call(props);
      if (props.show && props.closeOnEsc && eventEffectNotPerformed(e)) {
        !isComposingRef.value && doUpdateShow(false);
      }
    }
    provide(modalInjectionKey, {
      getMousePosition: () => {
        if (NDialogProvider) {
          const { clickedRef: clickedRef2, clickPositionRef } = NDialogProvider;
          if (clickedRef2.value && clickPositionRef.value) {
            return clickPositionRef.value;
          }
        }
        if (clickedRef.value) {
          return clickedPositionRef.value;
        }
        return null;
      },
      mergedClsPrefixRef,
      mergedThemeRef: themeRef,
      isMountedRef,
      appearRef: toRef(props, "internalAppear"),
      transformOriginRef: toRef(props, "transformOrigin")
    });
    const cssVarsRef = computed(() => {
      const { common: { cubicBezierEaseOut }, self: { boxShadow, color, textColor } } = themeRef.value;
      return {
        "--n-bezier-ease-out": cubicBezierEaseOut,
        "--n-box-shadow": boxShadow,
        "--n-color": color,
        "--n-text-color": textColor
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("theme-class", void 0, cssVarsRef, props) : void 0;
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      namespace: namespaceRef,
      isMounted: isMountedRef,
      containerRef,
      presetProps: computed(() => {
        const pickedProps = keep(props, presetPropsKeys);
        return pickedProps;
      }),
      handleEsc,
      handleAfterLeave,
      handleClickoutside,
      handleBeforeLeave,
      doUpdateShow,
      handleNegativeClick,
      handlePositiveClick,
      handleCloseClick,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    const { mergedClsPrefix } = this;
    return h(LazyTeleport, { to: this.to, show: this.show }, {
      default: () => {
        var _a;
        (_a = this.onRender) === null || _a === void 0 ? void 0 : _a.call(this);
        const { unstableShowMask } = this;
        return withDirectives(h(
          "div",
          { role: "none", ref: "containerRef", class: [
            `${mergedClsPrefix}-modal-container`,
            this.themeClass,
            this.namespace
          ], style: this.cssVars },
          h(NModalBodyWrapper, Object.assign({ style: this.overlayStyle }, this.$attrs, { ref: "bodyWrapper", displayDirective: this.displayDirective, show: this.show, preset: this.preset, autoFocus: this.autoFocus, trapFocus: this.trapFocus, blockScroll: this.blockScroll }, this.presetProps, { onEsc: this.handleEsc, onClose: this.handleCloseClick, onNegativeClick: this.handleNegativeClick, onPositiveClick: this.handlePositiveClick, onBeforeLeave: this.handleBeforeLeave, onAfterEnter: this.onAfterEnter, onAfterLeave: this.handleAfterLeave, onClickoutside: unstableShowMask ? void 0 : this.handleClickoutside, renderMask: unstableShowMask ? () => {
            var _a2;
            return h(Transition, { name: "fade-in-transition", key: "mask", appear: (_a2 = this.internalAppear) !== null && _a2 !== void 0 ? _a2 : this.isMounted }, {
              default: () => {
                return this.show ? h("div", { "aria-hidden": true, ref: "containerRef", class: `${mergedClsPrefix}-modal-mask`, onClick: this.handleClickoutside }) : null;
              }
            });
          } : void 0 }), this.$slots)
        ), [
          [
            zindexable,
            {
              zIndex: this.zIndex,
              enabled: this.show
            }
          ]
        ]);
      }
    });
  }
});
export {
  NModal as N
};
