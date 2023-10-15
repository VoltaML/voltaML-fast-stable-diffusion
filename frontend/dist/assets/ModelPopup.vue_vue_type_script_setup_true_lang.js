import { bk as upperFirst, bl as toString, bm as createCompounder, bn as cloneVNode, a6 as provide, S as createInjectionKey, U as inject, a$ as throwError, d as defineComponent, V as useConfig, r as ref, bo as onBeforeUpdate, s as h, bp as indexMap, c as computed, ba as onMounted, aC as onBeforeUnmount, T as cB, au as cE, ad as c, ae as cM, ah as useMergedState, Y as toRef, ai as watchEffect, bq as onUpdated, O as watch, W as useTheme, Z as useThemeClass, ax as flatten, aM as VResizeObserver, br as resolveSlotWithProps, bs as withDirectives, bt as vShow, aY as Transition, a7 as keep, aD as off, a3 as nextTick, bu as carouselLight, bb as normalizeStyle, bv as getPreciseEventTarget, aE as on, af as cNotM, as as useFormItem, g as renderList, al as NBaseIcon, bw as rateLight, ao as createKey, bx as color2Class, a2 as call, x as useMessage, u as useSettings, b9 as reactive, o as openBlock, k as createBlock, w as withCtx, f as unref, e as createVNode, a as createElementBlock, L as NTabPane, H as NGrid, A as NGi, F as Fragment, b as createBaseVNode, N as NCard, i as createTextVNode, t as toDisplayString, by as NTag, p as NSelect, h as NButton, M as NTabs, m as NModal, z as serverUrl } from "./index.js";
import { a as NDescriptions, N as NDescriptionsItem } from "./DescriptionsItem.js";
function capitalize(string) {
  return upperFirst(toString(string).toLowerCase());
}
var camelCase = createCompounder(function(result, word, index) {
  word = word.toLowerCase();
  return result + (index ? capitalize(word) : word);
});
const camelCase$1 = camelCase;
function addDuplicateSlides(slides) {
  const { length } = slides;
  if (length > 1) {
    slides.push(duplicateSlide(slides[0], 0, "append"));
    slides.unshift(duplicateSlide(slides[length - 1], length - 1, "prepend"));
    return slides;
  }
  return slides;
}
function duplicateSlide(child, index, position) {
  return cloneVNode(child, {
    // for patch
    key: `carousel-item-duplicate-${index}-${position}`
  });
}
function getDisplayIndex(current, length, duplicatedable) {
  return !duplicatedable ? current : current === 0 ? length - 3 : current === length - 1 ? 0 : current - 1;
}
function getRealIndex(current, duplicatedable) {
  return !duplicatedable ? current : current + 1;
}
function getPrevIndex(current, length, duplicatedable) {
  if (current < 0)
    return null;
  return current === 0 ? duplicatedable ? length - 1 : null : current - 1;
}
function getNextIndex(current, length, duplicatedable) {
  if (current > length - 1)
    return null;
  return current === length - 1 ? duplicatedable ? 0 : null : current + 1;
}
function getDisplayTotalView(total, duplicatedable) {
  return duplicatedable && total > 3 ? total - 2 : total;
}
function isTouchEvent(e) {
  return window.TouchEvent && e instanceof window.TouchEvent;
}
function calculateSize(element, innerOnly) {
  let { offsetWidth: width, offsetHeight: height } = element;
  if (innerOnly) {
    const style2 = getComputedStyle(element);
    width = width - parseFloat(style2.getPropertyValue("padding-left")) - parseFloat(style2.getPropertyValue("padding-right"));
    height = height - parseFloat(style2.getPropertyValue("padding-top")) - parseFloat(style2.getPropertyValue("padding-bottom"));
  }
  return { width, height };
}
function clampValue(value, min, max) {
  return value < min ? min : value > max ? max : value;
}
function resolveSpeed(value) {
  if (value === void 0)
    return 0;
  if (typeof value === "number")
    return value;
  const timeRE = /^((\d+)?\.?\d+?)(ms|s)?$/;
  const match = value.match(timeRE);
  if (match) {
    const [, number, , unit = "ms"] = match;
    return Number(number) * (unit === "ms" ? 1 : 1e3);
  }
  return 0;
}
const carouselMethodsInjectionKey = createInjectionKey("n-carousel-methods");
const provideCarouselContext = (contextValue) => {
  provide(carouselMethodsInjectionKey, contextValue);
};
const useCarouselContext = (location = "unknown", component = "component") => {
  const CarouselContext = inject(carouselMethodsInjectionKey);
  if (!CarouselContext) {
    throwError(location, `\`${component}\` must be placed inside \`n-carousel\`.`);
  }
  return CarouselContext;
};
const carouselDotsProps = {
  total: {
    type: Number,
    default: 0
  },
  currentIndex: {
    type: Number,
    default: 0
  },
  dotType: {
    type: String,
    default: "dot"
  },
  trigger: {
    type: String,
    default: "click"
  },
  keyboard: Boolean
};
const NCarouselDots = defineComponent({
  name: "CarouselDots",
  props: carouselDotsProps,
  setup(props) {
    const { mergedClsPrefixRef } = useConfig(props);
    const dotElsRef = ref([]);
    const NCarousel2 = useCarouselContext();
    function handleKeydown(e, current) {
      switch (e.key) {
        case "Enter":
        case " ":
          e.preventDefault();
          NCarousel2.to(current);
          return;
      }
      if (props.keyboard) {
        handleKeyboard(e);
      }
    }
    function handleMouseenter(current) {
      if (props.trigger === "hover") {
        NCarousel2.to(current);
      }
    }
    function handleClick(current) {
      if (props.trigger === "click") {
        NCarousel2.to(current);
      }
    }
    function handleKeyboard(e) {
      var _a;
      if (e.shiftKey || e.altKey || e.ctrlKey || e.metaKey) {
        return;
      }
      const nodeName = (_a = document.activeElement) === null || _a === void 0 ? void 0 : _a.nodeName.toLowerCase();
      if (nodeName === "input" || nodeName === "textarea") {
        return;
      }
      const { code: keycode } = e;
      const isVerticalNext = keycode === "PageUp" || keycode === "ArrowUp";
      const isVerticalPrev = keycode === "PageDown" || keycode === "ArrowDown";
      const isHorizontalNext = keycode === "PageUp" || keycode === "ArrowRight";
      const isHorizontalPrev = keycode === "PageDown" || keycode === "ArrowLeft";
      const vertical = NCarousel2.isVertical();
      const wantToNext = vertical ? isVerticalNext : isHorizontalNext;
      const wantToPrev = vertical ? isVerticalPrev : isHorizontalPrev;
      if (!wantToNext && !wantToPrev) {
        return;
      }
      e.preventDefault();
      if (wantToNext && !NCarousel2.isNextDisabled()) {
        NCarousel2.next();
        focusDot(NCarousel2.currentIndexRef.value);
      } else if (wantToPrev && !NCarousel2.isPrevDisabled()) {
        NCarousel2.prev();
        focusDot(NCarousel2.currentIndexRef.value);
      }
    }
    function focusDot(index) {
      var _a;
      (_a = dotElsRef.value[index]) === null || _a === void 0 ? void 0 : _a.focus();
    }
    onBeforeUpdate(() => dotElsRef.value.length = 0);
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      dotEls: dotElsRef,
      handleKeydown,
      handleMouseenter,
      handleClick
    };
  },
  render() {
    const { mergedClsPrefix, dotEls } = this;
    return h("div", { class: [
      `${mergedClsPrefix}-carousel__dots`,
      `${mergedClsPrefix}-carousel__dots--${this.dotType}`
    ], role: "tablist" }, indexMap(this.total, (i) => {
      const selected = i === this.currentIndex;
      return h("div", { "aria-selected": selected, ref: (el) => dotEls.push(el), role: "button", tabindex: "0", class: [
        `${mergedClsPrefix}-carousel__dot`,
        selected && `${mergedClsPrefix}-carousel__dot--active`
      ], key: i, onClick: () => {
        this.handleClick(i);
      }, onMouseenter: () => {
        this.handleMouseenter(i);
      }, onKeydown: (e) => {
        this.handleKeydown(e, i);
      } });
    }));
  }
});
const backwardIcon = h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 16 16" },
  h(
    "g",
    { fill: "none" },
    h("path", { d: "M10.26 3.2a.75.75 0 0 1 .04 1.06L6.773 8l3.527 3.74a.75.75 0 1 1-1.1 1.02l-4-4.25a.75.75 0 0 1 0-1.02l4-4.25a.75.75 0 0 1 1.06-.04z", fill: "currentColor" })
  )
);
const forwardIcon = h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 16 16" },
  h(
    "g",
    { fill: "none" },
    h("path", { d: "M5.74 3.2a.75.75 0 0 0-.04 1.06L9.227 8L5.7 11.74a.75.75 0 1 0 1.1 1.02l4-4.25a.75.75 0 0 0 0-1.02l-4-4.25a.75.75 0 0 0-1.06-.04z", fill: "currentColor" })
  )
);
const NCarouselArrow = defineComponent({
  name: "CarouselArrow",
  setup(props) {
    const { mergedClsPrefixRef } = useConfig(props);
    const { isVertical, isPrevDisabled, isNextDisabled, prev, next } = useCarouselContext();
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      isVertical,
      isPrevDisabled,
      isNextDisabled,
      prev,
      next
    };
  },
  render() {
    const { mergedClsPrefix } = this;
    return h(
      "div",
      { class: `${mergedClsPrefix}-carousel__arrow-group` },
      h("div", { class: [
        `${mergedClsPrefix}-carousel__arrow`,
        this.isPrevDisabled() && `${mergedClsPrefix}-carousel__arrow--disabled`
      ], role: "button", onClick: this.prev }, backwardIcon),
      h("div", { class: [
        `${mergedClsPrefix}-carousel__arrow`,
        this.isNextDisabled() && `${mergedClsPrefix}-carousel__arrow--disabled`
      ], role: "button", onClick: this.next }, forwardIcon)
    );
  }
});
const CarouselItemName = "CarouselItem";
const isCarouselItem = (child) => {
  var _a;
  return ((_a = child.type) === null || _a === void 0 ? void 0 : _a.name) === CarouselItemName;
};
const NCarouselItem = defineComponent({
  name: CarouselItemName,
  setup(props) {
    const { mergedClsPrefixRef } = useConfig(props);
    const NCarousel2 = useCarouselContext(camelCase$1(CarouselItemName), `n-${camelCase$1(CarouselItemName)}`);
    const selfElRef = ref();
    const indexRef = computed(() => {
      const { value: selfEl } = selfElRef;
      return selfEl ? NCarousel2.getSlideIndex(selfEl) : -1;
    });
    const isPrevRef = computed(() => NCarousel2.isPrev(indexRef.value));
    const isNextRef = computed(() => NCarousel2.isNext(indexRef.value));
    const isActiveRef = computed(() => NCarousel2.isActive(indexRef.value));
    const styleRef = computed(() => NCarousel2.getSlideStyle(indexRef.value));
    onMounted(() => {
      NCarousel2.addSlide(selfElRef.value);
    });
    onBeforeUnmount(() => {
      NCarousel2.removeSlide(selfElRef.value);
    });
    function handleClick(event) {
      const { value: index } = indexRef;
      if (index !== void 0) {
        NCarousel2 === null || NCarousel2 === void 0 ? void 0 : NCarousel2.onCarouselItemClick(index, event);
      }
    }
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      selfElRef,
      isPrev: isPrevRef,
      isNext: isNextRef,
      isActive: isActiveRef,
      index: indexRef,
      style: styleRef,
      handleClick
    };
  },
  render() {
    var _a;
    const { $slots: slots, mergedClsPrefix, isPrev, isNext, isActive, index, style: style2 } = this;
    const className = [
      `${mergedClsPrefix}-carousel__slide`,
      {
        [`${mergedClsPrefix}-carousel__slide--current`]: isActive,
        [`${mergedClsPrefix}-carousel__slide--prev`]: isPrev,
        [`${mergedClsPrefix}-carousel__slide--next`]: isNext
      }
    ];
    return h("div", {
      ref: "selfElRef",
      class: className,
      role: "option",
      tabindex: "-1",
      "data-index": index,
      "aria-hidden": !isActive,
      style: style2,
      // We use ts-ignore for vue-tsc, since it seems to patch native event
      // for vue components
      // @ts-expect-error vue's tsx has type for capture events
      onClickCapture: this.handleClick
    }, (_a = slots.default) === null || _a === void 0 ? void 0 : _a.call(slots, {
      isPrev,
      isNext,
      isActive,
      index
    }));
  }
});
const style$1 = cB("carousel", `
 position: relative;
 width: 100%;
 height: 100%;
 touch-action: pan-y;
 overflow: hidden;
`, [cE("slides", `
 display: flex;
 width: 100%;
 height: 100%;
 transition-timing-function: var(--n-bezier);
 transition-property: transform;
 `, [cE("slide", `
 flex-shrink: 0;
 position: relative;
 width: 100%;
 height: 100%;
 outline: none;
 overflow: hidden;
 `, [c("> img", `
 display: block;
 `)])]), cE("dots", `
 position: absolute;
 display: flex;
 flex-wrap: nowrap;
 `, [cM("dot", [cE("dot", `
 height: var(--n-dot-size);
 width: var(--n-dot-size);
 background-color: var(--n-dot-color);
 border-radius: 50%;
 cursor: pointer;
 transition:
 box-shadow .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 outline: none;
 `, [c("&:focus", `
 background-color: var(--n-dot-color-focus);
 `), cM("active", `
 background-color: var(--n-dot-color-active);
 `)])]), cM("line", [cE("dot", `
 border-radius: 9999px;
 width: var(--n-dot-line-width);
 height: 4px;
 background-color: var(--n-dot-color);
 cursor: pointer;
 transition:
 width .3s var(--n-bezier),
 box-shadow .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 outline: none;
 `, [c("&:focus", `
 background-color: var(--n-dot-color-focus);
 `), cM("active", `
 width: var(--n-dot-line-width-active);
 background-color: var(--n-dot-color-active);
 `)])])]), cE("arrow", `
 transition: background-color .3s var(--n-bezier);
 cursor: pointer;
 height: 28px;
 width: 28px;
 display: flex;
 align-items: center;
 justify-content: center;
 background-color: rgba(255, 255, 255, .2);
 color: var(--n-arrow-color);
 border-radius: 8px;
 user-select: none;
 -webkit-user-select: none;
 font-size: 18px;
 `, [c("svg", `
 height: 1em;
 width: 1em;
 `), c("&:hover", `
 background-color: rgba(255, 255, 255, .3);
 `)]), cM("vertical", `
 touch-action: pan-x;
 `, [cE("slides", `
 flex-direction: column;
 `), cM("fade", [cE("slide", `
 top: 50%;
 left: unset;
 transform: translateY(-50%);
 `)]), cM("card", [cE("slide", `
 top: 50%;
 left: unset;
 transform: translateY(-50%) translateZ(-400px);
 `, [cM("current", `
 transform: translateY(-50%) translateZ(0);
 `), cM("prev", `
 transform: translateY(-100%) translateZ(-200px);
 `), cM("next", `
 transform: translateY(0%) translateZ(-200px);
 `)])])]), cM("usercontrol", [cE("slides", [c(">", [c("div", `
 position: absolute;
 top: 50%;
 left: 50%;
 width: 100%;
 height: 100%;
 transform: translate(-50%, -50%);
 `)])])]), cM("left", [cE("dots", `
 transform: translateY(-50%);
 top: 50%;
 left: 12px;
 flex-direction: column;
 `, [cM("line", [cE("dot", `
 width: 4px;
 height: var(--n-dot-line-width);
 margin: 4px 0;
 transition:
 height .3s var(--n-bezier),
 box-shadow .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 outline: none;
 `, [cM("active", `
 height: var(--n-dot-line-width-active);
 `)])])]), cE("dot", `
 margin: 4px 0;
 `)]), cE("arrow-group", `
 position: absolute;
 display: flex;
 flex-wrap: nowrap;
 `), cM("vertical", [cE("arrow", `
 transform: rotate(90deg);
 `)]), cM("show-arrow", [cM("bottom", [cE("dots", `
 transform: translateX(0);
 bottom: 18px;
 left: 18px;
 `)]), cM("top", [cE("dots", `
 transform: translateX(0);
 top: 18px;
 left: 18px;
 `)]), cM("left", [cE("dots", `
 transform: translateX(0);
 top: 18px;
 left: 18px;
 `)]), cM("right", [cE("dots", `
 transform: translateX(0);
 top: 18px;
 right: 18px;
 `)])]), cM("left", [cE("arrow-group", `
 bottom: 12px;
 left: 12px;
 flex-direction: column;
 `, [c("> *:first-child", `
 margin-bottom: 12px;
 `)])]), cM("right", [cE("dots", `
 transform: translateY(-50%);
 top: 50%;
 right: 12px;
 flex-direction: column;
 `, [cM("line", [cE("dot", `
 width: 4px;
 height: var(--n-dot-line-width);
 margin: 4px 0;
 transition:
 height .3s var(--n-bezier),
 box-shadow .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 outline: none;
 `, [cM("active", `
 height: var(--n-dot-line-width-active);
 `)])])]), cE("dot", `
 margin: 4px 0;
 `), cE("arrow-group", `
 bottom: 12px;
 right: 12px;
 flex-direction: column;
 `, [c("> *:first-child", `
 margin-bottom: 12px;
 `)])]), cM("top", [cE("dots", `
 transform: translateX(-50%);
 top: 12px;
 left: 50%;
 `, [cM("line", [cE("dot", `
 margin: 0 4px;
 `)])]), cE("dot", `
 margin: 0 4px;
 `), cE("arrow-group", `
 top: 12px;
 right: 12px;
 `, [c("> *:first-child", `
 margin-right: 12px;
 `)])]), cM("bottom", [cE("dots", `
 transform: translateX(-50%);
 bottom: 12px;
 left: 50%;
 `, [cM("line", [cE("dot", `
 margin: 0 4px;
 `)])]), cE("dot", `
 margin: 0 4px;
 `), cE("arrow-group", `
 bottom: 12px;
 right: 12px;
 `, [c("> *:first-child", `
 margin-right: 12px;
 `)])]), cM("fade", [cE("slide", `
 position: absolute;
 opacity: 0;
 transition-property: opacity;
 pointer-events: none;
 `, [cM("current", `
 opacity: 1;
 pointer-events: auto;
 `)])]), cM("card", [cE("slides", `
 perspective: 1000px;
 `), cE("slide", `
 position: absolute;
 left: 50%;
 opacity: 0;
 transform: translateX(-50%) translateZ(-400px);
 transition-property: opacity, transform;
 `, [cM("current", `
 opacity: 1;
 transform: translateX(-50%) translateZ(0);
 z-index: 1;
 `), cM("prev", `
 opacity: 0.4;
 transform: translateX(-100%) translateZ(-200px);
 `), cM("next", `
 opacity: 0.4;
 transform: translateX(0%) translateZ(-200px);
 `)])])]);
const transitionProperties = [
  "transitionDuration",
  "transitionTimingFunction"
];
const carouselProps = Object.assign(Object.assign({}, useTheme.props), { defaultIndex: {
  type: Number,
  default: 0
}, currentIndex: Number, showArrow: Boolean, dotType: {
  type: String,
  default: "dot"
}, dotPlacement: {
  type: String,
  default: "bottom"
}, slidesPerView: {
  type: [Number, String],
  default: 1
}, spaceBetween: {
  type: Number,
  default: 0
}, centeredSlides: Boolean, direction: {
  type: String,
  default: "horizontal"
}, autoplay: Boolean, interval: {
  type: Number,
  default: 5e3
}, loop: {
  type: Boolean,
  default: true
}, effect: {
  type: String,
  default: "slide"
}, showDots: {
  type: Boolean,
  default: true
}, trigger: {
  type: String,
  default: "click"
}, transitionStyle: {
  type: Object,
  default: () => ({
    transitionDuration: "300ms"
  })
}, transitionProps: Object, draggable: Boolean, prevSlideStyle: [Object, String], nextSlideStyle: [Object, String], touchable: {
  type: Boolean,
  default: true
}, mousewheel: Boolean, keyboard: Boolean, "onUpdate:currentIndex": Function, onUpdateCurrentIndex: Function });
let globalDragging = false;
const NCarousel = defineComponent({
  name: "Carousel",
  props: carouselProps,
  setup(props) {
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const selfElRef = ref(null);
    const slidesElRef = ref(null);
    const slideElsRef = ref([]);
    const slideVNodesRef = { value: [] };
    const verticalRef = computed(() => props.direction === "vertical");
    const sizeAxisRef = computed(() => verticalRef.value ? "height" : "width");
    const spaceAxisRef = computed(() => verticalRef.value ? "bottom" : "right");
    const sequenceLayoutRef = computed(() => props.effect === "slide");
    const duplicatedableRef = computed(
      // duplicate the copy operation in `slide` mode,
      // because only its DOM is sequence layout
      () => props.loop && props.slidesPerView === 1 && sequenceLayoutRef.value
    );
    const userWantsControlRef = computed(() => props.effect === "custom");
    const displaySlidesPerViewRef = computed(() => !sequenceLayoutRef.value || props.centeredSlides ? 1 : props.slidesPerView);
    const realSlidesPerViewRef = computed(() => userWantsControlRef.value ? 1 : props.slidesPerView);
    const autoSlideSizeRef = computed(() => displaySlidesPerViewRef.value === "auto" || props.slidesPerView === "auto" && props.centeredSlides);
    const perViewSizeRef = ref({ width: 0, height: 0 });
    const slideSizesRef = computed(() => {
      const { value: slidesEls } = slideElsRef;
      if (!slidesEls.length)
        return [];
      const { value: autoSlideSize } = autoSlideSizeRef;
      if (autoSlideSize) {
        return slidesEls.map((slide) => calculateSize(slide));
      }
      const { value: slidesPerView } = realSlidesPerViewRef;
      const { value: perViewSize } = perViewSizeRef;
      const { value: axis } = sizeAxisRef;
      let axisSize = perViewSize[axis];
      if (slidesPerView !== "auto") {
        const { spaceBetween } = props;
        const remaining = axisSize - (slidesPerView - 1) * spaceBetween;
        const percentage = 1 / Math.max(1, slidesPerView);
        axisSize = remaining * percentage;
      }
      const slideSize = Object.assign(Object.assign({}, perViewSize), { [axis]: axisSize });
      return slidesEls.map(() => slideSize);
    });
    const slideTranlatesRef = computed(() => {
      const { value: slideSizes } = slideSizesRef;
      if (!slideSizes.length)
        return [];
      const { centeredSlides, spaceBetween } = props;
      const { value: axis } = sizeAxisRef;
      const { [axis]: perViewSize } = perViewSizeRef.value;
      let previousTranslate2 = 0;
      return slideSizes.map(({ [axis]: slideSize }) => {
        let translate = previousTranslate2;
        if (centeredSlides) {
          translate += (slideSize - perViewSize) / 2;
        }
        previousTranslate2 += slideSize + spaceBetween;
        return translate;
      });
    });
    const isMountedRef = ref(false);
    const transitionStyleRef = computed(() => {
      const { transitionStyle } = props;
      return transitionStyle ? keep(transitionStyle, transitionProperties) : {};
    });
    const speedRef = computed(() => userWantsControlRef.value ? 0 : resolveSpeed(transitionStyleRef.value.transitionDuration));
    const slideStylesRef = computed(() => {
      const { value: slidesEls } = slideElsRef;
      if (!slidesEls.length)
        return [];
      const useComputedSize = !(autoSlideSizeRef.value || realSlidesPerViewRef.value === 1);
      const getSlideSize = (index) => {
        if (useComputedSize) {
          const { value: axis } = sizeAxisRef;
          return {
            [axis]: `${slideSizesRef.value[index][axis]}px`
          };
        }
      };
      if (userWantsControlRef.value) {
        return slidesEls.map((_, i) => getSlideSize(i));
      }
      const { effect, spaceBetween } = props;
      const { value: spaceAxis } = spaceAxisRef;
      return slidesEls.reduce((styles, _, i) => {
        const style2 = Object.assign(Object.assign({}, getSlideSize(i)), { [`margin-${spaceAxis}`]: `${spaceBetween}px` });
        styles.push(style2);
        if (isMountedRef.value && (effect === "fade" || effect === "card")) {
          Object.assign(style2, transitionStyleRef.value);
        }
        return styles;
      }, []);
    });
    const totalViewRef = computed(() => {
      const { value: slidesPerView } = displaySlidesPerViewRef;
      const { length: totalSlides } = slideElsRef.value;
      if (slidesPerView !== "auto") {
        return Math.max(totalSlides - slidesPerView, 0) + 1;
      } else {
        const { value: slideSizes } = slideSizesRef;
        const { length } = slideSizes;
        if (!length)
          return totalSlides;
        const { value: translates } = slideTranlatesRef;
        const { value: axis } = sizeAxisRef;
        const perViewSize = perViewSizeRef.value[axis];
        let lastViewSize = slideSizes[slideSizes.length - 1][axis];
        let i = length;
        while (i > 1 && lastViewSize < perViewSize) {
          i--;
          lastViewSize += translates[i] - translates[i - 1];
        }
        return clampValue(i + 1, 1, length);
      }
    });
    const displayTotalViewRef = computed(() => getDisplayTotalView(totalViewRef.value, duplicatedableRef.value));
    const defaultRealIndex = getRealIndex(props.defaultIndex, duplicatedableRef.value);
    const uncontrolledDisplayIndexRef = ref(getDisplayIndex(defaultRealIndex, totalViewRef.value, duplicatedableRef.value));
    const mergedDisplayIndexRef = useMergedState(toRef(props, "currentIndex"), uncontrolledDisplayIndexRef);
    const realIndexRef = computed(() => getRealIndex(mergedDisplayIndexRef.value, duplicatedableRef.value));
    function toRealIndex(index) {
      var _a, _b;
      index = clampValue(index, 0, totalViewRef.value - 1);
      const displayIndex = getDisplayIndex(index, totalViewRef.value, duplicatedableRef.value);
      const { value: lastDisplayIndex } = mergedDisplayIndexRef;
      if (displayIndex !== mergedDisplayIndexRef.value) {
        uncontrolledDisplayIndexRef.value = displayIndex;
        (_a = props["onUpdate:currentIndex"]) === null || _a === void 0 ? void 0 : _a.call(props, displayIndex, lastDisplayIndex);
        (_b = props.onUpdateCurrentIndex) === null || _b === void 0 ? void 0 : _b.call(props, displayIndex, lastDisplayIndex);
      }
    }
    function getRealPrevIndex(index = realIndexRef.value) {
      return getPrevIndex(index, totalViewRef.value, props.loop);
    }
    function getRealNextIndex(index = realIndexRef.value) {
      return getNextIndex(index, totalViewRef.value, props.loop);
    }
    function isRealPrev(slideOrIndex) {
      const index = getSlideIndex(slideOrIndex);
      return index !== null && getRealPrevIndex() === index;
    }
    function isRealNext(slideOrIndex) {
      const index = getSlideIndex(slideOrIndex);
      return index !== null && getRealNextIndex() === index;
    }
    function isRealActive(slideOrIndex) {
      return realIndexRef.value === getSlideIndex(slideOrIndex);
    }
    function isDisplayActive(index) {
      return mergedDisplayIndexRef.value === index;
    }
    function isPrevDisabled() {
      return getRealPrevIndex() === null;
    }
    function isNextDisabled() {
      return getRealNextIndex() === null;
    }
    function to(index) {
      const realIndex = clampValue(getRealIndex(index, duplicatedableRef.value), 0, totalViewRef.value);
      if (index !== mergedDisplayIndexRef.value || realIndex !== realIndexRef.value) {
        toRealIndex(realIndex);
      }
    }
    function prev() {
      const prevIndex = getRealPrevIndex();
      if (prevIndex !== null)
        toRealIndex(prevIndex);
    }
    function next() {
      const nextIndex = getRealNextIndex();
      if (nextIndex !== null)
        toRealIndex(nextIndex);
    }
    function prevIfSlideTransitionEnd() {
      if (!inTransition || !duplicatedableRef.value)
        prev();
    }
    function nextIfSlideTransitionEnd() {
      if (!inTransition || !duplicatedableRef.value)
        next();
    }
    let inTransition = false;
    let previousTranslate = 0;
    const translateStyleRef = ref({});
    function updateTranslate(translate, speed = 0) {
      translateStyleRef.value = Object.assign({}, transitionStyleRef.value, {
        transform: verticalRef.value ? `translateY(${-translate}px)` : `translateX(${-translate}px)`,
        transitionDuration: `${speed}ms`
      });
    }
    function fixTranslate(speed = 0) {
      if (sequenceLayoutRef.value) {
        translateTo(realIndexRef.value, speed);
      } else if (previousTranslate !== 0) {
        if (!inTransition && speed > 0) {
          inTransition = true;
        }
        updateTranslate(previousTranslate = 0, speed);
      }
    }
    function translateTo(index, speed) {
      const translate = getTranslate(index);
      if (translate !== previousTranslate && speed > 0) {
        inTransition = true;
      }
      previousTranslate = getTranslate(realIndexRef.value);
      updateTranslate(translate, speed);
    }
    function getTranslate(index) {
      let translate;
      if (index >= totalViewRef.value - 1) {
        translate = getLastViewTranslate();
      } else {
        translate = slideTranlatesRef.value[index] || 0;
      }
      return translate;
    }
    function getLastViewTranslate() {
      if (displaySlidesPerViewRef.value === "auto") {
        const { value: axis } = sizeAxisRef;
        const { [axis]: perViewSize } = perViewSizeRef.value;
        const { value: translates } = slideTranlatesRef;
        const lastTranslate = translates[translates.length - 1];
        let overallSize;
        if (lastTranslate === void 0) {
          overallSize = perViewSize;
        } else {
          const { value: slideSizes } = slideSizesRef;
          overallSize = lastTranslate + slideSizes[slideSizes.length - 1][axis];
        }
        return overallSize - perViewSize;
      } else {
        const { value: translates } = slideTranlatesRef;
        return translates[totalViewRef.value - 1] || 0;
      }
    }
    const carouselContext = {
      currentIndexRef: mergedDisplayIndexRef,
      to,
      prev: prevIfSlideTransitionEnd,
      next: nextIfSlideTransitionEnd,
      isVertical: () => verticalRef.value,
      isHorizontal: () => !verticalRef.value,
      isPrev: isRealPrev,
      isNext: isRealNext,
      isActive: isRealActive,
      isPrevDisabled,
      isNextDisabled,
      getSlideIndex,
      getSlideStyle,
      addSlide,
      removeSlide,
      onCarouselItemClick
    };
    provideCarouselContext(carouselContext);
    function addSlide(slide) {
      if (!slide)
        return;
      slideElsRef.value.push(slide);
    }
    function removeSlide(slide) {
      if (!slide)
        return;
      const index = getSlideIndex(slide);
      if (index !== -1) {
        slideElsRef.value.splice(index, 1);
      }
    }
    function getSlideIndex(slideOrIndex) {
      return typeof slideOrIndex === "number" ? slideOrIndex : slideOrIndex ? slideElsRef.value.indexOf(slideOrIndex) : -1;
    }
    function getSlideStyle(slide) {
      const index = getSlideIndex(slide);
      if (index !== -1) {
        const styles = [slideStylesRef.value[index]];
        const isPrev = carouselContext.isPrev(index);
        const isNext = carouselContext.isNext(index);
        if (isPrev) {
          styles.push(props.prevSlideStyle || "");
        }
        if (isNext) {
          styles.push(props.nextSlideStyle || "");
        }
        return normalizeStyle(styles);
      }
    }
    function onCarouselItemClick(index, event) {
      let allowClick = !inTransition && !dragging && !isEffectiveDrag;
      if (props.effect === "card" && allowClick && !isRealActive(index)) {
        to(index);
        allowClick = false;
      }
      if (!allowClick) {
        event.preventDefault();
        event.stopPropagation();
      }
    }
    let autoplayTimer = null;
    function stopAutoplay() {
      if (autoplayTimer) {
        clearInterval(autoplayTimer);
        autoplayTimer = null;
      }
    }
    function resetAutoplay() {
      stopAutoplay();
      const disabled = !props.autoplay || displayTotalViewRef.value < 2;
      if (!disabled) {
        autoplayTimer = window.setInterval(next, props.interval);
      }
    }
    let dragStartX = 0;
    let dragStartY = 0;
    let dragOffset = 0;
    let dragStartTime = 0;
    let dragging = false;
    let isEffectiveDrag = false;
    function handleTouchstart(event) {
      var _a;
      if (globalDragging)
        return;
      if (!((_a = slidesElRef.value) === null || _a === void 0 ? void 0 : _a.contains(getPreciseEventTarget(event)))) {
        return;
      }
      globalDragging = true;
      dragging = true;
      isEffectiveDrag = false;
      dragStartTime = Date.now();
      stopAutoplay();
      if (event.type !== "touchstart" && !event.target.isContentEditable) {
        event.preventDefault();
      }
      const touchEvent = isTouchEvent(event) ? event.touches[0] : event;
      if (verticalRef.value) {
        dragStartY = touchEvent.clientY;
      } else {
        dragStartX = touchEvent.clientX;
      }
      if (props.touchable) {
        on("touchmove", document, handleTouchmove, { passive: true });
        on("touchend", document, handleTouchend);
        on("touchcancel", document, handleTouchend);
      }
      if (props.draggable) {
        on("mousemove", document, handleTouchmove);
        on("mouseup", document, handleTouchend);
      }
    }
    function handleTouchmove(event) {
      const { value: vertical } = verticalRef;
      const { value: axis } = sizeAxisRef;
      const touchEvent = isTouchEvent(event) ? event.touches[0] : event;
      const offset = vertical ? touchEvent.clientY - dragStartY : touchEvent.clientX - dragStartX;
      const perViewSize = perViewSizeRef.value[axis];
      dragOffset = clampValue(offset, -perViewSize, perViewSize);
      if (event.cancelable) {
        event.preventDefault();
      }
      if (sequenceLayoutRef.value) {
        updateTranslate(previousTranslate - dragOffset, 0);
      }
    }
    function handleTouchend() {
      const { value: realIndex } = realIndexRef;
      let currentIndex = realIndex;
      if (!inTransition && dragOffset !== 0 && sequenceLayoutRef.value) {
        const currentTranslate = previousTranslate - dragOffset;
        const translates = [
          ...slideTranlatesRef.value.slice(0, totalViewRef.value - 1),
          getLastViewTranslate()
        ];
        let prevOffset = null;
        for (let i = 0; i < translates.length; i++) {
          const offset = Math.abs(translates[i] - currentTranslate);
          if (prevOffset !== null && prevOffset < offset) {
            break;
          }
          prevOffset = offset;
          currentIndex = i;
        }
      }
      if (currentIndex === realIndex) {
        const timeElapsed = Date.now() - dragStartTime;
        const { value: axis } = sizeAxisRef;
        const perViewSize = perViewSizeRef.value[axis];
        if (dragOffset > perViewSize / 2 || dragOffset / timeElapsed > 0.4) {
          currentIndex = getRealPrevIndex(realIndex);
        } else if (dragOffset < -perViewSize / 2 || dragOffset / timeElapsed < -0.4) {
          currentIndex = getRealNextIndex(realIndex);
        }
      }
      if (currentIndex !== null && currentIndex !== realIndex) {
        isEffectiveDrag = true;
        toRealIndex(currentIndex);
        void nextTick(() => {
          if (!duplicatedableRef.value || uncontrolledDisplayIndexRef.value !== mergedDisplayIndexRef.value) {
            fixTranslate(speedRef.value);
          }
        });
      } else {
        fixTranslate(speedRef.value);
      }
      resetDragStatus();
      resetAutoplay();
    }
    function resetDragStatus() {
      if (dragging) {
        globalDragging = false;
      }
      dragging = false;
      dragStartX = 0;
      dragStartY = 0;
      dragOffset = 0;
      dragStartTime = 0;
      off("touchmove", document, handleTouchmove);
      off("touchend", document, handleTouchend);
      off("touchcancel", document, handleTouchend);
      off("mousemove", document, handleTouchmove);
      off("mouseup", document, handleTouchend);
    }
    function handleTransitionEnd() {
      if (sequenceLayoutRef.value && inTransition) {
        const { value: realIndex } = realIndexRef;
        translateTo(realIndex, 0);
      } else {
        resetAutoplay();
      }
      if (sequenceLayoutRef.value) {
        translateStyleRef.value.transitionDuration = "0ms";
      }
      inTransition = false;
    }
    function handleMousewheel(event) {
      event.preventDefault();
      if (inTransition)
        return;
      let { deltaX, deltaY } = event;
      if (event.shiftKey && !deltaX) {
        deltaX = deltaY;
      }
      const prevMultiplier = -1;
      const nextMultiplier = 1;
      const m = (deltaX || deltaY) > 0 ? nextMultiplier : prevMultiplier;
      let rx = 0;
      let ry = 0;
      if (verticalRef.value) {
        ry = m;
      } else {
        rx = m;
      }
      const responseStep = 10;
      if (ry * deltaY >= responseStep || rx * deltaX >= responseStep) {
        if (m === nextMultiplier && !isNextDisabled()) {
          next();
        } else if (m === prevMultiplier && !isPrevDisabled()) {
          prev();
        }
      }
    }
    function handleResize() {
      perViewSizeRef.value = calculateSize(selfElRef.value, true);
      resetAutoplay();
    }
    function handleSlideResize() {
      var _a, _b;
      if (autoSlideSizeRef.value) {
        (_b = (_a = slideSizesRef.effect).scheduler) === null || _b === void 0 ? void 0 : _b.call(_a);
        slideSizesRef.effect.run();
      }
    }
    function handleMouseenter() {
      if (props.autoplay) {
        stopAutoplay();
      }
    }
    function handleMouseleave() {
      if (props.autoplay) {
        resetAutoplay();
      }
    }
    onMounted(() => {
      watchEffect(resetAutoplay);
      requestAnimationFrame(() => isMountedRef.value = true);
    });
    onBeforeUnmount(() => {
      resetDragStatus();
      stopAutoplay();
    });
    onUpdated(() => {
      const { value: slidesEls } = slideElsRef;
      const { value: slideVNodes } = slideVNodesRef;
      const indexMap2 = /* @__PURE__ */ new Map();
      const getDisplayIndex2 = (el) => (
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        indexMap2.has(el) ? indexMap2.get(el) : -1
      );
      let isChanged = false;
      for (let i = 0; i < slidesEls.length; i++) {
        const index = slideVNodes.findIndex((v) => v.el === slidesEls[i]);
        if (index !== i) {
          isChanged = true;
        }
        indexMap2.set(slidesEls[i], index);
      }
      if (isChanged) {
        slidesEls.sort((a, b) => getDisplayIndex2(a) - getDisplayIndex2(b));
      }
    });
    watch(realIndexRef, (realIndex, lastRealIndex) => {
      if (realIndex === lastRealIndex)
        return;
      resetAutoplay();
      if (sequenceLayoutRef.value) {
        if (duplicatedableRef.value && displayTotalViewRef.value > 2) {
          const { value: length } = totalViewRef;
          if (realIndex === length - 2 && lastRealIndex === 1) {
            realIndex = 0;
          } else if (realIndex === 1 && lastRealIndex === length - 2) {
            realIndex = length - 1;
          }
        }
        translateTo(realIndex, speedRef.value);
      } else {
        fixTranslate();
      }
    }, { immediate: true });
    watch([duplicatedableRef, displaySlidesPerViewRef], () => void nextTick(() => {
      toRealIndex(realIndexRef.value);
    }));
    watch(slideTranlatesRef, () => {
      sequenceLayoutRef.value && fixTranslate();
    }, {
      deep: true
    });
    watch(sequenceLayoutRef, (value) => {
      if (!value) {
        inTransition = false;
        updateTranslate(previousTranslate = 0);
      } else {
        fixTranslate();
      }
    });
    const slidesControlListenersRef = computed(() => {
      return {
        onTouchstartPassive: props.touchable ? handleTouchstart : void 0,
        onMousedown: props.draggable ? handleTouchstart : void 0,
        onWheel: props.mousewheel ? handleMousewheel : void 0
      };
    });
    const arrowSlotPropsRef = computed(() => Object.assign(Object.assign({}, keep(carouselContext, [
      "to",
      "prev",
      "next",
      "isPrevDisabled",
      "isNextDisabled"
    ])), { total: displayTotalViewRef.value, currentIndex: mergedDisplayIndexRef.value }));
    const dotSlotPropsRef = computed(() => ({
      total: displayTotalViewRef.value,
      currentIndex: mergedDisplayIndexRef.value,
      to: carouselContext.to
    }));
    const caroulseExposedMethod = {
      getCurrentIndex: () => mergedDisplayIndexRef.value,
      to,
      prev,
      next
    };
    const themeRef = useTheme("Carousel", "-carousel", style$1, carouselLight, props, mergedClsPrefixRef);
    const cssVarsRef = computed(() => {
      const { common: { cubicBezierEaseInOut }, self: { dotSize, dotColor, dotColorActive, dotColorFocus, dotLineWidth, dotLineWidthActive, arrowColor } } = themeRef.value;
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-dot-color": dotColor,
        "--n-dot-color-focus": dotColorFocus,
        "--n-dot-color-active": dotColorActive,
        "--n-dot-size": dotSize,
        "--n-dot-line-width": dotLineWidth,
        "--n-dot-line-width-active": dotLineWidthActive,
        "--n-arrow-color": arrowColor
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("carousel", void 0, cssVarsRef, props) : void 0;
    return Object.assign(Object.assign({
      mergedClsPrefix: mergedClsPrefixRef,
      selfElRef,
      slidesElRef,
      slideVNodes: slideVNodesRef,
      duplicatedable: duplicatedableRef,
      userWantsControl: userWantsControlRef,
      autoSlideSize: autoSlideSizeRef,
      displayIndex: mergedDisplayIndexRef,
      realIndex: realIndexRef,
      slideStyles: slideStylesRef,
      translateStyle: translateStyleRef,
      slidesControlListeners: slidesControlListenersRef,
      handleTransitionEnd,
      handleResize,
      handleSlideResize,
      handleMouseenter,
      handleMouseleave,
      isActive: isDisplayActive,
      arrowSlotProps: arrowSlotPropsRef,
      dotSlotProps: dotSlotPropsRef
    }, caroulseExposedMethod), { cssVars: inlineThemeDisabled ? void 0 : cssVarsRef, themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass, onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender });
  },
  render() {
    var _a;
    const { mergedClsPrefix, showArrow, userWantsControl, slideStyles, dotType, dotPlacement, slidesControlListeners, transitionProps = {}, arrowSlotProps, dotSlotProps, $slots: { default: defaultSlot, dots: dotsSlot, arrow: arrowSlot } } = this;
    const children = defaultSlot && flatten(defaultSlot()) || [];
    let slides = filterCarouselItem(children);
    if (!slides.length) {
      slides = children.map((ch) => h(NCarouselItem, null, {
        default: () => cloneVNode(ch)
      }));
    }
    if (this.duplicatedable) {
      slides = addDuplicateSlides(slides);
    }
    this.slideVNodes.value = slides;
    if (this.autoSlideSize) {
      slides = slides.map((slide) => h(VResizeObserver, { onResize: this.handleSlideResize }, {
        default: () => slide
      }));
    }
    (_a = this.onRender) === null || _a === void 0 ? void 0 : _a.call(this);
    return h(
      "div",
      Object.assign({ ref: "selfElRef", class: [
        this.themeClass,
        `${mergedClsPrefix}-carousel`,
        this.direction === "vertical" && `${mergedClsPrefix}-carousel--vertical`,
        this.showArrow && `${mergedClsPrefix}-carousel--show-arrow`,
        `${mergedClsPrefix}-carousel--${dotPlacement}`,
        `${mergedClsPrefix}-carousel--${this.direction}`,
        `${mergedClsPrefix}-carousel--${this.effect}`,
        userWantsControl && `${mergedClsPrefix}-carousel--usercontrol`
      ], style: this.cssVars }, slidesControlListeners, { onMouseenter: this.handleMouseenter, onMouseleave: this.handleMouseleave }),
      h(VResizeObserver, { onResize: this.handleResize }, {
        default: () => h("div", { ref: "slidesElRef", class: `${mergedClsPrefix}-carousel__slides`, role: "listbox", style: this.translateStyle, onTransitionend: this.handleTransitionEnd }, userWantsControl ? slides.map((slide, i) => h("div", { style: slideStyles[i], key: i }, withDirectives(h(Transition, Object.assign({}, transitionProps), {
          default: () => slide
        }), [[vShow, this.isActive(i)]]))) : slides)
      }),
      this.showDots && dotSlotProps.total > 1 && resolveSlotWithProps(dotsSlot, dotSlotProps, () => [
        h(NCarouselDots, { key: dotType + dotPlacement, total: dotSlotProps.total, currentIndex: dotSlotProps.currentIndex, dotType, trigger: this.trigger, keyboard: this.keyboard })
      ]),
      showArrow && resolveSlotWithProps(arrowSlot, arrowSlotProps, () => [
        h(NCarouselArrow, null)
      ])
    );
  }
});
function filterCarouselItem(vnodes) {
  return vnodes.reduce((carouselItems, vnode) => {
    if (isCarouselItem(vnode)) {
      carouselItems.push(vnode);
    }
    return carouselItems;
  }, []);
}
const StarIcon = h(
  "svg",
  { viewBox: "0 0 512 512" },
  h("path", { d: "M394 480a16 16 0 01-9.39-3L256 383.76 127.39 477a16 16 0 01-24.55-18.08L153 310.35 23 221.2a16 16 0 019-29.2h160.38l48.4-148.95a16 16 0 0130.44 0l48.4 149H480a16 16 0 019.05 29.2L359 310.35l50.13 148.53A16 16 0 01394 480z" })
);
const style = cB("rate", {
  display: "inline-flex",
  flexWrap: "nowrap"
}, [c("&:hover", [cE("item", `
 transition:
 transform .1s var(--n-bezier),
 color .3s var(--n-bezier);
 `)]), cE("item", `
 position: relative;
 display: flex;
 transition:
 transform .1s var(--n-bezier),
 color .3s var(--n-bezier);
 transform: scale(1);
 font-size: var(--n-item-size);
 color: var(--n-item-color);
 `, [c("&:not(:first-child)", `
 margin-left: 6px;
 `), cM("active", `
 color: var(--n-item-color-active);
 `)]), cNotM("readonly", `
 cursor: pointer;
 `, [cE("item", [c("&:hover", `
 transform: scale(1.05);
 `), c("&:active", `
 transform: scale(0.96);
 `)])]), cE("half", `
 display: flex;
 transition: inherit;
 position: absolute;
 top: 0;
 left: 0;
 bottom: 0;
 width: 50%;
 overflow: hidden;
 color: rgba(255, 255, 255, 0);
 `, [cM("active", `
 color: var(--n-item-color-active);
 `)])]);
const rateProps = Object.assign(Object.assign({}, useTheme.props), { allowHalf: Boolean, count: {
  type: Number,
  default: 5
}, value: Number, defaultValue: {
  type: Number,
  default: null
}, readonly: Boolean, size: {
  type: [String, Number],
  default: "medium"
}, clearable: Boolean, color: String, onClear: Function, "onUpdate:value": [Function, Array], onUpdateValue: [Function, Array] });
const NRate = defineComponent({
  name: "Rate",
  props: rateProps,
  setup(props) {
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const themeRef = useTheme("Rate", "-rate", style, rateLight, props, mergedClsPrefixRef);
    const controlledValueRef = toRef(props, "value");
    const uncontrolledValueRef = ref(props.defaultValue);
    const hoverIndexRef = ref(null);
    const formItem = useFormItem(props);
    const mergedValue = useMergedState(controlledValueRef, uncontrolledValueRef);
    function doUpdateValue(value) {
      const { "onUpdate:value": _onUpdateValue, onUpdateValue } = props;
      const { nTriggerFormChange, nTriggerFormInput } = formItem;
      if (_onUpdateValue) {
        call(_onUpdateValue, value);
      }
      if (onUpdateValue) {
        call(onUpdateValue, value);
      }
      uncontrolledValueRef.value = value;
      nTriggerFormChange();
      nTriggerFormInput();
    }
    function getDerivedValue(index, e) {
      if (props.allowHalf) {
        if (e.offsetX >= Math.floor(e.currentTarget.offsetWidth / 2)) {
          return index + 1;
        } else {
          return index + 0.5;
        }
      } else {
        return index + 1;
      }
    }
    let cleared = false;
    function handleMouseMove(index, e) {
      if (cleared)
        return;
      hoverIndexRef.value = getDerivedValue(index, e);
    }
    function handleMouseLeave() {
      hoverIndexRef.value = null;
    }
    function handleClick(index, e) {
      var _a;
      const { clearable } = props;
      const derivedValue = getDerivedValue(index, e);
      if (clearable && derivedValue === mergedValue.value) {
        cleared = true;
        (_a = props.onClear) === null || _a === void 0 ? void 0 : _a.call(props);
        hoverIndexRef.value = null;
        doUpdateValue(null);
      } else {
        doUpdateValue(derivedValue);
      }
    }
    function handleMouseEnterSomeStar() {
      cleared = false;
    }
    const mergedSizeRef = computed(() => {
      const { size } = props;
      const { self } = themeRef.value;
      if (typeof size === "number") {
        return `${size}px`;
      } else {
        return self[createKey("size", size)];
      }
    });
    const cssVarsRef = computed(() => {
      const { common: { cubicBezierEaseInOut }, self } = themeRef.value;
      const { itemColor, itemColorActive } = self;
      const { color } = props;
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-item-color": itemColor,
        "--n-item-color-active": color || itemColorActive,
        "--n-item-size": mergedSizeRef.value
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("rate", computed(() => {
      const size = mergedSizeRef.value;
      const { color } = props;
      let hash = "";
      if (size) {
        hash += size[0];
      }
      if (color) {
        hash += color2Class(color);
      }
      return hash;
    }), cssVarsRef, props) : void 0;
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      mergedValue,
      hoverIndex: hoverIndexRef,
      handleMouseMove,
      handleClick,
      handleMouseLeave,
      handleMouseEnterSomeStar,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    const { readonly, hoverIndex, mergedValue, mergedClsPrefix, onRender, $slots: { default: defaultSlot } } = this;
    onRender === null || onRender === void 0 ? void 0 : onRender();
    return h("div", { class: [
      `${mergedClsPrefix}-rate`,
      {
        [`${mergedClsPrefix}-rate--readonly`]: readonly
      },
      this.themeClass
    ], style: this.cssVars, onMouseleave: this.handleMouseLeave }, renderList(this.count, (_, index) => {
      const icon = defaultSlot ? defaultSlot({ index }) : h(NBaseIcon, { clsPrefix: mergedClsPrefix }, { default: () => StarIcon });
      const entireStarActive = hoverIndex !== null ? index + 1 <= hoverIndex : index + 1 <= (mergedValue || 0);
      return h(
        "div",
        { key: index, class: [
          `${mergedClsPrefix}-rate__item`,
          entireStarActive && `${mergedClsPrefix}-rate__item--active`
        ], onClick: readonly ? void 0 : (e) => {
          this.handleClick(index, e);
        }, onMouseenter: this.handleMouseEnterSomeStar, onMousemove: readonly ? void 0 : (e) => {
          this.handleMouseMove(index, e);
        } },
        icon,
        this.allowHalf ? h("div", { class: [
          `${mergedClsPrefix}-rate__half`,
          {
            [`${mergedClsPrefix}-rate__half--active`]: !entireStarActive && hoverIndex !== null ? index + 0.5 <= hoverIndex : index + 0.5 <= (mergedValue || 0)
          }
        ] }, icon) : null
      );
    }));
  }
});
function nsfwIndex(nsfwLevel) {
  switch (nsfwLevel) {
    case "None":
      return 0;
    case "Soft":
      return 1;
    case "Mature":
      return 2;
    case "X":
      return 3;
  }
}
const _hoisted_1 = ["src"];
const _hoisted_2 = /* @__PURE__ */ createBaseVNode("i", null, [
  /* @__PURE__ */ createTextVNode("Data provided by "),
  /* @__PURE__ */ createBaseVNode("a", { href: "https://civitai.com" }, "CivitAI"),
  /* @__PURE__ */ createTextVNode(", go and support them")
], -1);
const _hoisted_3 = { style: { "height": "90%" } };
const _hoisted_4 = { style: { "display": "inline-flex", "justify-content": "center", "align-items": "center" } };
const _hoisted_5 = { style: { "margin-top": "0", "margin-bottom": "0" } };
const _hoisted_6 = { style: { "line-height": "32px" } };
const _hoisted_7 = { style: { "width": "100%", "display": "inline-flex", "height": "40px", "align-items": "center", "margin-top": "8px" } };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ModelPopup",
  props: {
    model: {},
    showModal: { type: Boolean }
  },
  emits: ["update:showModal"],
  setup(__props, { emit }) {
    const props = __props;
    const message = useMessage();
    const settings = useSettings();
    const tabValue = ref("");
    const tabsInstRef = ref(null);
    const selectedModel = reactive(/* @__PURE__ */ new Map());
    const dateFormat = new Intl.DateTimeFormat(navigator.language, {
      year: "numeric",
      month: "long",
      day: "numeric",
      minute: "numeric",
      hour: "numeric"
    });
    watch(props, (newProps) => {
      if (newProps.model) {
        tabValue.value = newProps.model.modelVersions[0].name;
      }
      nextTick(() => {
        var _a;
        (_a = tabsInstRef.value) == null ? void 0 : _a.syncBarPosition();
      });
    });
    function generateDownloadOptions(submodel) {
      return submodel.map((file) => ({
        label: `${file.metadata.format} ${file.metadata.size} ${file.metadata.fp} [${(file.sizeKB / 1024 / 1024).toFixed(2)} GB]`,
        value: file.downloadUrl
      }));
    }
    function downloadModel(model) {
      var _a;
      message.success("Download started");
      const url = new URL(`${serverUrl}/api/models/download-model`);
      url.searchParams.append("link", model.downloadUrl);
      url.searchParams.append("model_type", (_a = props.model) == null ? void 0 : _a.type);
      fetch(url, { method: "POST" }).then((res) => {
        if (res.ok) {
          message.success("Download finished");
        } else {
          message.error(`Download failed: ${res.status}`);
        }
      }).catch((e) => {
        message.error(`Download failed: ${e}`);
      });
    }
    return (_ctx, _cache) => {
      var _a, _b;
      return openBlock(), createBlock(unref(NModal), {
        show: _ctx.showModal,
        title: ((_a = _ctx.model) == null ? void 0 : _a.name) + " (by " + ((_b = _ctx.model) == null ? void 0 : _b.creator.username) + ")" || "Loading...",
        preset: "card",
        style: { "width": "90vw" },
        "onUpdate:show": _cache[1] || (_cache[1] = ($event) => emit("update:showModal", $event))
      }, {
        default: withCtx(() => [
          createVNode(unref(NTabs), {
            "justify-content": "start",
            type: "bar",
            value: tabValue.value,
            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => tabValue.value = $event),
            animated: ""
          }, {
            default: withCtx(() => {
              var _a2;
              return [
                (openBlock(true), createElementBlock(Fragment, null, renderList((_a2 = props.model) == null ? void 0 : _a2.modelVersions, (subModel) => {
                  return openBlock(), createBlock(unref(NTabPane), {
                    name: subModel.name,
                    key: subModel.id,
                    style: { "display": "flex", "flex-direction": "column" }
                  }, {
                    default: withCtx(() => [
                      createVNode(unref(NGrid), { cols: "1 850:2" }, {
                        default: withCtx(() => [
                          createVNode(unref(NGi), null, {
                            default: withCtx(() => [
                              createVNode(unref(NCarousel), {
                                style: { "height": "70vh", "width": "100%" },
                                draggable: "",
                                "slides-per-view": 2,
                                "centered-slides": true,
                                effect: "card",
                                "dot-type": "line",
                                keyboard: "",
                                mousewheel: ""
                              }, {
                                default: withCtx(() => [
                                  (openBlock(true), createElementBlock(Fragment, null, renderList(subModel.images.length > 1 ? subModel.images : [subModel.images[0], subModel.images[0]], (image) => {
                                    return openBlock(), createElementBlock("div", {
                                      key: image.hash
                                    }, [
                                      createBaseVNode("img", {
                                        src: image.url,
                                        style: normalizeStyle({
                                          width: "100%",
                                          filter: unref(nsfwIndex)(image.nsfw) > unref(settings).data.settings.frontend.nsfw_ok_threshold ? "blur(12px)" : "none"
                                        })
                                      }, null, 12, _hoisted_1)
                                    ]);
                                  }), 128))
                                ]),
                                _: 2
                              }, 1024)
                            ]),
                            _: 2
                          }, 1024),
                          createVNode(unref(NGi), null, {
                            default: withCtx(() => [
                              createVNode(unref(NCard), {
                                title: subModel.name,
                                style: { "height": "auto" },
                                segmented: "",
                                hoverable: "",
                                "content-style": {
                                  paddingBottom: "8px"
                                }
                              }, {
                                footer: withCtx(() => [
                                  _hoisted_2
                                ]),
                                default: withCtx(() => {
                                  var _a3;
                                  return [
                                    createBaseVNode("div", _hoisted_3, [
                                      createBaseVNode("div", _hoisted_4, [
                                        createVNode(unref(NRate), {
                                          value: subModel.stats.rating,
                                          "allow-half": "",
                                          readonly: ""
                                        }, null, 8, ["value"]),
                                        createBaseVNode("p", _hoisted_5, [
                                          createTextVNode(" ("),
                                          createBaseVNode("i", null, toDisplayString(subModel.stats.ratingCount), 1),
                                          createTextVNode(") ")
                                        ])
                                      ]),
                                      createBaseVNode("div", _hoisted_6, [
                                        (openBlock(true), createElementBlock(Fragment, null, renderList((_a3 = _ctx.model) == null ? void 0 : _a3.tags, (tag) => {
                                          return openBlock(), createBlock(unref(NTag), {
                                            key: tag,
                                            style: { "margin-right": "4px" }
                                          }, {
                                            default: withCtx(() => [
                                              createTextVNode(toDisplayString(tag), 1)
                                            ]),
                                            _: 2
                                          }, 1024);
                                        }), 128))
                                      ]),
                                      createVNode(unref(NDescriptions), {
                                        "label-placement": "left",
                                        "label-align": "left",
                                        bordered: "",
                                        columns: 1,
                                        style: { "margin-top": "8px" }
                                      }, {
                                        default: withCtx(() => [
                                          createVNode(unref(NDescriptionsItem), { label: "Base Model" }, {
                                            default: withCtx(() => [
                                              createTextVNode(toDisplayString(subModel.baseModel), 1)
                                            ]),
                                            _: 2
                                          }, 1024),
                                          createVNode(unref(NDescriptionsItem), { label: "Downloads" }, {
                                            default: withCtx(() => [
                                              createTextVNode(toDisplayString(subModel.stats.downloadCount.toLocaleString()), 1)
                                            ]),
                                            _: 2
                                          }, 1024),
                                          createVNode(unref(NDescriptionsItem), { label: "Keywords" }, {
                                            default: withCtx(() => [
                                              createTextVNode(toDisplayString(subModel.trainedWords.length !== 0 ? subModel.trainedWords.join(", ") : "No keywords"), 1)
                                            ]),
                                            _: 2
                                          }, 1024),
                                          createVNode(unref(NDescriptionsItem), { label: "Last Updated" }, {
                                            default: withCtx(() => [
                                              createTextVNode(toDisplayString(unref(dateFormat).format(new Date(subModel.updatedAt))), 1)
                                            ]),
                                            _: 2
                                          }, 1024),
                                          createVNode(unref(NDescriptionsItem), { label: "Created" }, {
                                            default: withCtx(() => [
                                              createTextVNode(toDisplayString(unref(dateFormat).format(new Date(subModel.createdAt))), 1)
                                            ]),
                                            _: 2
                                          }, 1024)
                                        ]),
                                        _: 2
                                      }, 1024)
                                    ]),
                                    createBaseVNode("div", _hoisted_7, [
                                      createVNode(unref(NSelect), {
                                        options: generateDownloadOptions(subModel.files),
                                        onUpdateValue: (value) => selectedModel.set(subModel.name, value)
                                      }, null, 8, ["options", "onUpdateValue"]),
                                      createVNode(unref(NButton), {
                                        style: { "margin-left": "4px" },
                                        type: "primary",
                                        ghost: "",
                                        disabled: !selectedModel.get(subModel.name),
                                        onClick: () => downloadModel(subModel)
                                      }, {
                                        default: withCtx(() => [
                                          createTextVNode(" Download ")
                                        ]),
                                        _: 2
                                      }, 1032, ["disabled", "onClick"])
                                    ])
                                  ];
                                }),
                                _: 2
                              }, 1032, ["title"])
                            ]),
                            _: 2
                          }, 1024)
                        ]),
                        _: 2
                      }, 1024)
                    ]),
                    _: 2
                  }, 1032, ["name"]);
                }), 128))
              ];
            }),
            _: 1
          }, 8, ["value"])
        ]),
        _: 1
      }, 8, ["show", "title"]);
    };
  }
});
export {
  _sfc_main as _,
  nsfwIndex as n
};
