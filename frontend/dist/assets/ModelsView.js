import { d as defineComponent, D as h, P as replaceable, Q as useConfig, R as useFormItem, E as ref, c as computed, S as useMergedState, T as provide, U as toRef, V as createInjectionKey, W as call, X as c, Y as cB, Z as cE, $ as cM, a0 as iconSwitchTransition, a1 as insideModal, a2 as insidePopover, a3 as inject, a4 as useMemo, a5 as useTheme, a6 as checkboxLight, a7 as useRtl, a8 as createKey, a9 as useThemeClass, aa as createId, ab as NIconSwitchTransition, ac as on, ad as popselectLight, J as watch, ae as NInternalSelectMenu, af as createTreeMate, ag as happensIn, ah as nextTick, ai as keysOf, aj as createTmOptions, ak as keep, al as createRefSetter, am as mergeEventHandlers, an as omit, ao as NPopover, ap as popoverBaseProps, aq as cNotM, ar as useLocale, as as watchEffect, at as resolveSlot, k as NInput, r as NSelect, L as Fragment, au as NBaseIcon, av as useAdjustedTo, aw as paginationLight, ax as ellipsisLight, ay as onDeactivated, q as NTooltip, az as mergeProps, aA as radioLight, aB as resolveWrappedSlot, aC as flatten$1, aD as getSlot, aE as depx, aF as formatLength, F as NButton, aG as NScrollbar, aH as onBeforeUnmount, aI as off, aJ as ChevronDownIcon, aK as NDropdown, aL as pxfy, aM as get, aN as NBaseLoading, aO as ChevronRightIcon, o as onUnmounted, aP as VResizeObserver, aQ as warn, aR as VVirtualList, aS as NEmpty, aT as cssrAnchorMetaName, aU as repeat, aV as beforeNextFrameOnce, aW as fadeInScaleUpTransition, aX as Transition, aY as dataTableLight, aZ as throwError, a_ as isBrowser, a$ as AddIcon, b0 as NProgress, b1 as NFadeInExpandTransition, b2 as EyeIcon, b3 as fadeInHeightExpandTransition, b4 as Teleport, b5 as uploadLight, e as openBlock, f as createElementBlock, n as createBaseVNode, x as createBlock, w as withCtx, g as createVNode, h as unref, G as NIcon, b6 as NResult, u as useState, b as useMessage, b7 as reactive, b8 as huggingfaceModelsFile, m as createTextVNode, i as NCard, s as serverUrl, B as pushScopeId, C as popScopeId, _ as _export_sfc, j as NSpace, b9 as NModal, N as NGi, ba as NText, M as renderList, t as toDisplayString, z as NGrid, H as NTabPane, I as NTabs } from "./index.js";
import { N as NSwitch } from "./Switch.js";
import { N as NImage, a as NImageGroup } from "./Image.js";
import { C as CloudUpload } from "./CloudUpload.js";
import { T as TrashBin } from "./TrashBin.js";
function smallerSize(size) {
  switch (size) {
    case "tiny":
      return "mini";
    case "small":
      return "tiny";
    case "medium":
      return "small";
    case "large":
      return "medium";
    case "huge":
      return "large";
  }
  throw Error(`${size} has no smaller size.`);
}
const ArrowDownIcon = defineComponent({
  name: "ArrowDown",
  render() {
    return h(
      "svg",
      { viewBox: "0 0 28 28", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
      h(
        "g",
        { stroke: "none", "stroke-width": "1", "fill-rule": "evenodd" },
        h(
          "g",
          { "fill-rule": "nonzero" },
          h("path", { d: "M23.7916,15.2664 C24.0788,14.9679 24.0696,14.4931 23.7711,14.206 C23.4726,13.9188 22.9978,13.928 22.7106,14.2265 L14.7511,22.5007 L14.7511,3.74792 C14.7511,3.33371 14.4153,2.99792 14.0011,2.99792 C13.5869,2.99792 13.2511,3.33371 13.2511,3.74793 L13.2511,22.4998 L5.29259,14.2265 C5.00543,13.928 4.53064,13.9188 4.23213,14.206 C3.93361,14.4931 3.9244,14.9679 4.21157,15.2664 L13.2809,24.6944 C13.6743,25.1034 14.3289,25.1034 14.7223,24.6944 L23.7916,15.2664 Z" })
        )
      )
    );
  }
});
const AttachIcon = replaceable("attach", h(
  "svg",
  { viewBox: "0 0 16 16", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
  h(
    "g",
    { stroke: "none", "stroke-width": "1", fill: "none", "fill-rule": "evenodd" },
    h(
      "g",
      { fill: "currentColor", "fill-rule": "nonzero" },
      h("path", { d: "M3.25735931,8.70710678 L7.85355339,4.1109127 C8.82986412,3.13460197 10.4127766,3.13460197 11.3890873,4.1109127 C12.365398,5.08722343 12.365398,6.67013588 11.3890873,7.64644661 L6.08578644,12.9497475 C5.69526215,13.3402718 5.06209717,13.3402718 4.67157288,12.9497475 C4.28104858,12.5592232 4.28104858,11.9260582 4.67157288,11.5355339 L9.97487373,6.23223305 C10.1701359,6.0369709 10.1701359,5.72038841 9.97487373,5.52512627 C9.77961159,5.32986412 9.4630291,5.32986412 9.26776695,5.52512627 L3.96446609,10.8284271 C3.18341751,11.6094757 3.18341751,12.8758057 3.96446609,13.6568542 C4.74551468,14.4379028 6.01184464,14.4379028 6.79289322,13.6568542 L12.0961941,8.35355339 C13.4630291,6.98671837 13.4630291,4.77064094 12.0961941,3.40380592 C10.7293591,2.0369709 8.51328163,2.0369709 7.14644661,3.40380592 L2.55025253,8 C2.35499039,8.19526215 2.35499039,8.51184464 2.55025253,8.70710678 C2.74551468,8.90236893 3.06209717,8.90236893 3.25735931,8.70710678 Z" })
    )
  )
));
const BackwardIcon = defineComponent({
  name: "Backward",
  render() {
    return h(
      "svg",
      { viewBox: "0 0 20 20", fill: "none", xmlns: "http://www.w3.org/2000/svg" },
      h("path", { d: "M12.2674 15.793C11.9675 16.0787 11.4927 16.0672 11.2071 15.7673L6.20572 10.5168C5.9298 10.2271 5.9298 9.7719 6.20572 9.48223L11.2071 4.23177C11.4927 3.93184 11.9675 3.92031 12.2674 4.206C12.5673 4.49169 12.5789 4.96642 12.2932 5.26634L7.78458 9.99952L12.2932 14.7327C12.5789 15.0326 12.5673 15.5074 12.2674 15.793Z", fill: "currentColor" })
    );
  }
});
const TrashIcon = replaceable("trash", h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 512 512" },
  h("path", { d: "M432,144,403.33,419.74A32,32,0,0,1,371.55,448H140.46a32,32,0,0,1-31.78-28.26L80,144", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 32px;" }),
  h("rect", { x: "32", y: "64", width: "448", height: "80", rx: "16", ry: "16", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 32px;" }),
  h("line", { x1: "312", y1: "240", x2: "200", y2: "352", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 32px;" }),
  h("line", { x1: "312", y1: "352", x2: "200", y2: "240", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 32px;" })
));
const DownloadIcon = replaceable("download", h(
  "svg",
  { viewBox: "0 0 16 16", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
  h(
    "g",
    { stroke: "none", "stroke-width": "1", fill: "none", "fill-rule": "evenodd" },
    h(
      "g",
      { fill: "currentColor", "fill-rule": "nonzero" },
      h("path", { d: "M3.5,13 L12.5,13 C12.7761424,13 13,13.2238576 13,13.5 C13,13.7454599 12.8231248,13.9496084 12.5898756,13.9919443 L12.5,14 L3.5,14 C3.22385763,14 3,13.7761424 3,13.5 C3,13.2545401 3.17687516,13.0503916 3.41012437,13.0080557 L3.5,13 L12.5,13 L3.5,13 Z M7.91012437,1.00805567 L8,1 C8.24545989,1 8.44960837,1.17687516 8.49194433,1.41012437 L8.5,1.5 L8.5,10.292 L11.1819805,7.6109127 C11.3555469,7.43734635 11.6249713,7.4180612 11.8198394,7.55305725 L11.8890873,7.6109127 C12.0626536,7.78447906 12.0819388,8.05390346 11.9469427,8.2487716 L11.8890873,8.31801948 L8.35355339,11.8535534 C8.17998704,12.0271197 7.91056264,12.0464049 7.7156945,11.9114088 L7.64644661,11.8535534 L4.1109127,8.31801948 C3.91565056,8.12275734 3.91565056,7.80617485 4.1109127,7.6109127 C4.28447906,7.43734635 4.55390346,7.4180612 4.7487716,7.55305725 L4.81801948,7.6109127 L7.5,10.292 L7.5,1.5 C7.5,1.25454011 7.67687516,1.05039163 7.91012437,1.00805567 L8,1 L7.91012437,1.00805567 Z" })
    )
  )
));
const FastBackwardIcon = defineComponent({
  name: "FastBackward",
  render() {
    return h(
      "svg",
      { viewBox: "0 0 20 20", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
      h(
        "g",
        { stroke: "none", "stroke-width": "1", fill: "none", "fill-rule": "evenodd" },
        h(
          "g",
          { fill: "currentColor", "fill-rule": "nonzero" },
          h("path", { d: "M8.73171,16.7949 C9.03264,17.0795 9.50733,17.0663 9.79196,16.7654 C10.0766,16.4644 10.0634,15.9897 9.76243,15.7051 L4.52339,10.75 L17.2471,10.75 C17.6613,10.75 17.9971,10.4142 17.9971,10 C17.9971,9.58579 17.6613,9.25 17.2471,9.25 L4.52112,9.25 L9.76243,4.29275 C10.0634,4.00812 10.0766,3.53343 9.79196,3.2325 C9.50733,2.93156 9.03264,2.91834 8.73171,3.20297 L2.31449,9.27241 C2.14819,9.4297 2.04819,9.62981 2.01448,9.8386 C2.00308,9.89058 1.99707,9.94459 1.99707,10 C1.99707,10.0576 2.00356,10.1137 2.01585,10.1675 C2.05084,10.3733 2.15039,10.5702 2.31449,10.7254 L8.73171,16.7949 Z" })
        )
      )
    );
  }
});
const FastForwardIcon = defineComponent({
  name: "FastForward",
  render() {
    return h(
      "svg",
      { viewBox: "0 0 20 20", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
      h(
        "g",
        { stroke: "none", "stroke-width": "1", fill: "none", "fill-rule": "evenodd" },
        h(
          "g",
          { fill: "currentColor", "fill-rule": "nonzero" },
          h("path", { d: "M11.2654,3.20511 C10.9644,2.92049 10.4897,2.93371 10.2051,3.23464 C9.92049,3.53558 9.93371,4.01027 10.2346,4.29489 L15.4737,9.25 L2.75,9.25 C2.33579,9.25 2,9.58579 2,10.0000012 C2,10.4142 2.33579,10.75 2.75,10.75 L15.476,10.75 L10.2346,15.7073 C9.93371,15.9919 9.92049,16.4666 10.2051,16.7675 C10.4897,17.0684 10.9644,17.0817 11.2654,16.797 L17.6826,10.7276 C17.8489,10.5703 17.9489,10.3702 17.9826,10.1614 C17.994,10.1094 18,10.0554 18,10.0000012 C18,9.94241 17.9935,9.88633 17.9812,9.83246 C17.9462,9.62667 17.8467,9.42976 17.6826,9.27455 L11.2654,3.20511 Z" })
        )
      )
    );
  }
});
const FilterIcon = defineComponent({
  name: "Filter",
  render() {
    return h(
      "svg",
      { viewBox: "0 0 28 28", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
      h(
        "g",
        { stroke: "none", "stroke-width": "1", "fill-rule": "evenodd" },
        h(
          "g",
          { "fill-rule": "nonzero" },
          h("path", { d: "M17,19 C17.5522847,19 18,19.4477153 18,20 C18,20.5522847 17.5522847,21 17,21 L11,21 C10.4477153,21 10,20.5522847 10,20 C10,19.4477153 10.4477153,19 11,19 L17,19 Z M21,13 C21.5522847,13 22,13.4477153 22,14 C22,14.5522847 21.5522847,15 21,15 L7,15 C6.44771525,15 6,14.5522847 6,14 C6,13.4477153 6.44771525,13 7,13 L21,13 Z M24,7 C24.5522847,7 25,7.44771525 25,8 C25,8.55228475 24.5522847,9 24,9 L4,9 C3.44771525,9 3,8.55228475 3,8 C3,7.44771525 3.44771525,7 4,7 L24,7 Z" })
        )
      )
    );
  }
});
const ForwardIcon = defineComponent({
  name: "Forward",
  render() {
    return h(
      "svg",
      { viewBox: "0 0 20 20", fill: "none", xmlns: "http://www.w3.org/2000/svg" },
      h("path", { d: "M7.73271 4.20694C8.03263 3.92125 8.50737 3.93279 8.79306 4.23271L13.7944 9.48318C14.0703 9.77285 14.0703 10.2281 13.7944 10.5178L8.79306 15.7682C8.50737 16.0681 8.03263 16.0797 7.73271 15.794C7.43279 15.5083 7.42125 15.0336 7.70694 14.7336L12.2155 10.0005L7.70694 5.26729C7.42125 4.96737 7.43279 4.49264 7.73271 4.20694Z", fill: "currentColor" })
    );
  }
});
const MoreIcon = defineComponent({
  name: "More",
  render() {
    return h(
      "svg",
      { viewBox: "0 0 16 16", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
      h(
        "g",
        { stroke: "none", "stroke-width": "1", fill: "none", "fill-rule": "evenodd" },
        h(
          "g",
          { fill: "currentColor", "fill-rule": "nonzero" },
          h("path", { d: "M4,7 C4.55228,7 5,7.44772 5,8 C5,8.55229 4.55228,9 4,9 C3.44772,9 3,8.55229 3,8 C3,7.44772 3.44772,7 4,7 Z M8,7 C8.55229,7 9,7.44772 9,8 C9,8.55229 8.55229,9 8,9 C7.44772,9 7,8.55229 7,8 C7,7.44772 7.44772,7 8,7 Z M12,7 C12.5523,7 13,7.44772 13,8 C13,8.55229 12.5523,9 12,9 C11.4477,9 11,8.55229 11,8 C11,7.44772 11.4477,7 12,7 Z" })
        )
      )
    );
  }
});
const CancelIcon = replaceable("cancel", h(
  "svg",
  { viewBox: "0 0 16 16", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
  h(
    "g",
    { stroke: "none", "stroke-width": "1", fill: "none", "fill-rule": "evenodd" },
    h(
      "g",
      { fill: "currentColor", "fill-rule": "nonzero" },
      h("path", { d: "M2.58859116,2.7156945 L2.64644661,2.64644661 C2.82001296,2.47288026 3.08943736,2.45359511 3.2843055,2.58859116 L3.35355339,2.64644661 L8,7.293 L12.6464466,2.64644661 C12.8417088,2.45118446 13.1582912,2.45118446 13.3535534,2.64644661 C13.5488155,2.84170876 13.5488155,3.15829124 13.3535534,3.35355339 L8.707,8 L13.3535534,12.6464466 C13.5271197,12.820013 13.5464049,13.0894374 13.4114088,13.2843055 L13.3535534,13.3535534 C13.179987,13.5271197 12.9105626,13.5464049 12.7156945,13.4114088 L12.6464466,13.3535534 L8,8.707 L3.35355339,13.3535534 C3.15829124,13.5488155 2.84170876,13.5488155 2.64644661,13.3535534 C2.45118446,13.1582912 2.45118446,12.8417088 2.64644661,12.6464466 L7.293,8 L2.64644661,3.35355339 C2.47288026,3.17998704 2.45359511,2.91056264 2.58859116,2.7156945 L2.64644661,2.64644661 L2.58859116,2.7156945 Z" })
    )
  )
));
const RetryIcon = replaceable("retry", h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 512 512" },
  h("path", { d: "M320,146s24.36-12-64-12A160,160,0,1,0,416,294", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-miterlimit: 10; stroke-width: 32px;" }),
  h("polyline", { points: "256 58 336 138 256 218", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 32px;" })
));
const CheckMark = h(
  "svg",
  { viewBox: "0 0 64 64", class: "check-icon" },
  h("path", { d: "M50.42,16.76L22.34,39.45l-8.1-11.46c-1.12-1.58-3.3-1.96-4.88-0.84c-1.58,1.12-1.95,3.3-0.84,4.88l10.26,14.51  c0.56,0.79,1.42,1.31,2.38,1.45c0.16,0.02,0.32,0.03,0.48,0.03c0.8,0,1.57-0.27,2.2-0.78l30.99-25.03c1.5-1.21,1.74-3.42,0.52-4.92  C54.13,15.78,51.93,15.55,50.42,16.76z" })
);
const LineMark = h(
  "svg",
  { viewBox: "0 0 100 100", class: "line-icon" },
  h("path", { d: "M80.2,55.5H21.4c-2.8,0-5.1-2.5-5.1-5.5l0,0c0-3,2.3-5.5,5.1-5.5h58.7c2.8,0,5.1,2.5,5.1,5.5l0,0C85.2,53.1,82.9,55.5,80.2,55.5z" })
);
const checkboxGroupInjectionKey = createInjectionKey("n-checkbox-group");
const checkboxGroupProps = {
  min: Number,
  max: Number,
  size: String,
  value: Array,
  defaultValue: {
    type: Array,
    default: null
  },
  disabled: {
    type: Boolean,
    default: void 0
  },
  "onUpdate:value": [Function, Array],
  onUpdateValue: [Function, Array],
  // deprecated
  onChange: [Function, Array]
};
const NCheckboxGroup = defineComponent({
  name: "CheckboxGroup",
  props: checkboxGroupProps,
  setup(props) {
    const { mergedClsPrefixRef } = useConfig(props);
    const formItem = useFormItem(props);
    const { mergedSizeRef, mergedDisabledRef } = formItem;
    const uncontrolledValueRef = ref(props.defaultValue);
    const controlledValueRef = computed(() => props.value);
    const mergedValueRef = useMergedState(controlledValueRef, uncontrolledValueRef);
    const checkedCount = computed(() => {
      var _a;
      return ((_a = mergedValueRef.value) === null || _a === void 0 ? void 0 : _a.length) || 0;
    });
    const valueSetRef = computed(() => {
      if (Array.isArray(mergedValueRef.value)) {
        return new Set(mergedValueRef.value);
      }
      return /* @__PURE__ */ new Set();
    });
    function toggleCheckbox(checked, checkboxValue) {
      const { nTriggerFormInput, nTriggerFormChange } = formItem;
      const { onChange, "onUpdate:value": _onUpdateValue, onUpdateValue } = props;
      if (Array.isArray(mergedValueRef.value)) {
        const groupValue = Array.from(mergedValueRef.value);
        const index = groupValue.findIndex((value) => value === checkboxValue);
        if (checked) {
          if (!~index) {
            groupValue.push(checkboxValue);
            if (onUpdateValue) {
              call(onUpdateValue, groupValue, {
                actionType: "check",
                value: checkboxValue
              });
            }
            if (_onUpdateValue) {
              call(_onUpdateValue, groupValue, {
                actionType: "check",
                value: checkboxValue
              });
            }
            nTriggerFormInput();
            nTriggerFormChange();
            uncontrolledValueRef.value = groupValue;
            if (onChange)
              call(onChange, groupValue);
          }
        } else {
          if (~index) {
            groupValue.splice(index, 1);
            if (onUpdateValue) {
              call(onUpdateValue, groupValue, {
                actionType: "uncheck",
                value: checkboxValue
              });
            }
            if (_onUpdateValue) {
              call(_onUpdateValue, groupValue, {
                actionType: "uncheck",
                value: checkboxValue
              });
            }
            if (onChange)
              call(onChange, groupValue);
            uncontrolledValueRef.value = groupValue;
            nTriggerFormInput();
            nTriggerFormChange();
          }
        }
      } else {
        if (checked) {
          if (onUpdateValue) {
            call(onUpdateValue, [checkboxValue], {
              actionType: "check",
              value: checkboxValue
            });
          }
          if (_onUpdateValue) {
            call(_onUpdateValue, [checkboxValue], {
              actionType: "check",
              value: checkboxValue
            });
          }
          if (onChange)
            call(onChange, [checkboxValue]);
          uncontrolledValueRef.value = [checkboxValue];
          nTriggerFormInput();
          nTriggerFormChange();
        } else {
          if (onUpdateValue) {
            call(onUpdateValue, [], {
              actionType: "uncheck",
              value: checkboxValue
            });
          }
          if (_onUpdateValue) {
            call(_onUpdateValue, [], {
              actionType: "uncheck",
              value: checkboxValue
            });
          }
          if (onChange)
            call(onChange, []);
          uncontrolledValueRef.value = [];
          nTriggerFormInput();
          nTriggerFormChange();
        }
      }
    }
    provide(checkboxGroupInjectionKey, {
      checkedCountRef: checkedCount,
      maxRef: toRef(props, "max"),
      minRef: toRef(props, "min"),
      valueSetRef,
      disabledRef: mergedDisabledRef,
      mergedSizeRef,
      toggleCheckbox
    });
    return {
      mergedClsPrefix: mergedClsPrefixRef
    };
  },
  render() {
    return h("div", { class: `${this.mergedClsPrefix}-checkbox-group`, role: "group" }, this.$slots);
  }
});
const style$7 = c([
  cB("checkbox", `
 line-height: var(--n-label-line-height);
 font-size: var(--n-font-size);
 outline: none;
 cursor: pointer;
 display: inline-flex;
 flex-wrap: nowrap;
 align-items: flex-start;
 word-break: break-word;
 --n-merged-color-table: var(--n-color-table);
 `, [c("&:hover", [cB("checkbox-box", [cE("border", {
    border: "var(--n-border-checked)"
  })])]), c("&:focus:not(:active)", [cB("checkbox-box", [cE("border", `
 border: var(--n-border-focus);
 box-shadow: var(--n-box-shadow-focus);
 `)])]), cM("inside-table", [cB("checkbox-box", `
 background-color: var(--n-merged-color-table);
 `)]), cM("checked", [cB("checkbox-box", `
 background-color: var(--n-color-checked);
 `, [cB("checkbox-icon", [
    // if not set width to 100%, safari & old chrome won't display the icon
    c(".check-icon", `
 opacity: 1;
 transform: scale(1);
 `)
  ])])]), cM("indeterminate", [cB("checkbox-box", [cB("checkbox-icon", [c(".check-icon", `
 opacity: 0;
 transform: scale(.5);
 `), c(".line-icon", `
 opacity: 1;
 transform: scale(1);
 `)])])]), cM("checked, indeterminate", [c("&:focus:not(:active)", [cB("checkbox-box", [cE("border", `
 border: var(--n-border-checked);
 box-shadow: var(--n-box-shadow-focus);
 `)])]), cB("checkbox-box", `
 background-color: var(--n-color-checked);
 border-left: 0;
 border-top: 0;
 `, [cE("border", {
    border: "var(--n-border-checked)"
  })])]), cM("disabled", {
    cursor: "not-allowed"
  }, [cM("checked", [cB("checkbox-box", `
 background-color: var(--n-color-disabled-checked);
 `, [cE("border", {
    border: "var(--n-border-disabled-checked)"
  }), cB("checkbox-icon", [c(".check-icon, .line-icon", {
    fill: "var(--n-check-mark-color-disabled-checked)"
  })])])]), cB("checkbox-box", `
 background-color: var(--n-color-disabled);
 `, [cE("border", {
    border: "var(--n-border-disabled)"
  }), cB("checkbox-icon", [c(".check-icon, .line-icon", {
    fill: "var(--n-check-mark-color-disabled)"
  })])]), cE("label", {
    color: "var(--n-text-color-disabled)"
  })]), cB("checkbox-box-wrapper", `
 position: relative;
 width: var(--n-size);
 flex-shrink: 0;
 flex-grow: 0;
 user-select: none;
 -webkit-user-select: none;
 `), cB("checkbox-box", `
 position: absolute;
 left: 0;
 top: 50%;
 transform: translateY(-50%);
 height: var(--n-size);
 width: var(--n-size);
 display: inline-block;
 box-sizing: border-box;
 border-radius: var(--n-border-radius);
 background-color: var(--n-color);
 transition: background-color 0.3s var(--n-bezier);
 `, [cE("border", `
 transition:
 border-color .3s var(--n-bezier),
 box-shadow .3s var(--n-bezier);
 border-radius: inherit;
 position: absolute;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 border: var(--n-border);
 `), cB("checkbox-icon", `
 display: flex;
 align-items: center;
 justify-content: center;
 position: absolute;
 left: 1px;
 right: 1px;
 top: 1px;
 bottom: 1px;
 `, [c(".check-icon, .line-icon", `
 width: 100%;
 fill: var(--n-check-mark-color);
 opacity: 0;
 transform: scale(0.5);
 transform-origin: center;
 transition:
 fill 0.3s var(--n-bezier),
 transform 0.3s var(--n-bezier),
 opacity 0.3s var(--n-bezier),
 border-color 0.3s var(--n-bezier);
 `), iconSwitchTransition({
    left: "1px",
    top: "1px"
  })])]), cE("label", `
 color: var(--n-text-color);
 transition: color .3s var(--n-bezier);
 user-select: none;
 -webkit-user-select: none;
 padding: var(--n-label-padding);
 font-weight: var(--n-label-font-weight);
 `, [c("&:empty", {
    display: "none"
  })])]),
  // modal table header checkbox
  insideModal(cB("checkbox", `
 --n-merged-color-table: var(--n-color-table-modal);
 `)),
  // popover table header checkbox
  insidePopover(cB("checkbox", `
 --n-merged-color-table: var(--n-color-table-popover);
 `))
]);
const checkboxProps = Object.assign(Object.assign({}, useTheme.props), {
  size: String,
  checked: {
    type: [Boolean, String, Number],
    default: void 0
  },
  defaultChecked: {
    type: [Boolean, String, Number],
    default: false
  },
  value: [String, Number],
  disabled: {
    type: Boolean,
    default: void 0
  },
  indeterminate: Boolean,
  label: String,
  focusable: {
    type: Boolean,
    default: true
  },
  checkedValue: {
    type: [Boolean, String, Number],
    default: true
  },
  uncheckedValue: {
    type: [Boolean, String, Number],
    default: false
  },
  "onUpdate:checked": [Function, Array],
  onUpdateChecked: [Function, Array],
  // private
  privateInsideTable: Boolean,
  // deprecated
  onChange: [Function, Array]
});
const NCheckbox = defineComponent({
  name: "Checkbox",
  props: checkboxProps,
  setup(props) {
    const selfRef = ref(null);
    const { mergedClsPrefixRef, inlineThemeDisabled, mergedRtlRef } = useConfig(props);
    const formItem = useFormItem(props, {
      mergedSize(NFormItem) {
        const { size } = props;
        if (size !== void 0)
          return size;
        if (NCheckboxGroup2) {
          const { value: mergedSize } = NCheckboxGroup2.mergedSizeRef;
          if (mergedSize !== void 0) {
            return mergedSize;
          }
        }
        if (NFormItem) {
          const { mergedSize } = NFormItem;
          if (mergedSize !== void 0)
            return mergedSize.value;
        }
        return "medium";
      },
      mergedDisabled(NFormItem) {
        const { disabled } = props;
        if (disabled !== void 0)
          return disabled;
        if (NCheckboxGroup2) {
          if (NCheckboxGroup2.disabledRef.value)
            return true;
          const { maxRef: { value: max }, checkedCountRef } = NCheckboxGroup2;
          if (max !== void 0 && checkedCountRef.value >= max && !renderedCheckedRef.value) {
            return true;
          }
          const { minRef: { value: min } } = NCheckboxGroup2;
          if (min !== void 0 && checkedCountRef.value <= min && renderedCheckedRef.value) {
            return true;
          }
        }
        if (NFormItem) {
          return NFormItem.disabled.value;
        }
        return false;
      }
    });
    const { mergedDisabledRef, mergedSizeRef } = formItem;
    const NCheckboxGroup2 = inject(checkboxGroupInjectionKey, null);
    const uncontrolledCheckedRef = ref(props.defaultChecked);
    const controlledCheckedRef = toRef(props, "checked");
    const mergedCheckedRef = useMergedState(controlledCheckedRef, uncontrolledCheckedRef);
    const renderedCheckedRef = useMemo(() => {
      if (NCheckboxGroup2) {
        const groupValueSet = NCheckboxGroup2.valueSetRef.value;
        if (groupValueSet && props.value !== void 0) {
          return groupValueSet.has(props.value);
        }
        return false;
      } else {
        return mergedCheckedRef.value === props.checkedValue;
      }
    });
    const themeRef = useTheme("Checkbox", "-checkbox", style$7, checkboxLight, props, mergedClsPrefixRef);
    function toggle(e) {
      if (NCheckboxGroup2 && props.value !== void 0) {
        NCheckboxGroup2.toggleCheckbox(!renderedCheckedRef.value, props.value);
      } else {
        const { onChange, "onUpdate:checked": _onUpdateCheck, onUpdateChecked } = props;
        const { nTriggerFormInput, nTriggerFormChange } = formItem;
        const nextChecked = renderedCheckedRef.value ? props.uncheckedValue : props.checkedValue;
        if (_onUpdateCheck) {
          call(_onUpdateCheck, nextChecked, e);
        }
        if (onUpdateChecked) {
          call(onUpdateChecked, nextChecked, e);
        }
        if (onChange)
          call(onChange, nextChecked, e);
        nTriggerFormInput();
        nTriggerFormChange();
        uncontrolledCheckedRef.value = nextChecked;
      }
    }
    function handleClick(e) {
      if (!mergedDisabledRef.value) {
        toggle(e);
      }
    }
    function handleKeyUp(e) {
      if (mergedDisabledRef.value)
        return;
      switch (e.key) {
        case " ":
        case "Enter":
          toggle(e);
      }
    }
    function handleKeyDown(e) {
      switch (e.key) {
        case " ":
          e.preventDefault();
      }
    }
    const exposedMethods = {
      focus: () => {
        var _a;
        (_a = selfRef.value) === null || _a === void 0 ? void 0 : _a.focus();
      },
      blur: () => {
        var _a;
        (_a = selfRef.value) === null || _a === void 0 ? void 0 : _a.blur();
      }
    };
    const rtlEnabledRef = useRtl("Checkbox", mergedRtlRef, mergedClsPrefixRef);
    const cssVarsRef = computed(() => {
      const { value: mergedSize } = mergedSizeRef;
      const { common: { cubicBezierEaseInOut }, self: { borderRadius, color, colorChecked, colorDisabled, colorTableHeader, colorTableHeaderModal, colorTableHeaderPopover, checkMarkColor, checkMarkColorDisabled, border, borderFocus, borderDisabled, borderChecked, boxShadowFocus, textColor, textColorDisabled, checkMarkColorDisabledChecked, colorDisabledChecked, borderDisabledChecked, labelPadding, labelLineHeight, labelFontWeight, [createKey("fontSize", mergedSize)]: fontSize, [createKey("size", mergedSize)]: size } } = themeRef.value;
      return {
        "--n-label-line-height": labelLineHeight,
        "--n-label-font-weight": labelFontWeight,
        "--n-size": size,
        "--n-bezier": cubicBezierEaseInOut,
        "--n-border-radius": borderRadius,
        "--n-border": border,
        "--n-border-checked": borderChecked,
        "--n-border-focus": borderFocus,
        "--n-border-disabled": borderDisabled,
        "--n-border-disabled-checked": borderDisabledChecked,
        "--n-box-shadow-focus": boxShadowFocus,
        "--n-color": color,
        "--n-color-checked": colorChecked,
        "--n-color-table": colorTableHeader,
        "--n-color-table-modal": colorTableHeaderModal,
        "--n-color-table-popover": colorTableHeaderPopover,
        "--n-color-disabled": colorDisabled,
        "--n-color-disabled-checked": colorDisabledChecked,
        "--n-text-color": textColor,
        "--n-text-color-disabled": textColorDisabled,
        "--n-check-mark-color": checkMarkColor,
        "--n-check-mark-color-disabled": checkMarkColorDisabled,
        "--n-check-mark-color-disabled-checked": checkMarkColorDisabledChecked,
        "--n-font-size": fontSize,
        "--n-label-padding": labelPadding
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("checkbox", computed(() => mergedSizeRef.value[0]), cssVarsRef, props) : void 0;
    return Object.assign(formItem, exposedMethods, {
      rtlEnabled: rtlEnabledRef,
      selfRef,
      mergedClsPrefix: mergedClsPrefixRef,
      mergedDisabled: mergedDisabledRef,
      renderedChecked: renderedCheckedRef,
      mergedTheme: themeRef,
      labelId: createId(),
      handleClick,
      handleKeyUp,
      handleKeyDown,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    });
  },
  render() {
    var _a;
    const { $slots, renderedChecked, mergedDisabled, indeterminate, privateInsideTable, cssVars, labelId, label, mergedClsPrefix, focusable, handleKeyUp, handleKeyDown, handleClick } = this;
    (_a = this.onRender) === null || _a === void 0 ? void 0 : _a.call(this);
    return h(
      "div",
      { ref: "selfRef", class: [
        `${mergedClsPrefix}-checkbox`,
        this.themeClass,
        this.rtlEnabled && `${mergedClsPrefix}-checkbox--rtl`,
        renderedChecked && `${mergedClsPrefix}-checkbox--checked`,
        mergedDisabled && `${mergedClsPrefix}-checkbox--disabled`,
        indeterminate && `${mergedClsPrefix}-checkbox--indeterminate`,
        privateInsideTable && `${mergedClsPrefix}-checkbox--inside-table`
      ], tabindex: mergedDisabled || !focusable ? void 0 : 0, role: "checkbox", "aria-checked": indeterminate ? "mixed" : renderedChecked, "aria-labelledby": labelId, style: cssVars, onKeyup: handleKeyUp, onKeydown: handleKeyDown, onClick: handleClick, onMousedown: () => {
        on("selectstart", window, (e) => {
          e.preventDefault();
        }, {
          once: true
        });
      } },
      h(
        "div",
        { class: `${mergedClsPrefix}-checkbox-box-wrapper` },
        " ",
        h(
          "div",
          { class: `${mergedClsPrefix}-checkbox-box` },
          h(NIconSwitchTransition, null, {
            default: () => this.indeterminate ? h("div", { key: "indeterminate", class: `${mergedClsPrefix}-checkbox-icon` }, LineMark) : h("div", { key: "check", class: `${mergedClsPrefix}-checkbox-icon` }, CheckMark)
          }),
          h("div", { class: `${mergedClsPrefix}-checkbox-box__border` })
        )
      ),
      label !== null || $slots.default ? h("span", { class: `${mergedClsPrefix}-checkbox__label`, id: labelId }, $slots.default ? $slots.default() : label) : null
    );
  }
});
const popselectInjectionKey = createInjectionKey("n-popselect");
const style$6 = cB("popselect-menu", `
 box-shadow: var(--n-menu-box-shadow);
`);
const panelProps = {
  multiple: Boolean,
  value: {
    type: [String, Number, Array],
    default: null
  },
  cancelable: Boolean,
  options: {
    type: Array,
    default: () => []
  },
  size: {
    type: String,
    default: "medium"
  },
  scrollable: Boolean,
  "onUpdate:value": [Function, Array],
  onUpdateValue: [Function, Array],
  onMouseenter: Function,
  onMouseleave: Function,
  renderLabel: Function,
  showCheckmark: {
    type: Boolean,
    default: void 0
  },
  nodeProps: Function,
  virtualScroll: Boolean,
  // deprecated
  onChange: [Function, Array]
};
const panelPropKeys = keysOf(panelProps);
const NPopselectPanel = defineComponent({
  name: "PopselectPanel",
  props: panelProps,
  setup(props) {
    const NPopselect2 = inject(popselectInjectionKey);
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const themeRef = useTheme("Popselect", "-pop-select", style$6, popselectLight, NPopselect2.props, mergedClsPrefixRef);
    const treeMateRef = computed(() => {
      return createTreeMate(props.options, createTmOptions("value", "children"));
    });
    function doUpdateValue(value, option) {
      const { onUpdateValue, "onUpdate:value": _onUpdateValue, onChange } = props;
      if (onUpdateValue)
        call(onUpdateValue, value, option);
      if (_onUpdateValue) {
        call(_onUpdateValue, value, option);
      }
      if (onChange)
        call(onChange, value, option);
    }
    function handleToggle(tmNode) {
      toggle(tmNode.key);
    }
    function handleMenuMousedown(e) {
      if (!happensIn(e, "action"))
        e.preventDefault();
    }
    function toggle(value) {
      const { value: { getNode } } = treeMateRef;
      if (props.multiple) {
        if (Array.isArray(props.value)) {
          const newValue = [];
          const newOptions = [];
          let shouldAddValue = true;
          props.value.forEach((v) => {
            if (v === value) {
              shouldAddValue = false;
              return;
            }
            const tmNode = getNode(v);
            if (tmNode) {
              newValue.push(tmNode.key);
              newOptions.push(tmNode.rawNode);
            }
          });
          if (shouldAddValue) {
            newValue.push(value);
            newOptions.push(getNode(value).rawNode);
          }
          doUpdateValue(newValue, newOptions);
        } else {
          const tmNode = getNode(value);
          if (tmNode) {
            doUpdateValue([value], [tmNode.rawNode]);
          }
        }
      } else {
        if (props.value === value && props.cancelable) {
          doUpdateValue(null, null);
        } else {
          const tmNode = getNode(value);
          if (tmNode) {
            doUpdateValue(value, tmNode.rawNode);
          }
          const { "onUpdate:show": _onUpdateShow, onUpdateShow } = NPopselect2.props;
          if (_onUpdateShow)
            call(_onUpdateShow, false);
          if (onUpdateShow)
            call(onUpdateShow, false);
          NPopselect2.setShow(false);
        }
      }
      void nextTick(() => {
        NPopselect2.syncPosition();
      });
    }
    watch(toRef(props, "options"), () => {
      void nextTick(() => {
        NPopselect2.syncPosition();
      });
    });
    const cssVarsRef = computed(() => {
      const { self: { menuBoxShadow } } = themeRef.value;
      return {
        "--n-menu-box-shadow": menuBoxShadow
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("select", void 0, cssVarsRef, NPopselect2.props) : void 0;
    return {
      mergedTheme: NPopselect2.mergedThemeRef,
      mergedClsPrefix: mergedClsPrefixRef,
      treeMate: treeMateRef,
      handleToggle,
      handleMenuMousedown,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    var _a;
    (_a = this.onRender) === null || _a === void 0 ? void 0 : _a.call(this);
    return h(NInternalSelectMenu, { clsPrefix: this.mergedClsPrefix, focusable: true, nodeProps: this.nodeProps, class: [`${this.mergedClsPrefix}-popselect-menu`, this.themeClass], style: this.cssVars, theme: this.mergedTheme.peers.InternalSelectMenu, themeOverrides: this.mergedTheme.peerOverrides.InternalSelectMenu, multiple: this.multiple, treeMate: this.treeMate, size: this.size, value: this.value, virtualScroll: this.virtualScroll, scrollable: this.scrollable, renderLabel: this.renderLabel, onToggle: this.handleToggle, onMouseenter: this.onMouseenter, onMouseleave: this.onMouseenter, onMousedown: this.handleMenuMousedown, showCheckmark: this.showCheckmark }, {
      action: () => {
        var _a2, _b;
        return ((_b = (_a2 = this.$slots).action) === null || _b === void 0 ? void 0 : _b.call(_a2)) || [];
      },
      empty: () => {
        var _a2, _b;
        return ((_b = (_a2 = this.$slots).empty) === null || _b === void 0 ? void 0 : _b.call(_a2)) || [];
      }
    });
  }
});
const popselectProps = Object.assign(Object.assign(Object.assign(Object.assign({}, useTheme.props), omit(popoverBaseProps, ["showArrow", "arrow"])), { placement: Object.assign(Object.assign({}, popoverBaseProps.placement), { default: "bottom" }), trigger: {
  type: String,
  default: "hover"
} }), panelProps);
const NPopselect = defineComponent({
  name: "Popselect",
  props: popselectProps,
  inheritAttrs: false,
  __popover__: true,
  setup(props) {
    const { mergedClsPrefixRef } = useConfig(props);
    const themeRef = useTheme("Popselect", "-popselect", void 0, popselectLight, props, mergedClsPrefixRef);
    const popoverInstRef = ref(null);
    function syncPosition() {
      var _a;
      (_a = popoverInstRef.value) === null || _a === void 0 ? void 0 : _a.syncPosition();
    }
    function setShow(value) {
      var _a;
      (_a = popoverInstRef.value) === null || _a === void 0 ? void 0 : _a.setShow(value);
    }
    provide(popselectInjectionKey, {
      props,
      mergedThemeRef: themeRef,
      syncPosition,
      setShow
    });
    const exposedMethods = {
      syncPosition,
      setShow
    };
    return Object.assign(Object.assign({}, exposedMethods), { popoverInstRef, mergedTheme: themeRef });
  },
  render() {
    const { mergedTheme } = this;
    const popoverProps = {
      theme: mergedTheme.peers.Popover,
      themeOverrides: mergedTheme.peerOverrides.Popover,
      builtinThemeOverrides: {
        padding: "0"
      },
      ref: "popoverInstRef",
      internalRenderBody: (className, ref2, style2, onMouseenter, onMouseleave) => {
        const { $attrs } = this;
        return h(NPopselectPanel, Object.assign({}, $attrs, { class: [$attrs.class, className], style: [$attrs.style, style2] }, keep(this.$props, panelPropKeys), { ref: createRefSetter(ref2), onMouseenter: mergeEventHandlers([
          onMouseenter,
          $attrs.onMouseenter
        ]), onMouseleave: mergeEventHandlers([
          onMouseleave,
          $attrs.onMouseleave
        ]) }), {
          action: () => {
            var _a, _b;
            return (_b = (_a = this.$slots).action) === null || _b === void 0 ? void 0 : _b.call(_a);
          },
          empty: () => {
            var _a, _b;
            return (_b = (_a = this.$slots).empty) === null || _b === void 0 ? void 0 : _b.call(_a);
          }
        });
      }
    };
    return h(NPopover, Object.assign({}, omit(this.$props, panelPropKeys), popoverProps, { internalDeactivateImmediately: true }), {
      trigger: () => {
        var _a, _b;
        return (_b = (_a = this.$slots).default) === null || _b === void 0 ? void 0 : _b.call(_a);
      }
    });
  }
});
function createPageItemsInfo(currentPage, pageCount, pageSlot) {
  let hasFastBackward = false;
  let hasFastForward = false;
  let fastBackwardTo = 1;
  let fastForwardTo = pageCount;
  if (pageCount === 1) {
    return {
      hasFastBackward: false,
      hasFastForward: false,
      fastForwardTo,
      fastBackwardTo,
      items: [
        {
          type: "page",
          label: 1,
          active: currentPage === 1,
          mayBeFastBackward: false,
          mayBeFastForward: false
        }
      ]
    };
  }
  if (pageCount === 2) {
    return {
      hasFastBackward: false,
      hasFastForward: false,
      fastForwardTo,
      fastBackwardTo,
      items: [
        {
          type: "page",
          label: 1,
          active: currentPage === 1,
          mayBeFastBackward: false,
          mayBeFastForward: false
        },
        {
          type: "page",
          label: 2,
          active: currentPage === 2,
          mayBeFastBackward: true,
          mayBeFastForward: false
        }
      ]
    };
  }
  const firstPage = 1;
  const lastPage = pageCount;
  let middleStart = currentPage;
  let middleEnd = currentPage;
  const middleDelta = (pageSlot - 5) / 2;
  middleEnd += Math.ceil(middleDelta);
  middleEnd = Math.min(Math.max(middleEnd, firstPage + pageSlot - 3), lastPage - 2);
  middleStart -= Math.floor(middleDelta);
  middleStart = Math.max(Math.min(middleStart, lastPage - pageSlot + 3), firstPage + 2);
  let leftSplit = false;
  let rightSplit = false;
  if (middleStart > firstPage + 2)
    leftSplit = true;
  if (middleEnd < lastPage - 2)
    rightSplit = true;
  const items = [];
  items.push({
    type: "page",
    label: 1,
    active: currentPage === 1,
    mayBeFastBackward: false,
    mayBeFastForward: false
  });
  if (leftSplit) {
    hasFastBackward = true;
    fastBackwardTo = middleStart - 1;
    items.push({
      type: "fast-backward",
      active: false,
      label: void 0,
      options: createRange(firstPage + 1, middleStart - 1)
    });
  } else if (lastPage >= firstPage + 1) {
    items.push({
      type: "page",
      label: firstPage + 1,
      mayBeFastBackward: true,
      mayBeFastForward: false,
      active: currentPage === firstPage + 1
    });
  }
  for (let i = middleStart; i <= middleEnd; ++i) {
    items.push({
      type: "page",
      label: i,
      mayBeFastBackward: false,
      mayBeFastForward: false,
      active: currentPage === i
    });
  }
  if (rightSplit) {
    hasFastForward = true;
    fastForwardTo = middleEnd + 1;
    items.push({
      type: "fast-forward",
      active: false,
      label: void 0,
      options: createRange(middleEnd + 1, lastPage - 1)
    });
  } else if (middleEnd === lastPage - 2 && items[items.length - 1].label !== lastPage - 1) {
    items.push({
      type: "page",
      mayBeFastForward: true,
      mayBeFastBackward: false,
      label: lastPage - 1,
      active: currentPage === lastPage - 1
    });
  }
  if (items[items.length - 1].label !== lastPage) {
    items.push({
      type: "page",
      mayBeFastForward: false,
      mayBeFastBackward: false,
      label: lastPage,
      active: currentPage === lastPage
    });
  }
  return {
    hasFastBackward,
    hasFastForward,
    fastBackwardTo,
    fastForwardTo,
    items
  };
}
function createRange(from, to) {
  const range = [];
  for (let i = from; i <= to; ++i) {
    range.push({
      label: `${i}`,
      value: i
    });
  }
  return range;
}
const hoverStyleProps = `
 background: var(--n-item-color-hover);
 color: var(--n-item-text-color-hover);
 border: var(--n-item-border-hover);
`;
const hoverStyleChildren = [cM("button", `
 background: var(--n-button-color-hover);
 border: var(--n-button-border-hover);
 color: var(--n-button-icon-color-hover);
 `)];
const style$5 = cB("pagination", `
 display: flex;
 vertical-align: middle;
 font-size: var(--n-item-font-size);
 flex-wrap: nowrap;
`, [cB("pagination-prefix", `
 display: flex;
 align-items: center;
 margin: var(--n-prefix-margin);
 `), cB("pagination-suffix", `
 display: flex;
 align-items: center;
 margin: var(--n-suffix-margin);
 `), c("> *:not(:first-child)", `
 margin: var(--n-item-margin);
 `), cB("select", `
 width: var(--n-select-width);
 `), c("&.transition-disabled", [cB("pagination-item", "transition: none!important;")]), cB("pagination-quick-jumper", `
 white-space: nowrap;
 display: flex;
 color: var(--n-jumper-text-color);
 transition: color .3s var(--n-bezier);
 align-items: center;
 font-size: var(--n-jumper-font-size);
 `, [cB("input", `
 margin: var(--n-input-margin);
 width: var(--n-input-width);
 `)]), cB("pagination-item", `
 position: relative;
 cursor: pointer;
 user-select: none;
 -webkit-user-select: none;
 display: flex;
 align-items: center;
 justify-content: center;
 box-sizing: border-box;
 min-width: var(--n-item-size);
 height: var(--n-item-size);
 padding: var(--n-item-padding);
 background-color: var(--n-item-color);
 color: var(--n-item-text-color);
 border-radius: var(--n-item-border-radius);
 border: var(--n-item-border);
 fill: var(--n-button-icon-color);
 transition:
 color .3s var(--n-bezier),
 border-color .3s var(--n-bezier),
 background-color .3s var(--n-bezier),
 fill .3s var(--n-bezier);
 `, [cM("button", `
 background: var(--n-button-color);
 color: var(--n-button-icon-color);
 border: var(--n-button-border);
 padding: 0;
 `, [cB("base-icon", `
 font-size: var(--n-button-icon-size);
 `)]), cNotM("disabled", [cM("hover", hoverStyleProps, hoverStyleChildren), c("&:hover", hoverStyleProps, hoverStyleChildren), c("&:active", `
 background: var(--n-item-color-pressed);
 color: var(--n-item-text-color-pressed);
 border: var(--n-item-border-pressed);
 `, [cM("button", `
 background: var(--n-button-color-pressed);
 border: var(--n-button-border-pressed);
 color: var(--n-button-icon-color-pressed);
 `)]), cM("active", `
 background: var(--n-item-color-active);
 color: var(--n-item-text-color-active);
 border: var(--n-item-border-active);
 `, [c("&:hover", `
 background: var(--n-item-color-active-hover);
 `)])]), cM("disabled", `
 cursor: not-allowed;
 color: var(--n-item-text-color-disabled);
 `, [cM("active, button", `
 background-color: var(--n-item-color-disabled);
 border: var(--n-item-border-disabled);
 `)])]), cM("disabled", `
 cursor: not-allowed;
 `, [cB("pagination-quick-jumper", `
 color: var(--n-jumper-text-color-disabled);
 `)]), cM("simple", `
 display: flex;
 align-items: center;
 flex-wrap: nowrap;
 `, [cB("pagination-quick-jumper", [cB("input", `
 margin: 0;
 `)])])]);
const paginationProps = Object.assign(Object.assign({}, useTheme.props), {
  simple: Boolean,
  page: Number,
  defaultPage: {
    type: Number,
    default: 1
  },
  itemCount: Number,
  pageCount: Number,
  defaultPageCount: {
    type: Number,
    default: 1
  },
  showSizePicker: Boolean,
  pageSize: Number,
  defaultPageSize: Number,
  pageSizes: {
    type: Array,
    default() {
      return [10];
    }
  },
  showQuickJumper: Boolean,
  size: {
    type: String,
    default: "medium"
  },
  disabled: Boolean,
  pageSlot: {
    type: Number,
    default: 9
  },
  selectProps: Object,
  prev: Function,
  next: Function,
  goto: Function,
  prefix: Function,
  suffix: Function,
  label: Function,
  displayOrder: {
    type: Array,
    default: ["pages", "size-picker", "quick-jumper"]
  },
  to: useAdjustedTo.propTo,
  "onUpdate:page": [Function, Array],
  onUpdatePage: [Function, Array],
  "onUpdate:pageSize": [Function, Array],
  onUpdatePageSize: [Function, Array],
  /** @deprecated */
  onPageSizeChange: [Function, Array],
  /** @deprecated */
  onChange: [Function, Array]
});
const NPagination = defineComponent({
  name: "Pagination",
  props: paginationProps,
  setup(props) {
    const { mergedComponentPropsRef, mergedClsPrefixRef, inlineThemeDisabled, mergedRtlRef } = useConfig(props);
    const themeRef = useTheme("Pagination", "-pagination", style$5, paginationLight, props, mergedClsPrefixRef);
    const { localeRef } = useLocale("Pagination");
    const selfRef = ref(null);
    const uncontrolledPageRef = ref(props.defaultPage);
    const getDefaultPageSize = () => {
      const { defaultPageSize } = props;
      if (defaultPageSize !== void 0)
        return defaultPageSize;
      const pageSizeOption = props.pageSizes[0];
      if (typeof pageSizeOption === "number")
        return pageSizeOption;
      return pageSizeOption.value || 10;
    };
    const uncontrolledPageSizeRef = ref(getDefaultPageSize());
    const mergedPageRef = useMergedState(toRef(props, "page"), uncontrolledPageRef);
    const mergedPageSizeRef = useMergedState(toRef(props, "pageSize"), uncontrolledPageSizeRef);
    const mergedPageCountRef = computed(() => {
      const { itemCount } = props;
      if (itemCount !== void 0) {
        return Math.max(1, Math.ceil(itemCount / mergedPageSizeRef.value));
      }
      const { pageCount } = props;
      if (pageCount !== void 0)
        return Math.max(pageCount, 1);
      return 1;
    });
    const jumperValueRef = ref("");
    watchEffect(() => {
      void props.simple;
      jumperValueRef.value = String(mergedPageRef.value);
    });
    const fastForwardActiveRef = ref(false);
    const fastBackwardActiveRef = ref(false);
    const showFastForwardMenuRef = ref(false);
    const showFastBackwardMenuRef = ref(false);
    const handleFastForwardMouseenter = () => {
      if (props.disabled)
        return;
      fastForwardActiveRef.value = true;
      disableTransitionOneTick();
    };
    const handleFastForwardMouseleave = () => {
      if (props.disabled)
        return;
      fastForwardActiveRef.value = false;
      disableTransitionOneTick();
    };
    const handleFastBackwardMouseenter = () => {
      fastBackwardActiveRef.value = true;
      disableTransitionOneTick();
    };
    const handleFastBackwardMouseleave = () => {
      fastBackwardActiveRef.value = false;
      disableTransitionOneTick();
    };
    const handleMenuSelect = (value) => {
      doUpdatePage(value);
    };
    const pageItemsInfo = computed(() => createPageItemsInfo(mergedPageRef.value, mergedPageCountRef.value, props.pageSlot));
    watchEffect(() => {
      if (!pageItemsInfo.value.hasFastBackward) {
        fastBackwardActiveRef.value = false;
        showFastBackwardMenuRef.value = false;
      } else if (!pageItemsInfo.value.hasFastForward) {
        fastForwardActiveRef.value = false;
        showFastForwardMenuRef.value = false;
      }
    });
    const pageSizeOptionsRef = computed(() => {
      const suffix = localeRef.value.selectionSuffix;
      return props.pageSizes.map((size) => {
        if (typeof size === "number") {
          return {
            label: `${size} / ${suffix}`,
            value: size
          };
        } else {
          return size;
        }
      });
    });
    const inputSizeRef = computed(() => {
      var _a, _b;
      return ((_b = (_a = mergedComponentPropsRef === null || mergedComponentPropsRef === void 0 ? void 0 : mergedComponentPropsRef.value) === null || _a === void 0 ? void 0 : _a.Pagination) === null || _b === void 0 ? void 0 : _b.inputSize) || smallerSize(props.size);
    });
    const selectSizeRef = computed(() => {
      var _a, _b;
      return ((_b = (_a = mergedComponentPropsRef === null || mergedComponentPropsRef === void 0 ? void 0 : mergedComponentPropsRef.value) === null || _a === void 0 ? void 0 : _a.Pagination) === null || _b === void 0 ? void 0 : _b.selectSize) || smallerSize(props.size);
    });
    const startIndexRef = computed(() => {
      return (mergedPageRef.value - 1) * mergedPageSizeRef.value;
    });
    const endIndexRef = computed(() => {
      const endIndex = mergedPageRef.value * mergedPageSizeRef.value - 1;
      const { itemCount } = props;
      if (itemCount !== void 0) {
        return endIndex > itemCount - 1 ? itemCount - 1 : endIndex;
      }
      return endIndex;
    });
    const mergedItemCountRef = computed(() => {
      const { itemCount } = props;
      if (itemCount !== void 0)
        return itemCount;
      return (props.pageCount || 1) * mergedPageSizeRef.value;
    });
    const rtlEnabledRef = useRtl("Pagination", mergedRtlRef, mergedClsPrefixRef);
    const disableTransitionOneTick = () => {
      void nextTick(() => {
        var _a;
        const { value: selfEl } = selfRef;
        if (!selfEl)
          return;
        selfEl.classList.add("transition-disabled");
        void ((_a = selfRef.value) === null || _a === void 0 ? void 0 : _a.offsetWidth);
        selfEl.classList.remove("transition-disabled");
      });
    };
    function doUpdatePage(page) {
      if (page === mergedPageRef.value)
        return;
      const { "onUpdate:page": _onUpdatePage, onUpdatePage, onChange, simple } = props;
      if (_onUpdatePage)
        call(_onUpdatePage, page);
      if (onUpdatePage)
        call(onUpdatePage, page);
      if (onChange)
        call(onChange, page);
      uncontrolledPageRef.value = page;
      if (simple) {
        jumperValueRef.value = String(page);
      }
    }
    function doUpdatePageSize(pageSize) {
      if (pageSize === mergedPageSizeRef.value)
        return;
      const { "onUpdate:pageSize": _onUpdatePageSize, onUpdatePageSize, onPageSizeChange } = props;
      if (_onUpdatePageSize)
        call(_onUpdatePageSize, pageSize);
      if (onUpdatePageSize)
        call(onUpdatePageSize, pageSize);
      if (onPageSizeChange)
        call(onPageSizeChange, pageSize);
      uncontrolledPageSizeRef.value = pageSize;
      if (mergedPageCountRef.value < mergedPageRef.value) {
        doUpdatePage(mergedPageCountRef.value);
      }
    }
    function forward() {
      if (props.disabled)
        return;
      const page = Math.min(mergedPageRef.value + 1, mergedPageCountRef.value);
      doUpdatePage(page);
    }
    function backward() {
      if (props.disabled)
        return;
      const page = Math.max(mergedPageRef.value - 1, 1);
      doUpdatePage(page);
    }
    function fastForward() {
      if (props.disabled)
        return;
      const page = Math.min(pageItemsInfo.value.fastForwardTo, mergedPageCountRef.value);
      doUpdatePage(page);
    }
    function fastBackward() {
      if (props.disabled)
        return;
      const page = Math.max(pageItemsInfo.value.fastBackwardTo, 1);
      doUpdatePage(page);
    }
    function handleSizePickerChange(value) {
      doUpdatePageSize(value);
    }
    function doQuickJump() {
      const page = parseInt(jumperValueRef.value);
      if (Number.isNaN(page))
        return;
      doUpdatePage(Math.max(1, Math.min(page, mergedPageCountRef.value)));
      if (!props.simple) {
        jumperValueRef.value = "";
      }
    }
    function handleQuickJumperChange() {
      doQuickJump();
    }
    function handlePageItemClick(pageItem) {
      if (props.disabled)
        return;
      switch (pageItem.type) {
        case "page":
          doUpdatePage(pageItem.label);
          break;
        case "fast-backward":
          fastBackward();
          break;
        case "fast-forward":
          fastForward();
          break;
      }
    }
    function handleJumperInput(value) {
      jumperValueRef.value = value.replace(/\D+/g, "");
    }
    watchEffect(() => {
      void mergedPageRef.value;
      void mergedPageSizeRef.value;
      disableTransitionOneTick();
    });
    const cssVarsRef = computed(() => {
      const { size } = props;
      const { self: { buttonBorder, buttonBorderHover, buttonBorderPressed, buttonIconColor, buttonIconColorHover, buttonIconColorPressed, itemTextColor, itemTextColorHover, itemTextColorPressed, itemTextColorActive, itemTextColorDisabled, itemColor, itemColorHover, itemColorPressed, itemColorActive, itemColorActiveHover, itemColorDisabled, itemBorder, itemBorderHover, itemBorderPressed, itemBorderActive, itemBorderDisabled, itemBorderRadius, jumperTextColor, jumperTextColorDisabled, buttonColor, buttonColorHover, buttonColorPressed, [createKey("itemPadding", size)]: itemPadding, [createKey("itemMargin", size)]: itemMargin, [createKey("inputWidth", size)]: inputWidth, [createKey("selectWidth", size)]: selectWidth, [createKey("inputMargin", size)]: inputMargin, [createKey("selectMargin", size)]: selectMargin, [createKey("jumperFontSize", size)]: jumperFontSize, [createKey("prefixMargin", size)]: prefixMargin, [createKey("suffixMargin", size)]: suffixMargin, [createKey("itemSize", size)]: itemSize, [createKey("buttonIconSize", size)]: buttonIconSize, [createKey("itemFontSize", size)]: itemFontSize, [`${createKey("itemMargin", size)}Rtl`]: itemMarginRtl, [`${createKey("inputMargin", size)}Rtl`]: inputMarginRtl }, common: { cubicBezierEaseInOut } } = themeRef.value;
      return {
        "--n-prefix-margin": prefixMargin,
        "--n-suffix-margin": suffixMargin,
        "--n-item-font-size": itemFontSize,
        "--n-select-width": selectWidth,
        "--n-select-margin": selectMargin,
        "--n-input-width": inputWidth,
        "--n-input-margin": inputMargin,
        "--n-input-margin-rtl": inputMarginRtl,
        "--n-item-size": itemSize,
        "--n-item-text-color": itemTextColor,
        "--n-item-text-color-disabled": itemTextColorDisabled,
        "--n-item-text-color-hover": itemTextColorHover,
        "--n-item-text-color-active": itemTextColorActive,
        "--n-item-text-color-pressed": itemTextColorPressed,
        "--n-item-color": itemColor,
        "--n-item-color-hover": itemColorHover,
        "--n-item-color-disabled": itemColorDisabled,
        "--n-item-color-active": itemColorActive,
        "--n-item-color-active-hover": itemColorActiveHover,
        "--n-item-color-pressed": itemColorPressed,
        "--n-item-border": itemBorder,
        "--n-item-border-hover": itemBorderHover,
        "--n-item-border-disabled": itemBorderDisabled,
        "--n-item-border-active": itemBorderActive,
        "--n-item-border-pressed": itemBorderPressed,
        "--n-item-padding": itemPadding,
        "--n-item-border-radius": itemBorderRadius,
        "--n-bezier": cubicBezierEaseInOut,
        "--n-jumper-font-size": jumperFontSize,
        "--n-jumper-text-color": jumperTextColor,
        "--n-jumper-text-color-disabled": jumperTextColorDisabled,
        "--n-item-margin": itemMargin,
        "--n-item-margin-rtl": itemMarginRtl,
        "--n-button-icon-size": buttonIconSize,
        "--n-button-icon-color": buttonIconColor,
        "--n-button-icon-color-hover": buttonIconColorHover,
        "--n-button-icon-color-pressed": buttonIconColorPressed,
        "--n-button-color-hover": buttonColorHover,
        "--n-button-color": buttonColor,
        "--n-button-color-pressed": buttonColorPressed,
        "--n-button-border": buttonBorder,
        "--n-button-border-hover": buttonBorderHover,
        "--n-button-border-pressed": buttonBorderPressed
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("pagination", computed(() => {
      let hash = "";
      const { size } = props;
      hash += size[0];
      return hash;
    }), cssVarsRef, props) : void 0;
    return {
      rtlEnabled: rtlEnabledRef,
      mergedClsPrefix: mergedClsPrefixRef,
      locale: localeRef,
      selfRef,
      mergedPage: mergedPageRef,
      pageItems: computed(() => {
        return pageItemsInfo.value.items;
      }),
      mergedItemCount: mergedItemCountRef,
      jumperValue: jumperValueRef,
      pageSizeOptions: pageSizeOptionsRef,
      mergedPageSize: mergedPageSizeRef,
      inputSize: inputSizeRef,
      selectSize: selectSizeRef,
      mergedTheme: themeRef,
      mergedPageCount: mergedPageCountRef,
      startIndex: startIndexRef,
      endIndex: endIndexRef,
      showFastForwardMenu: showFastForwardMenuRef,
      showFastBackwardMenu: showFastBackwardMenuRef,
      fastForwardActive: fastForwardActiveRef,
      fastBackwardActive: fastBackwardActiveRef,
      handleMenuSelect,
      handleFastForwardMouseenter,
      handleFastForwardMouseleave,
      handleFastBackwardMouseenter,
      handleFastBackwardMouseleave,
      handleJumperInput,
      handleBackwardClick: backward,
      handleForwardClick: forward,
      handlePageItemClick,
      handleSizePickerChange,
      handleQuickJumperChange,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    const { $slots, mergedClsPrefix, disabled, cssVars, mergedPage, mergedPageCount, pageItems, showSizePicker, showQuickJumper, mergedTheme, locale, inputSize, selectSize, mergedPageSize, pageSizeOptions, jumperValue, simple, prev, next, prefix, suffix, label, goto, handleJumperInput, handleSizePickerChange, handleBackwardClick, handlePageItemClick, handleForwardClick, handleQuickJumperChange, onRender } = this;
    onRender === null || onRender === void 0 ? void 0 : onRender();
    const renderPrefix = $slots.prefix || prefix;
    const renderSuffix = $slots.suffix || suffix;
    const renderPrev = prev || $slots.prev;
    const renderNext = next || $slots.next;
    const renderLabel = label || $slots.label;
    return h(
      "div",
      { ref: "selfRef", class: [
        `${mergedClsPrefix}-pagination`,
        this.themeClass,
        this.rtlEnabled && `${mergedClsPrefix}-pagination--rtl`,
        disabled && `${mergedClsPrefix}-pagination--disabled`,
        simple && `${mergedClsPrefix}-pagination--simple`
      ], style: cssVars },
      renderPrefix ? h("div", { class: `${mergedClsPrefix}-pagination-prefix` }, renderPrefix({
        page: mergedPage,
        pageSize: mergedPageSize,
        pageCount: mergedPageCount,
        startIndex: this.startIndex,
        endIndex: this.endIndex,
        itemCount: this.mergedItemCount
      })) : null,
      this.displayOrder.map((part) => {
        switch (part) {
          case "pages":
            return h(
              Fragment,
              null,
              h("div", { class: [
                `${mergedClsPrefix}-pagination-item`,
                !renderPrev && `${mergedClsPrefix}-pagination-item--button`,
                (mergedPage <= 1 || mergedPage > mergedPageCount || disabled) && `${mergedClsPrefix}-pagination-item--disabled`
              ], onClick: handleBackwardClick }, renderPrev ? renderPrev({
                page: mergedPage,
                pageSize: mergedPageSize,
                pageCount: mergedPageCount,
                startIndex: this.startIndex,
                endIndex: this.endIndex,
                itemCount: this.mergedItemCount
              }) : h(NBaseIcon, { clsPrefix: mergedClsPrefix }, {
                default: () => this.rtlEnabled ? h(ForwardIcon, null) : h(BackwardIcon, null)
              })),
              simple ? h(
                Fragment,
                null,
                h(
                  "div",
                  { class: `${mergedClsPrefix}-pagination-quick-jumper` },
                  h(NInput, { value: jumperValue, onUpdateValue: handleJumperInput, size: inputSize, placeholder: "", disabled, theme: mergedTheme.peers.Input, themeOverrides: mergedTheme.peerOverrides.Input, onChange: handleQuickJumperChange })
                ),
                " / ",
                mergedPageCount
              ) : pageItems.map((pageItem, index) => {
                let contentNode;
                let onMouseenter;
                let onMouseleave;
                const { type } = pageItem;
                switch (type) {
                  case "page":
                    const pageNode = pageItem.label;
                    if (renderLabel) {
                      contentNode = renderLabel({
                        type: "page",
                        node: pageNode,
                        active: pageItem.active
                      });
                    } else {
                      contentNode = pageNode;
                    }
                    break;
                  case "fast-forward":
                    const fastForwardNode = this.fastForwardActive ? h(NBaseIcon, { clsPrefix: mergedClsPrefix }, {
                      default: () => this.rtlEnabled ? h(FastBackwardIcon, null) : h(FastForwardIcon, null)
                    }) : h(NBaseIcon, { clsPrefix: mergedClsPrefix }, { default: () => h(MoreIcon, null) });
                    if (renderLabel) {
                      contentNode = renderLabel({
                        type: "fast-forward",
                        node: fastForwardNode,
                        active: this.fastForwardActive || this.showFastForwardMenu
                      });
                    } else {
                      contentNode = fastForwardNode;
                    }
                    onMouseenter = this.handleFastForwardMouseenter;
                    onMouseleave = this.handleFastForwardMouseleave;
                    break;
                  case "fast-backward":
                    const fastBackwardNode = this.fastBackwardActive ? h(NBaseIcon, { clsPrefix: mergedClsPrefix }, {
                      default: () => this.rtlEnabled ? h(FastForwardIcon, null) : h(FastBackwardIcon, null)
                    }) : h(NBaseIcon, { clsPrefix: mergedClsPrefix }, { default: () => h(MoreIcon, null) });
                    if (renderLabel) {
                      contentNode = renderLabel({
                        type: "fast-backward",
                        node: fastBackwardNode,
                        active: this.fastBackwardActive || this.showFastBackwardMenu
                      });
                    } else {
                      contentNode = fastBackwardNode;
                    }
                    onMouseenter = this.handleFastBackwardMouseenter;
                    onMouseleave = this.handleFastBackwardMouseleave;
                    break;
                }
                const itemNode = h("div", { key: index, class: [
                  `${mergedClsPrefix}-pagination-item`,
                  pageItem.active && `${mergedClsPrefix}-pagination-item--active`,
                  type !== "page" && (type === "fast-backward" && this.showFastBackwardMenu || type === "fast-forward" && this.showFastForwardMenu) && `${mergedClsPrefix}-pagination-item--hover`,
                  disabled && `${mergedClsPrefix}-pagination-item--disabled`,
                  type === "page" && `${mergedClsPrefix}-pagination-item--clickable`
                ], onClick: () => {
                  handlePageItemClick(pageItem);
                }, onMouseenter, onMouseleave }, contentNode);
                if (type === "page" && !pageItem.mayBeFastBackward && !pageItem.mayBeFastForward) {
                  return itemNode;
                } else {
                  const key = pageItem.type === "page" ? pageItem.mayBeFastBackward ? "fast-backward" : "fast-forward" : pageItem.type;
                  return h(NPopselect, { to: this.to, key, disabled, trigger: "hover", virtualScroll: true, style: { width: "60px" }, theme: mergedTheme.peers.Popselect, themeOverrides: mergedTheme.peerOverrides.Popselect, builtinThemeOverrides: {
                    peers: {
                      InternalSelectMenu: {
                        height: "calc(var(--n-option-height) * 4.6)"
                      }
                    }
                  }, nodeProps: () => ({
                    style: {
                      justifyContent: "center"
                    }
                  }), show: type === "page" ? false : type === "fast-backward" ? this.showFastBackwardMenu : this.showFastForwardMenu, onUpdateShow: (value) => {
                    if (type === "page")
                      return;
                    if (value) {
                      if (type === "fast-backward") {
                        this.showFastBackwardMenu = value;
                      } else {
                        this.showFastForwardMenu = value;
                      }
                    } else {
                      this.showFastBackwardMenu = false;
                      this.showFastForwardMenu = false;
                    }
                  }, options: pageItem.type !== "page" ? pageItem.options : [], onUpdateValue: this.handleMenuSelect, scrollable: true, showCheckmark: false }, { default: () => itemNode });
                }
              }),
              h("div", { class: [
                `${mergedClsPrefix}-pagination-item`,
                !renderNext && `${mergedClsPrefix}-pagination-item--button`,
                {
                  [`${mergedClsPrefix}-pagination-item--disabled`]: mergedPage < 1 || mergedPage >= mergedPageCount || disabled
                }
              ], onClick: handleForwardClick }, renderNext ? renderNext({
                page: mergedPage,
                pageSize: mergedPageSize,
                pageCount: mergedPageCount,
                itemCount: this.mergedItemCount,
                startIndex: this.startIndex,
                endIndex: this.endIndex
              }) : h(NBaseIcon, { clsPrefix: mergedClsPrefix }, {
                default: () => this.rtlEnabled ? h(BackwardIcon, null) : h(ForwardIcon, null)
              }))
            );
          case "size-picker": {
            return !simple && showSizePicker ? h(NSelect, Object.assign({ consistentMenuWidth: false, placeholder: "", showCheckmark: false, to: this.to }, this.selectProps, { size: selectSize, options: pageSizeOptions, value: mergedPageSize, disabled, theme: mergedTheme.peers.Select, themeOverrides: mergedTheme.peerOverrides.Select, onUpdateValue: handleSizePickerChange })) : null;
          }
          case "quick-jumper":
            return !simple && showQuickJumper ? h(
              "div",
              { class: `${mergedClsPrefix}-pagination-quick-jumper` },
              goto ? goto() : resolveSlot(this.$slots.goto, () => [locale.goto]),
              h(NInput, { value: jumperValue, onUpdateValue: handleJumperInput, size: inputSize, placeholder: "", disabled, theme: mergedTheme.peers.Input, themeOverrides: mergedTheme.peerOverrides.Input, onChange: handleQuickJumperChange })
            ) : null;
          default:
            return null;
        }
      }),
      renderSuffix ? h("div", { class: `${mergedClsPrefix}-pagination-suffix` }, renderSuffix({
        page: mergedPage,
        pageSize: mergedPageSize,
        pageCount: mergedPageCount,
        startIndex: this.startIndex,
        endIndex: this.endIndex,
        itemCount: this.mergedItemCount
      })) : null
    );
  }
});
const style$4 = cB("ellipsis", {
  overflow: "hidden"
}, [cNotM("line-clamp", `
 white-space: nowrap;
 display: inline-block;
 vertical-align: bottom;
 max-width: 100%;
 `), cM("line-clamp", `
 display: -webkit-inline-box;
 -webkit-box-orient: vertical;
 `), cM("cursor-pointer", `
 cursor: pointer;
 `)]);
function createLineClampClass(clsPrefix) {
  return `${clsPrefix}-ellipsis--line-clamp`;
}
function createCursorClass(clsPrefix, cursor) {
  return `${clsPrefix}-ellipsis--cursor-${cursor}`;
}
const ellipsisProps = Object.assign(Object.assign({}, useTheme.props), { expandTrigger: String, lineClamp: [Number, String], tooltip: {
  type: [Boolean, Object],
  default: true
} });
const NEllipsis = defineComponent({
  name: "Ellipsis",
  inheritAttrs: false,
  props: ellipsisProps,
  setup(props, { slots, attrs }) {
    const { mergedClsPrefixRef } = useConfig(props);
    const mergedTheme = useTheme("Ellipsis", "-ellipsis", style$4, ellipsisLight, props, mergedClsPrefixRef);
    const triggerRef = ref(null);
    const triggerInnerRef = ref(null);
    const tooltipRef = ref(null);
    const expandedRef = ref(false);
    const ellipsisStyleRef = computed(() => {
      const { lineClamp } = props;
      const { value: expanded } = expandedRef;
      if (lineClamp !== void 0) {
        return {
          textOverflow: "",
          "-webkit-line-clamp": expanded ? "" : lineClamp
        };
      } else {
        return {
          textOverflow: expanded ? "" : "ellipsis",
          "-webkit-line-clamp": ""
        };
      }
    });
    function getTooltipDisabled() {
      let tooltipDisabled = false;
      const { value: expanded } = expandedRef;
      if (expanded)
        return true;
      const { value: trigger } = triggerRef;
      if (trigger) {
        const { lineClamp } = props;
        syncEllipsisStyle(trigger);
        if (lineClamp !== void 0) {
          tooltipDisabled = trigger.scrollHeight <= trigger.offsetHeight;
        } else {
          const { value: triggerInner } = triggerInnerRef;
          if (triggerInner) {
            tooltipDisabled = triggerInner.getBoundingClientRect().width <= trigger.getBoundingClientRect().width;
          }
        }
        syncCursorStyle(trigger, tooltipDisabled);
      }
      return tooltipDisabled;
    }
    const handleClickRef = computed(() => {
      return props.expandTrigger === "click" ? () => {
        var _a;
        const { value: expanded } = expandedRef;
        if (expanded) {
          (_a = tooltipRef.value) === null || _a === void 0 ? void 0 : _a.setShow(false);
        }
        expandedRef.value = !expanded;
      } : void 0;
    });
    onDeactivated(() => {
      var _a;
      if (props.tooltip) {
        (_a = tooltipRef.value) === null || _a === void 0 ? void 0 : _a.setShow(false);
      }
    });
    const renderTrigger = () => h("span", Object.assign({}, mergeProps(attrs, {
      class: [
        `${mergedClsPrefixRef.value}-ellipsis`,
        props.lineClamp !== void 0 ? createLineClampClass(mergedClsPrefixRef.value) : void 0,
        props.expandTrigger === "click" ? createCursorClass(mergedClsPrefixRef.value, "pointer") : void 0
      ],
      style: ellipsisStyleRef.value
    }), { ref: "triggerRef", onClick: handleClickRef.value, onMouseenter: (
      // get tooltip disabled will derive cursor style
      props.expandTrigger === "click" ? getTooltipDisabled : void 0
    ) }), props.lineClamp ? slots : h("span", { ref: "triggerInnerRef" }, slots));
    function syncEllipsisStyle(trigger) {
      if (!trigger)
        return;
      const latestStyle = ellipsisStyleRef.value;
      const lineClampClass = createLineClampClass(mergedClsPrefixRef.value);
      if (props.lineClamp !== void 0) {
        syncTriggerClass(trigger, lineClampClass, "add");
      } else {
        syncTriggerClass(trigger, lineClampClass, "remove");
      }
      for (const key in latestStyle) {
        if (trigger.style[key] !== latestStyle[key]) {
          trigger.style[key] = latestStyle[key];
        }
      }
    }
    function syncCursorStyle(trigger, tooltipDisabled) {
      const cursorClass = createCursorClass(mergedClsPrefixRef.value, "pointer");
      if (props.expandTrigger === "click" && !tooltipDisabled) {
        syncTriggerClass(trigger, cursorClass, "add");
      } else {
        syncTriggerClass(trigger, cursorClass, "remove");
      }
    }
    function syncTriggerClass(trigger, styleClass, action) {
      if (action === "add") {
        if (!trigger.classList.contains(styleClass)) {
          trigger.classList.add(styleClass);
        }
      } else {
        if (trigger.classList.contains(styleClass)) {
          trigger.classList.remove(styleClass);
        }
      }
    }
    return {
      mergedTheme,
      triggerRef,
      triggerInnerRef,
      tooltipRef,
      handleClick: handleClickRef,
      renderTrigger,
      getTooltipDisabled
    };
  },
  render() {
    var _a;
    const { tooltip, renderTrigger, $slots } = this;
    if (tooltip) {
      const { mergedTheme } = this;
      return h(NTooltip, Object.assign({ ref: "tooltipRef", placement: "top" }, tooltip, { getDisabled: this.getTooltipDisabled, theme: mergedTheme.peers.Tooltip, themeOverrides: mergedTheme.peerOverrides.Tooltip }), {
        trigger: renderTrigger,
        default: (_a = $slots.tooltip) !== null && _a !== void 0 ? _a : $slots.default
      });
    } else
      return renderTrigger();
  }
});
const RenderSorter = defineComponent({
  name: "DataTableRenderSorter",
  props: {
    render: {
      type: Function,
      required: true
    },
    order: {
      // asc, desc
      type: [String, Boolean],
      default: false
    }
  },
  render() {
    const { render: render5, order } = this;
    return render5({
      order
    });
  }
});
const dataTableProps = Object.assign(Object.assign({}, useTheme.props), {
  onUnstableColumnResize: Function,
  pagination: {
    type: [Object, Boolean],
    default: false
  },
  paginateSinglePage: {
    type: Boolean,
    default: true
  },
  minHeight: [Number, String],
  maxHeight: [Number, String],
  // Use any type as row data to make prop data acceptable
  columns: {
    type: Array,
    default: () => []
  },
  rowClassName: [String, Function],
  rowProps: Function,
  rowKey: Function,
  summary: [Function],
  data: {
    type: Array,
    default: () => []
  },
  loading: Boolean,
  bordered: {
    type: Boolean,
    default: void 0
  },
  bottomBordered: {
    type: Boolean,
    default: void 0
  },
  striped: Boolean,
  scrollX: [Number, String],
  defaultCheckedRowKeys: {
    type: Array,
    default: () => []
  },
  checkedRowKeys: Array,
  singleLine: {
    type: Boolean,
    default: true
  },
  singleColumn: Boolean,
  size: {
    type: String,
    default: "medium"
  },
  remote: Boolean,
  defaultExpandedRowKeys: {
    type: Array,
    default: []
  },
  defaultExpandAll: Boolean,
  expandedRowKeys: Array,
  stickyExpandedRows: Boolean,
  virtualScroll: Boolean,
  tableLayout: {
    type: String,
    default: "auto"
  },
  allowCheckingNotLoaded: Boolean,
  cascade: {
    type: Boolean,
    default: true
  },
  childrenKey: {
    type: String,
    default: "children"
  },
  indent: {
    type: Number,
    default: 16
  },
  flexHeight: Boolean,
  summaryPlacement: {
    type: String,
    default: "bottom"
  },
  paginationBehaviorOnFilter: {
    type: String,
    default: "current"
  },
  scrollbarProps: Object,
  renderCell: Function,
  renderExpandIcon: Function,
  spinProps: { type: Object, default: {} },
  onLoad: Function,
  "onUpdate:page": [Function, Array],
  onUpdatePage: [Function, Array],
  "onUpdate:pageSize": [Function, Array],
  onUpdatePageSize: [Function, Array],
  "onUpdate:sorter": [Function, Array],
  onUpdateSorter: [Function, Array],
  "onUpdate:filters": [Function, Array],
  onUpdateFilters: [Function, Array],
  "onUpdate:checkedRowKeys": [Function, Array],
  onUpdateCheckedRowKeys: [Function, Array],
  "onUpdate:expandedRowKeys": [Function, Array],
  onUpdateExpandedRowKeys: [Function, Array],
  onScroll: Function,
  // deprecated
  onPageChange: [Function, Array],
  onPageSizeChange: [Function, Array],
  onSorterChange: [Function, Array],
  onFiltersChange: [Function, Array],
  onCheckedRowKeysChange: [Function, Array]
});
const dataTableInjectionKey = createInjectionKey("n-data-table");
const SortButton = defineComponent({
  name: "SortIcon",
  props: {
    column: {
      type: Object,
      required: true
    }
  },
  setup(props) {
    const { mergedComponentPropsRef } = useConfig();
    const { mergedSortStateRef, mergedClsPrefixRef } = inject(dataTableInjectionKey);
    const sortStateRef = computed(() => mergedSortStateRef.value.find((state) => state.columnKey === props.column.key));
    const activeRef = computed(() => {
      return sortStateRef.value !== void 0;
    });
    const mergedSortOrderRef = computed(() => {
      const { value: sortState } = sortStateRef;
      if (sortState && activeRef.value) {
        return sortState.order;
      }
      return false;
    });
    const mergedRenderSorterRef = computed(() => {
      var _a, _b;
      return ((_b = (_a = mergedComponentPropsRef === null || mergedComponentPropsRef === void 0 ? void 0 : mergedComponentPropsRef.value) === null || _a === void 0 ? void 0 : _a.DataTable) === null || _b === void 0 ? void 0 : _b.renderSorter) || props.column.renderSorter;
    });
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      active: activeRef,
      mergedSortOrder: mergedSortOrderRef,
      mergedRenderSorter: mergedRenderSorterRef
    };
  },
  render() {
    const { mergedRenderSorter, mergedSortOrder, mergedClsPrefix } = this;
    const { renderSorterIcon } = this.column;
    return mergedRenderSorter ? h(RenderSorter, { render: mergedRenderSorter, order: mergedSortOrder }) : h("span", { class: [
      `${mergedClsPrefix}-data-table-sorter`,
      mergedSortOrder === "ascend" && `${mergedClsPrefix}-data-table-sorter--asc`,
      mergedSortOrder === "descend" && `${mergedClsPrefix}-data-table-sorter--desc`
    ] }, renderSorterIcon ? renderSorterIcon({ order: mergedSortOrder }) : h(NBaseIcon, { clsPrefix: mergedClsPrefix }, { default: () => h(ArrowDownIcon, null) }));
  }
});
const RenderFilter = defineComponent({
  name: "DataTableRenderFilter",
  props: {
    render: {
      type: Function,
      required: true
    },
    active: {
      type: Boolean,
      default: false
    },
    show: {
      type: Boolean,
      default: false
    }
  },
  render() {
    const { render: render5, active, show } = this;
    return render5({
      active,
      show
    });
  }
});
const radioProps = {
  name: String,
  value: {
    type: [String, Number, Boolean],
    default: "on"
  },
  checked: {
    type: Boolean,
    default: void 0
  },
  defaultChecked: Boolean,
  disabled: {
    type: Boolean,
    default: void 0
  },
  label: String,
  size: String,
  onUpdateChecked: [Function, Array],
  "onUpdate:checked": [Function, Array],
  // deprecated
  checkedValue: {
    type: Boolean,
    default: void 0
  }
};
const radioGroupInjectionKey = createInjectionKey("n-radio-group");
function setup(props) {
  const formItem = useFormItem(props, {
    mergedSize(NFormItem) {
      const { size } = props;
      if (size !== void 0)
        return size;
      if (NRadioGroup2) {
        const { mergedSizeRef: { value: mergedSize } } = NRadioGroup2;
        if (mergedSize !== void 0) {
          return mergedSize;
        }
      }
      if (NFormItem) {
        return NFormItem.mergedSize.value;
      }
      return "medium";
    },
    mergedDisabled(NFormItem) {
      if (props.disabled)
        return true;
      if (NRadioGroup2 === null || NRadioGroup2 === void 0 ? void 0 : NRadioGroup2.disabledRef.value)
        return true;
      if (NFormItem === null || NFormItem === void 0 ? void 0 : NFormItem.disabled.value)
        return true;
      return false;
    }
  });
  const { mergedSizeRef, mergedDisabledRef } = formItem;
  const inputRef = ref(null);
  const labelRef = ref(null);
  const NRadioGroup2 = inject(radioGroupInjectionKey, null);
  const uncontrolledCheckedRef = ref(props.defaultChecked);
  const controlledCheckedRef = toRef(props, "checked");
  const mergedCheckedRef = useMergedState(controlledCheckedRef, uncontrolledCheckedRef);
  const renderSafeCheckedRef = useMemo(() => {
    if (NRadioGroup2)
      return NRadioGroup2.valueRef.value === props.value;
    return mergedCheckedRef.value;
  });
  const mergedNameRef = useMemo(() => {
    const { name } = props;
    if (name !== void 0)
      return name;
    if (NRadioGroup2)
      return NRadioGroup2.nameRef.value;
  });
  const focusRef = ref(false);
  function doUpdateChecked() {
    if (NRadioGroup2) {
      const { doUpdateValue } = NRadioGroup2;
      const { value } = props;
      call(doUpdateValue, value);
    } else {
      const { onUpdateChecked, "onUpdate:checked": _onUpdateChecked } = props;
      const { nTriggerFormInput, nTriggerFormChange } = formItem;
      if (onUpdateChecked)
        call(onUpdateChecked, true);
      if (_onUpdateChecked)
        call(_onUpdateChecked, true);
      nTriggerFormInput();
      nTriggerFormChange();
      uncontrolledCheckedRef.value = true;
    }
  }
  function toggle() {
    if (mergedDisabledRef.value)
      return;
    if (!renderSafeCheckedRef.value) {
      doUpdateChecked();
    }
  }
  function handleRadioInputChange() {
    toggle();
  }
  function handleRadioInputBlur() {
    focusRef.value = false;
  }
  function handleRadioInputFocus() {
    focusRef.value = true;
  }
  return {
    mergedClsPrefix: NRadioGroup2 ? NRadioGroup2.mergedClsPrefixRef : useConfig(props).mergedClsPrefixRef,
    inputRef,
    labelRef,
    mergedName: mergedNameRef,
    mergedDisabled: mergedDisabledRef,
    uncontrolledChecked: uncontrolledCheckedRef,
    renderSafeChecked: renderSafeCheckedRef,
    focus: focusRef,
    mergedSize: mergedSizeRef,
    handleRadioInputChange,
    handleRadioInputBlur,
    handleRadioInputFocus
  };
}
const style$3 = cB("radio", `
 line-height: var(--n-label-line-height);
 outline: none;
 position: relative;
 user-select: none;
 -webkit-user-select: none;
 display: inline-flex;
 align-items: flex-start;
 flex-wrap: nowrap;
 font-size: var(--n-font-size);
 word-break: break-word;
`, [cM("checked", [cE("dot", `
 background-color: var(--n-color-active);
 `)]), cE("dot-wrapper", `
 position: relative;
 flex-shrink: 0;
 flex-grow: 0;
 width: var(--n-radio-size);
 `), cB("radio-input", `
 position: absolute;
 border: 0;
 border-radius: inherit;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 opacity: 0;
 z-index: 1;
 cursor: pointer;
 `), cE("dot", `
 position: absolute;
 top: 50%;
 left: 0;
 transform: translateY(-50%);
 height: var(--n-radio-size);
 width: var(--n-radio-size);
 background: var(--n-color);
 box-shadow: var(--n-box-shadow);
 border-radius: 50%;
 transition:
 background-color .3s var(--n-bezier),
 box-shadow .3s var(--n-bezier);
 `, [c("&::before", `
 content: "";
 opacity: 0;
 position: absolute;
 left: 4px;
 top: 4px;
 height: calc(100% - 8px);
 width: calc(100% - 8px);
 border-radius: 50%;
 transform: scale(.8);
 background: var(--n-dot-color-active);
 transition: 
 opacity .3s var(--n-bezier),
 background-color .3s var(--n-bezier),
 transform .3s var(--n-bezier);
 `), cM("checked", {
  boxShadow: "var(--n-box-shadow-active)"
}, [c("&::before", `
 opacity: 1;
 transform: scale(1);
 `)])]), cE("label", `
 color: var(--n-text-color);
 padding: var(--n-label-padding);
 font-weight: var(--n-label-font-weight);
 display: inline-block;
 transition: color .3s var(--n-bezier);
 `), cNotM("disabled", `
 cursor: pointer;
 `, [c("&:hover", [cE("dot", {
  boxShadow: "var(--n-box-shadow-hover)"
})]), cM("focus", [c("&:not(:active)", [cE("dot", {
  boxShadow: "var(--n-box-shadow-focus)"
})])])]), cM("disabled", `
 cursor: not-allowed;
 `, [cE("dot", {
  boxShadow: "var(--n-box-shadow-disabled)",
  backgroundColor: "var(--n-color-disabled)"
}, [c("&::before", {
  backgroundColor: "var(--n-dot-color-disabled)"
}), cM("checked", `
 opacity: 1;
 `)]), cE("label", {
  color: "var(--n-text-color-disabled)"
}), cB("radio-input", `
 cursor: not-allowed;
 `)])]);
const NRadio = defineComponent({
  name: "Radio",
  props: Object.assign(Object.assign({}, useTheme.props), radioProps),
  setup(props) {
    const radio = setup(props);
    const themeRef = useTheme("Radio", "-radio", style$3, radioLight, props, radio.mergedClsPrefix);
    const cssVarsRef = computed(() => {
      const { mergedSize: { value: size } } = radio;
      const { common: { cubicBezierEaseInOut }, self: { boxShadow, boxShadowActive, boxShadowDisabled, boxShadowFocus, boxShadowHover, color, colorDisabled, colorActive, textColor, textColorDisabled, dotColorActive, dotColorDisabled, labelPadding, labelLineHeight, labelFontWeight, [createKey("fontSize", size)]: fontSize, [createKey("radioSize", size)]: radioSize } } = themeRef.value;
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-label-line-height": labelLineHeight,
        "--n-label-font-weight": labelFontWeight,
        "--n-box-shadow": boxShadow,
        "--n-box-shadow-active": boxShadowActive,
        "--n-box-shadow-disabled": boxShadowDisabled,
        "--n-box-shadow-focus": boxShadowFocus,
        "--n-box-shadow-hover": boxShadowHover,
        "--n-color": color,
        "--n-color-active": colorActive,
        "--n-color-disabled": colorDisabled,
        "--n-dot-color-active": dotColorActive,
        "--n-dot-color-disabled": dotColorDisabled,
        "--n-font-size": fontSize,
        "--n-radio-size": radioSize,
        "--n-text-color": textColor,
        "--n-text-color-disabled": textColorDisabled,
        "--n-label-padding": labelPadding
      };
    });
    const { inlineThemeDisabled, mergedClsPrefixRef, mergedRtlRef } = useConfig(props);
    const rtlEnabledRef = useRtl("Radio", mergedRtlRef, mergedClsPrefixRef);
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("radio", computed(() => radio.mergedSize.value[0]), cssVarsRef, props) : void 0;
    return Object.assign(radio, {
      rtlEnabled: rtlEnabledRef,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    });
  },
  render() {
    const { $slots, mergedClsPrefix, onRender, label } = this;
    onRender === null || onRender === void 0 ? void 0 : onRender();
    return h(
      "label",
      { class: [
        `${mergedClsPrefix}-radio`,
        this.themeClass,
        {
          [`${mergedClsPrefix}-radio--rtl`]: this.rtlEnabled,
          [`${mergedClsPrefix}-radio--disabled`]: this.mergedDisabled,
          [`${mergedClsPrefix}-radio--checked`]: this.renderSafeChecked,
          [`${mergedClsPrefix}-radio--focus`]: this.focus
        }
      ], style: this.cssVars },
      h("input", { ref: "inputRef", type: "radio", class: `${mergedClsPrefix}-radio-input`, value: this.value, name: this.mergedName, checked: this.renderSafeChecked, disabled: this.mergedDisabled, onChange: this.handleRadioInputChange, onFocus: this.handleRadioInputFocus, onBlur: this.handleRadioInputBlur }),
      h(
        "div",
        { class: `${mergedClsPrefix}-radio__dot-wrapper` },
        " ",
        h("div", { class: [
          `${mergedClsPrefix}-radio__dot`,
          this.renderSafeChecked && `${mergedClsPrefix}-radio__dot--checked`
        ] })
      ),
      resolveWrappedSlot($slots.default, (children) => {
        if (!children && !label)
          return null;
        return h("div", { ref: "labelRef", class: `${mergedClsPrefix}-radio__label` }, children || label);
      })
    );
  }
});
const style$2 = cB("radio-group", `
 display: inline-block;
 font-size: var(--n-font-size);
`, [cE("splitor", `
 display: inline-block;
 vertical-align: bottom;
 width: 1px;
 transition:
 background-color .3s var(--n-bezier),
 opacity .3s var(--n-bezier);
 background: var(--n-button-border-color);
 `, [cM("checked", {
  backgroundColor: "var(--n-button-border-color-active)"
}), cM("disabled", {
  opacity: "var(--n-opacity-disabled)"
})]), cM("button-group", `
 white-space: nowrap;
 height: var(--n-height);
 line-height: var(--n-height);
 `, [cB("radio-button", {
  height: "var(--n-height)",
  lineHeight: "var(--n-height)"
}), cE("splitor", {
  height: "var(--n-height)"
})]), cB("radio-button", `
 vertical-align: bottom;
 outline: none;
 position: relative;
 user-select: none;
 -webkit-user-select: none;
 display: inline-block;
 box-sizing: border-box;
 padding-left: 14px;
 padding-right: 14px;
 white-space: nowrap;
 transition:
 background-color .3s var(--n-bezier),
 opacity .3s var(--n-bezier),
 border-color .3s var(--n-bezier),
 color .3s var(--n-bezier);
 color: var(--n-button-text-color);
 border-top: 1px solid var(--n-button-border-color);
 border-bottom: 1px solid var(--n-button-border-color);
 `, [cB("radio-input", `
 pointer-events: none;
 position: absolute;
 border: 0;
 border-radius: inherit;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 opacity: 0;
 z-index: 1;
 `), cE("state-border", `
 z-index: 1;
 pointer-events: none;
 position: absolute;
 box-shadow: var(--n-button-box-shadow);
 transition: box-shadow .3s var(--n-bezier);
 left: -1px;
 bottom: -1px;
 right: -1px;
 top: -1px;
 `), c("&:first-child", `
 border-top-left-radius: var(--n-button-border-radius);
 border-bottom-left-radius: var(--n-button-border-radius);
 border-left: 1px solid var(--n-button-border-color);
 `, [cE("state-border", `
 border-top-left-radius: var(--n-button-border-radius);
 border-bottom-left-radius: var(--n-button-border-radius);
 `)]), c("&:last-child", `
 border-top-right-radius: var(--n-button-border-radius);
 border-bottom-right-radius: var(--n-button-border-radius);
 border-right: 1px solid var(--n-button-border-color);
 `, [cE("state-border", `
 border-top-right-radius: var(--n-button-border-radius);
 border-bottom-right-radius: var(--n-button-border-radius);
 `)]), cNotM("disabled", `
 cursor: pointer;
 `, [c("&:hover", [cE("state-border", `
 transition: box-shadow .3s var(--n-bezier);
 box-shadow: var(--n-button-box-shadow-hover);
 `), cNotM("checked", {
  color: "var(--n-button-text-color-hover)"
})]), cM("focus", [c("&:not(:active)", [cE("state-border", {
  boxShadow: "var(--n-button-box-shadow-focus)"
})])])]), cM("checked", `
 background: var(--n-button-color-active);
 color: var(--n-button-text-color-active);
 border-color: var(--n-button-border-color-active);
 `), cM("disabled", `
 cursor: not-allowed;
 opacity: var(--n-opacity-disabled);
 `)])]);
function mapSlot(defaultSlot, value, clsPrefix) {
  var _a;
  const children = [];
  let isButtonGroup = false;
  for (let i = 0; i < defaultSlot.length; ++i) {
    const wrappedInstance = defaultSlot[i];
    const name = (_a = wrappedInstance.type) === null || _a === void 0 ? void 0 : _a.name;
    if (name === "RadioButton") {
      isButtonGroup = true;
    }
    const instanceProps = wrappedInstance.props;
    if (name !== "RadioButton") {
      children.push(wrappedInstance);
      continue;
    }
    if (i === 0) {
      children.push(wrappedInstance);
    } else {
      const lastInstanceProps = children[children.length - 1].props;
      const lastInstanceChecked = value === lastInstanceProps.value;
      const lastInstanceDisabled = lastInstanceProps.disabled;
      const currentInstanceChecked = value === instanceProps.value;
      const currentInstanceDisabled = instanceProps.disabled;
      const lastInstancePriority = (lastInstanceChecked ? 2 : 0) + (!lastInstanceDisabled ? 1 : 0);
      const currentInstancePriority = (currentInstanceChecked ? 2 : 0) + (!currentInstanceDisabled ? 1 : 0);
      const lastInstanceClass = {
        [`${clsPrefix}-radio-group__splitor--disabled`]: lastInstanceDisabled,
        [`${clsPrefix}-radio-group__splitor--checked`]: lastInstanceChecked
      };
      const currentInstanceClass = {
        [`${clsPrefix}-radio-group__splitor--disabled`]: currentInstanceDisabled,
        [`${clsPrefix}-radio-group__splitor--checked`]: currentInstanceChecked
      };
      const splitorClass = lastInstancePriority < currentInstancePriority ? currentInstanceClass : lastInstanceClass;
      children.push(h("div", { class: [`${clsPrefix}-radio-group__splitor`, splitorClass] }), wrappedInstance);
    }
  }
  return {
    children,
    isButtonGroup
  };
}
const radioGroupProps = Object.assign(Object.assign({}, useTheme.props), { name: String, value: [String, Number, Boolean], defaultValue: {
  type: [String, Number, Boolean],
  default: null
}, size: String, disabled: {
  type: Boolean,
  default: void 0
}, "onUpdate:value": [Function, Array], onUpdateValue: [Function, Array] });
const NRadioGroup = defineComponent({
  name: "RadioGroup",
  props: radioGroupProps,
  setup(props) {
    const selfElRef = ref(null);
    const { mergedSizeRef, mergedDisabledRef, nTriggerFormChange, nTriggerFormInput, nTriggerFormBlur, nTriggerFormFocus } = useFormItem(props);
    const { mergedClsPrefixRef, inlineThemeDisabled, mergedRtlRef } = useConfig(props);
    const themeRef = useTheme("Radio", "-radio-group", style$2, radioLight, props, mergedClsPrefixRef);
    const uncontrolledValueRef = ref(props.defaultValue);
    const controlledValueRef = toRef(props, "value");
    const mergedValueRef = useMergedState(controlledValueRef, uncontrolledValueRef);
    function doUpdateValue(value) {
      const { onUpdateValue, "onUpdate:value": _onUpdateValue } = props;
      if (onUpdateValue) {
        call(onUpdateValue, value);
      }
      if (_onUpdateValue) {
        call(_onUpdateValue, value);
      }
      uncontrolledValueRef.value = value;
      nTriggerFormChange();
      nTriggerFormInput();
    }
    function handleFocusin(e) {
      const { value: selfEl } = selfElRef;
      if (!selfEl)
        return;
      if (selfEl.contains(e.relatedTarget))
        return;
      nTriggerFormFocus();
    }
    function handleFocusout(e) {
      const { value: selfEl } = selfElRef;
      if (!selfEl)
        return;
      if (selfEl.contains(e.relatedTarget))
        return;
      nTriggerFormBlur();
    }
    provide(radioGroupInjectionKey, {
      mergedClsPrefixRef,
      nameRef: toRef(props, "name"),
      valueRef: mergedValueRef,
      disabledRef: mergedDisabledRef,
      mergedSizeRef,
      doUpdateValue
    });
    const rtlEnabledRef = useRtl("Radio", mergedRtlRef, mergedClsPrefixRef);
    const cssVarsRef = computed(() => {
      const { value: size } = mergedSizeRef;
      const { common: { cubicBezierEaseInOut }, self: { buttonBorderColor, buttonBorderColorActive, buttonBorderRadius, buttonBoxShadow, buttonBoxShadowFocus, buttonBoxShadowHover, buttonColorActive, buttonTextColor, buttonTextColorActive, buttonTextColorHover, opacityDisabled, [createKey("buttonHeight", size)]: height, [createKey("fontSize", size)]: fontSize } } = themeRef.value;
      return {
        "--n-font-size": fontSize,
        "--n-bezier": cubicBezierEaseInOut,
        "--n-button-border-color": buttonBorderColor,
        "--n-button-border-color-active": buttonBorderColorActive,
        "--n-button-border-radius": buttonBorderRadius,
        "--n-button-box-shadow": buttonBoxShadow,
        "--n-button-box-shadow-focus": buttonBoxShadowFocus,
        "--n-button-box-shadow-hover": buttonBoxShadowHover,
        "--n-button-color-active": buttonColorActive,
        "--n-button-text-color": buttonTextColor,
        "--n-button-text-color-hover": buttonTextColorHover,
        "--n-button-text-color-active": buttonTextColorActive,
        "--n-height": height,
        "--n-opacity-disabled": opacityDisabled
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("radio-group", computed(() => mergedSizeRef.value[0]), cssVarsRef, props) : void 0;
    return {
      selfElRef,
      rtlEnabled: rtlEnabledRef,
      mergedClsPrefix: mergedClsPrefixRef,
      mergedValue: mergedValueRef,
      handleFocusout,
      handleFocusin,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    var _a;
    const { mergedValue, mergedClsPrefix, handleFocusin, handleFocusout } = this;
    const { children, isButtonGroup } = mapSlot(flatten$1(getSlot(this)), mergedValue, mergedClsPrefix);
    (_a = this.onRender) === null || _a === void 0 ? void 0 : _a.call(this);
    return h("div", { onFocusin: handleFocusin, onFocusout: handleFocusout, ref: "selfElRef", class: [
      `${mergedClsPrefix}-radio-group`,
      this.rtlEnabled && `${mergedClsPrefix}-radio-group--rtl`,
      this.themeClass,
      isButtonGroup && `${mergedClsPrefix}-radio-group--button-group`
    ], style: this.cssVars }, children);
  }
});
const SELECTION_COL_WIDTH = 40;
const EXPAND_COL_WIDTH = 40;
function getNumberColWidth(col) {
  if (col.type === "selection") {
    return col.width === void 0 ? SELECTION_COL_WIDTH : depx(col.width);
  }
  if (col.type === "expand") {
    return col.width === void 0 ? EXPAND_COL_WIDTH : depx(col.width);
  }
  if ("children" in col)
    return void 0;
  if (typeof col.width === "string") {
    return depx(col.width);
  }
  return col.width;
}
function getStringColWidth(col) {
  var _a, _b;
  if (col.type === "selection") {
    return formatLength((_a = col.width) !== null && _a !== void 0 ? _a : SELECTION_COL_WIDTH);
  }
  if (col.type === "expand") {
    return formatLength((_b = col.width) !== null && _b !== void 0 ? _b : EXPAND_COL_WIDTH);
  }
  if ("children" in col) {
    return void 0;
  }
  return formatLength(col.width);
}
function getColKey(col) {
  if (col.type === "selection")
    return "__n_selection__";
  if (col.type === "expand")
    return "__n_expand__";
  return col.key;
}
function createShallowClonedObject(object) {
  if (!object)
    return object;
  if (typeof object === "object") {
    return Object.assign({}, object);
  }
  return object;
}
function getFlagOfOrder(order) {
  if (order === "ascend")
    return 1;
  else if (order === "descend")
    return -1;
  return 0;
}
function clampValueFollowCSSRules(value, min, max) {
  if (max !== void 0) {
    value = Math.min(value, typeof max === "number" ? max : parseFloat(max));
  }
  if (min !== void 0) {
    value = Math.max(value, typeof min === "number" ? min : parseFloat(min));
  }
  return value;
}
function createCustomWidthStyle(column, resizedWidth) {
  if (resizedWidth !== void 0) {
    return {
      width: resizedWidth,
      minWidth: resizedWidth,
      maxWidth: resizedWidth
    };
  }
  const width = getStringColWidth(column);
  const { minWidth, maxWidth } = column;
  return {
    width,
    minWidth: formatLength(minWidth) || width,
    maxWidth: formatLength(maxWidth)
  };
}
function createRowClassName(row, index, rowClassName) {
  if (typeof rowClassName === "function")
    return rowClassName(row, index);
  return rowClassName || "";
}
function shouldUseArrayInSingleMode(column) {
  return column.filterOptionValues !== void 0 || column.filterOptionValue === void 0 && column.defaultFilterOptionValues !== void 0;
}
function isColumnSortable(column) {
  if ("children" in column)
    return false;
  return !!column.sorter;
}
function isColumnResizable(column) {
  if ("children" in column && !!column.children.length)
    return false;
  return !!column.resizable;
}
function isColumnFilterable(column) {
  if ("children" in column)
    return false;
  return !!column.filter && (!!column.filterOptions || !!column.renderFilterMenu);
}
function getNextOrderOf(order) {
  if (!order)
    return "descend";
  else if (order === "descend")
    return "ascend";
  return false;
}
function createNextSorter(column, currentSortState) {
  if (column.sorter === void 0)
    return null;
  if (currentSortState === null || currentSortState.columnKey !== column.key) {
    return {
      columnKey: column.key,
      sorter: column.sorter,
      order: getNextOrderOf(false)
    };
  } else {
    return Object.assign(Object.assign({}, currentSortState), { order: getNextOrderOf(currentSortState.order) });
  }
}
function isColumnSorting(column, mergedSortState) {
  return mergedSortState.find((state) => state.columnKey === column.key && state.order) !== void 0;
}
const NDataTableFilterMenu = defineComponent({
  name: "DataTableFilterMenu",
  props: {
    column: {
      type: Object,
      required: true
    },
    radioGroupName: {
      type: String,
      required: true
    },
    multiple: {
      type: Boolean,
      required: true
    },
    value: {
      type: [Array, String, Number],
      default: null
    },
    options: {
      type: Array,
      required: true
    },
    onConfirm: {
      type: Function,
      required: true
    },
    onClear: {
      type: Function,
      required: true
    },
    onChange: {
      type: Function,
      required: true
    }
  },
  setup(props) {
    const {
      mergedClsPrefixRef,
      mergedThemeRef,
      localeRef
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    } = inject(dataTableInjectionKey);
    const temporalValueRef = ref(props.value);
    const checkboxGroupValueRef = computed(() => {
      const { value: temporalValue } = temporalValueRef;
      if (!Array.isArray(temporalValue))
        return null;
      return temporalValue;
    });
    const radioGroupValueRef = computed(() => {
      const { value: temporalValue } = temporalValueRef;
      if (shouldUseArrayInSingleMode(props.column)) {
        return Array.isArray(temporalValue) && temporalValue.length && temporalValue[0] || null;
      }
      if (!Array.isArray(temporalValue))
        return temporalValue;
      return null;
    });
    function doChange(value) {
      props.onChange(value);
    }
    function handleChange(value) {
      if (props.multiple && Array.isArray(value)) {
        temporalValueRef.value = value;
      } else if (shouldUseArrayInSingleMode(props.column) && !Array.isArray(value)) {
        temporalValueRef.value = [value];
      } else {
        temporalValueRef.value = value;
      }
    }
    function handleConfirmClick() {
      doChange(temporalValueRef.value);
      props.onConfirm();
    }
    function handleClearClick() {
      if (props.multiple || shouldUseArrayInSingleMode(props.column)) {
        doChange([]);
      } else {
        doChange(null);
      }
      props.onClear();
    }
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      mergedTheme: mergedThemeRef,
      locale: localeRef,
      checkboxGroupValue: checkboxGroupValueRef,
      radioGroupValue: radioGroupValueRef,
      handleChange,
      handleConfirmClick,
      handleClearClick
    };
  },
  render() {
    const { mergedTheme, locale, mergedClsPrefix } = this;
    return h(
      "div",
      { class: `${mergedClsPrefix}-data-table-filter-menu` },
      h(NScrollbar, null, {
        default: () => {
          const { checkboxGroupValue, handleChange } = this;
          return this.multiple ? h(NCheckboxGroup, { value: checkboxGroupValue, class: `${mergedClsPrefix}-data-table-filter-menu__group`, onUpdateValue: handleChange }, {
            default: () => this.options.map((option) => {
              return h(NCheckbox, { key: option.value, theme: mergedTheme.peers.Checkbox, themeOverrides: mergedTheme.peerOverrides.Checkbox, value: option.value }, { default: () => option.label });
            })
          }) : h(NRadioGroup, { name: this.radioGroupName, class: `${mergedClsPrefix}-data-table-filter-menu__group`, value: this.radioGroupValue, onUpdateValue: this.handleChange }, {
            default: () => this.options.map((option) => h(NRadio, { key: option.value, value: option.value, theme: mergedTheme.peers.Radio, themeOverrides: mergedTheme.peerOverrides.Radio }, { default: () => option.label }))
          });
        }
      }),
      h(
        "div",
        { class: `${mergedClsPrefix}-data-table-filter-menu__action` },
        h(NButton, { size: "tiny", theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, onClick: this.handleClearClick }, { default: () => locale.clear }),
        h(NButton, { theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, type: "primary", size: "tiny", onClick: this.handleConfirmClick }, { default: () => locale.confirm })
      )
    );
  }
});
function createFilterState(currentFilterState, columnKey, mergedFilterValue) {
  const nextFilterState = Object.assign({}, currentFilterState);
  nextFilterState[columnKey] = mergedFilterValue;
  return nextFilterState;
}
const FilterButton = defineComponent({
  name: "DataTableFilterButton",
  props: {
    column: {
      type: Object,
      required: true
    },
    options: {
      type: Array,
      default: () => []
    }
  },
  setup(props) {
    const { mergedComponentPropsRef } = useConfig();
    const {
      mergedThemeRef,
      mergedClsPrefixRef,
      mergedFilterStateRef,
      filterMenuCssVarsRef,
      paginationBehaviorOnFilterRef,
      doUpdatePage,
      doUpdateFilters
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    } = inject(dataTableInjectionKey);
    const showPopoverRef = ref(false);
    const filterStateRef = mergedFilterStateRef;
    const filterMultipleRef = computed(() => {
      return props.column.filterMultiple !== false;
    });
    const mergedFilterValueRef = computed(() => {
      const filterValue = filterStateRef.value[props.column.key];
      if (filterValue === void 0) {
        const { value: multiple } = filterMultipleRef;
        if (multiple)
          return [];
        else
          return null;
      }
      return filterValue;
    });
    const activeRef = computed(() => {
      const { value: filterValue } = mergedFilterValueRef;
      if (Array.isArray(filterValue)) {
        return filterValue.length > 0;
      }
      return filterValue !== null;
    });
    const mergedRenderFilterRef = computed(() => {
      var _a, _b;
      return ((_b = (_a = mergedComponentPropsRef === null || mergedComponentPropsRef === void 0 ? void 0 : mergedComponentPropsRef.value) === null || _a === void 0 ? void 0 : _a.DataTable) === null || _b === void 0 ? void 0 : _b.renderFilter) || props.column.renderFilter;
    });
    function handleFilterChange(mergedFilterValue) {
      const nextFilterState = createFilterState(filterStateRef.value, props.column.key, mergedFilterValue);
      doUpdateFilters(nextFilterState, props.column);
      if (paginationBehaviorOnFilterRef.value === "first") {
        doUpdatePage(1);
      }
    }
    function handleFilterMenuCancel() {
      showPopoverRef.value = false;
    }
    function handleFilterMenuConfirm() {
      showPopoverRef.value = false;
    }
    return {
      mergedTheme: mergedThemeRef,
      mergedClsPrefix: mergedClsPrefixRef,
      active: activeRef,
      showPopover: showPopoverRef,
      mergedRenderFilter: mergedRenderFilterRef,
      filterMultiple: filterMultipleRef,
      mergedFilterValue: mergedFilterValueRef,
      filterMenuCssVars: filterMenuCssVarsRef,
      handleFilterChange,
      handleFilterMenuConfirm,
      handleFilterMenuCancel
    };
  },
  render() {
    const { mergedTheme, mergedClsPrefix, handleFilterMenuCancel } = this;
    return h(NPopover, { show: this.showPopover, onUpdateShow: (v) => this.showPopover = v, trigger: "click", theme: mergedTheme.peers.Popover, themeOverrides: mergedTheme.peerOverrides.Popover, placement: "bottom", style: { padding: 0 } }, {
      trigger: () => {
        const { mergedRenderFilter } = this;
        if (mergedRenderFilter) {
          return h(RenderFilter, { "data-data-table-filter": true, render: mergedRenderFilter, active: this.active, show: this.showPopover });
        }
        const { renderFilterIcon } = this.column;
        return h("div", { "data-data-table-filter": true, class: [
          `${mergedClsPrefix}-data-table-filter`,
          {
            [`${mergedClsPrefix}-data-table-filter--active`]: this.active,
            [`${mergedClsPrefix}-data-table-filter--show`]: this.showPopover
          }
        ] }, renderFilterIcon ? renderFilterIcon({
          active: this.active,
          show: this.showPopover
        }) : h(NBaseIcon, { clsPrefix: mergedClsPrefix }, { default: () => h(FilterIcon, null) }));
      },
      default: () => {
        const { renderFilterMenu } = this.column;
        return renderFilterMenu ? renderFilterMenu({ hide: handleFilterMenuCancel }) : h(NDataTableFilterMenu, { style: this.filterMenuCssVars, radioGroupName: String(this.column.key), multiple: this.filterMultiple, value: this.mergedFilterValue, options: this.options, column: this.column, onChange: this.handleFilterChange, onClear: this.handleFilterMenuCancel, onConfirm: this.handleFilterMenuConfirm });
      }
    });
  }
});
const ResizeButton = defineComponent({
  name: "ColumnResizeButton",
  props: {
    onResizeStart: Function,
    onResize: Function,
    onResizeEnd: Function
  },
  setup(props) {
    const { mergedClsPrefixRef } = inject(dataTableInjectionKey);
    const activeRef = ref(false);
    let startX = 0;
    function getMouseX(e) {
      return e.clientX;
    }
    function handleMousedown(e) {
      var _a;
      const alreadyStarted = activeRef.value;
      startX = getMouseX(e);
      activeRef.value = true;
      if (!alreadyStarted) {
        on("mousemove", window, handleMousemove);
        on("mouseup", window, handleMouseup);
        (_a = props.onResizeStart) === null || _a === void 0 ? void 0 : _a.call(props);
      }
    }
    function handleMousemove(e) {
      var _a;
      (_a = props.onResize) === null || _a === void 0 ? void 0 : _a.call(props, getMouseX(e) - startX);
    }
    function handleMouseup() {
      var _a;
      activeRef.value = false;
      (_a = props.onResizeEnd) === null || _a === void 0 ? void 0 : _a.call(props);
      off("mousemove", window, handleMousemove);
      off("mouseup", window, handleMouseup);
    }
    onBeforeUnmount(() => {
      off("mousemove", window, handleMousemove);
      off("mouseup", window, handleMouseup);
    });
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      active: activeRef,
      handleMousedown
    };
  },
  render() {
    const { mergedClsPrefix } = this;
    return h("span", { "data-data-table-resizable": true, class: [
      `${mergedClsPrefix}-data-table-resize-button`,
      this.active && `${mergedClsPrefix}-data-table-resize-button--active`
    ], onMousedown: this.handleMousedown });
  }
});
const allKey = "_n_all__";
const noneKey = "_n_none__";
function createSelectHandler(options, rawPaginatedDataRef, doCheckAll, doUncheckAll) {
  if (!options)
    return () => {
    };
  return (key) => {
    for (const option of options) {
      switch (key) {
        case allKey:
          doCheckAll(true);
          return;
        case noneKey:
          doUncheckAll(true);
          return;
        default:
          if (typeof option === "object" && option.key === key) {
            option.onSelect(rawPaginatedDataRef.value);
            return;
          }
      }
    }
  };
}
function createDropdownOptions(options, localeRef) {
  if (!options)
    return [];
  return options.map((option) => {
    switch (option) {
      case "all":
        return {
          label: localeRef.checkTableAll,
          key: allKey
        };
      case "none":
        return {
          label: localeRef.uncheckTableAll,
          key: noneKey
        };
      default:
        return option;
    }
  });
}
const SelectionMenu = defineComponent({
  name: "DataTableSelectionMenu",
  props: {
    clsPrefix: {
      type: String,
      required: true
    }
  },
  setup(props) {
    const {
      props: dataTableProps2,
      localeRef,
      checkOptionsRef,
      rawPaginatedDataRef,
      doCheckAll,
      doUncheckAll
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    } = inject(dataTableInjectionKey);
    const handleSelectRef = computed(() => createSelectHandler(checkOptionsRef.value, rawPaginatedDataRef, doCheckAll, doUncheckAll));
    const optionsRef = computed(() => createDropdownOptions(checkOptionsRef.value, localeRef.value));
    return () => {
      var _a, _b, _c, _d;
      const { clsPrefix } = props;
      return h(NDropdown, { theme: (_b = (_a = dataTableProps2.theme) === null || _a === void 0 ? void 0 : _a.peers) === null || _b === void 0 ? void 0 : _b.Dropdown, themeOverrides: (_d = (_c = dataTableProps2.themeOverrides) === null || _c === void 0 ? void 0 : _c.peers) === null || _d === void 0 ? void 0 : _d.Dropdown, options: optionsRef.value, onSelect: handleSelectRef.value }, {
        default: () => h(NBaseIcon, { clsPrefix, class: `${clsPrefix}-data-table-check-extra` }, {
          default: () => h(ChevronDownIcon, null)
        })
      });
    };
  }
});
function renderTitle(column) {
  return typeof column.title === "function" ? column.title(column) : column.title;
}
const TableHeader = defineComponent({
  name: "DataTableHeader",
  props: {
    discrete: {
      type: Boolean,
      default: true
    }
  },
  setup() {
    const {
      mergedClsPrefixRef,
      scrollXRef,
      fixedColumnLeftMapRef,
      fixedColumnRightMapRef,
      mergedCurrentPageRef,
      allRowsCheckedRef,
      someRowsCheckedRef,
      rowsRef,
      colsRef,
      mergedThemeRef,
      checkOptionsRef,
      mergedSortStateRef,
      componentId,
      scrollPartRef,
      mergedTableLayoutRef,
      headerCheckboxDisabledRef,
      onUnstableColumnResize,
      doUpdateResizableWidth,
      handleTableHeaderScroll,
      deriveNextSorter,
      doUncheckAll,
      doCheckAll
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    } = inject(dataTableInjectionKey);
    const cellElsRef = ref({});
    function getCellActualWidth(key) {
      const element = cellElsRef.value[key];
      return element === null || element === void 0 ? void 0 : element.getBoundingClientRect().width;
    }
    function handleCheckboxUpdateChecked() {
      if (allRowsCheckedRef.value) {
        doUncheckAll();
      } else {
        doCheckAll();
      }
    }
    function handleColHeaderClick(e, column) {
      if (happensIn(e, "dataTableFilter") || happensIn(e, "dataTableResizable")) {
        return;
      }
      if (!isColumnSortable(column))
        return;
      const activeSorter = mergedSortStateRef.value.find((state) => state.columnKey === column.key) || null;
      const nextSorter = createNextSorter(column, activeSorter);
      deriveNextSorter(nextSorter);
    }
    function handleMouseenter() {
      scrollPartRef.value = "head";
    }
    function handleMouseleave() {
      scrollPartRef.value = "body";
    }
    const resizeStartWidthMap = /* @__PURE__ */ new Map();
    function handleColumnResizeStart(column) {
      resizeStartWidthMap.set(column.key, getCellActualWidth(column.key));
    }
    function handleColumnResize(column, displacementX) {
      const startWidth = resizeStartWidthMap.get(column.key);
      if (startWidth === void 0) {
        return;
      }
      const widthAfterResize = startWidth + displacementX;
      const limitWidth = clampValueFollowCSSRules(widthAfterResize, column.minWidth, column.maxWidth);
      onUnstableColumnResize(widthAfterResize, limitWidth, column, getCellActualWidth);
      doUpdateResizableWidth(column, limitWidth);
    }
    return {
      cellElsRef,
      componentId,
      mergedSortState: mergedSortStateRef,
      mergedClsPrefix: mergedClsPrefixRef,
      scrollX: scrollXRef,
      fixedColumnLeftMap: fixedColumnLeftMapRef,
      fixedColumnRightMap: fixedColumnRightMapRef,
      currentPage: mergedCurrentPageRef,
      allRowsChecked: allRowsCheckedRef,
      someRowsChecked: someRowsCheckedRef,
      rows: rowsRef,
      cols: colsRef,
      mergedTheme: mergedThemeRef,
      checkOptions: checkOptionsRef,
      mergedTableLayout: mergedTableLayoutRef,
      headerCheckboxDisabled: headerCheckboxDisabledRef,
      handleMouseenter,
      handleMouseleave,
      handleCheckboxUpdateChecked,
      handleColHeaderClick,
      handleTableHeaderScroll,
      handleColumnResizeStart,
      handleColumnResize
    };
  },
  render() {
    const { cellElsRef, mergedClsPrefix, fixedColumnLeftMap, fixedColumnRightMap, currentPage, allRowsChecked, someRowsChecked, rows, cols, mergedTheme, checkOptions, componentId, discrete, mergedTableLayout, headerCheckboxDisabled, mergedSortState, handleColHeaderClick, handleCheckboxUpdateChecked, handleColumnResizeStart, handleColumnResize } = this;
    const theadVNode = h("thead", { class: `${mergedClsPrefix}-data-table-thead`, "data-n-id": componentId }, rows.map((row) => {
      return h("tr", { class: `${mergedClsPrefix}-data-table-tr` }, row.map(({ column, colSpan, rowSpan, isLast }) => {
        var _a, _b;
        const key = getColKey(column);
        const { ellipsis } = column;
        const createColumnVNode = () => {
          if (column.type === "selection") {
            return column.multiple !== false ? h(
              Fragment,
              null,
              h(NCheckbox, { key: currentPage, privateInsideTable: true, checked: allRowsChecked, indeterminate: someRowsChecked, disabled: headerCheckboxDisabled, onUpdateChecked: handleCheckboxUpdateChecked }),
              checkOptions ? h(SelectionMenu, { clsPrefix: mergedClsPrefix }) : null
            ) : null;
          }
          return h(
            Fragment,
            null,
            h(
              "div",
              { class: `${mergedClsPrefix}-data-table-th__title-wrapper` },
              h("div", { class: `${mergedClsPrefix}-data-table-th__title` }, ellipsis === true || ellipsis && !ellipsis.tooltip ? h("div", { class: `${mergedClsPrefix}-data-table-th__ellipsis` }, renderTitle(column)) : ellipsis && typeof ellipsis === "object" ? h(NEllipsis, Object.assign({}, ellipsis, { theme: mergedTheme.peers.Ellipsis, themeOverrides: mergedTheme.peerOverrides.Ellipsis }), {
                default: () => renderTitle(column)
              }) : renderTitle(column)),
              isColumnSortable(column) ? h(SortButton, { column }) : null
            ),
            isColumnFilterable(column) ? h(FilterButton, { column, options: column.filterOptions }) : null,
            isColumnResizable(column) ? h(ResizeButton, { onResizeStart: () => {
              handleColumnResizeStart(column);
            }, onResize: (displacementX) => {
              handleColumnResize(column, displacementX);
            } }) : null
          );
        };
        const leftFixed = key in fixedColumnLeftMap;
        const rightFixed = key in fixedColumnRightMap;
        return h("th", { ref: (el) => cellElsRef[key] = el, key, style: {
          textAlign: column.titleAlign || column.align,
          left: pxfy((_a = fixedColumnLeftMap[key]) === null || _a === void 0 ? void 0 : _a.start),
          right: pxfy((_b = fixedColumnRightMap[key]) === null || _b === void 0 ? void 0 : _b.start)
        }, colspan: colSpan, rowspan: rowSpan, "data-col-key": key, class: [
          `${mergedClsPrefix}-data-table-th`,
          (leftFixed || rightFixed) && `${mergedClsPrefix}-data-table-th--fixed-${leftFixed ? "left" : "right"}`,
          {
            [`${mergedClsPrefix}-data-table-th--hover`]: isColumnSorting(column, mergedSortState),
            [`${mergedClsPrefix}-data-table-th--filterable`]: isColumnFilterable(column),
            [`${mergedClsPrefix}-data-table-th--sortable`]: isColumnSortable(column),
            [`${mergedClsPrefix}-data-table-th--selection`]: column.type === "selection",
            [`${mergedClsPrefix}-data-table-th--last`]: isLast
          },
          column.className
        ], onClick: column.type !== "selection" && column.type !== "expand" && !("children" in column) ? (e) => {
          handleColHeaderClick(e, column);
        } : void 0 }, createColumnVNode());
      }));
    }));
    if (!discrete) {
      return theadVNode;
    }
    const { handleTableHeaderScroll, handleMouseenter, handleMouseleave, scrollX } = this;
    return h(
      "div",
      { class: `${mergedClsPrefix}-data-table-base-table-header`, onScroll: handleTableHeaderScroll, onMouseenter: handleMouseenter, onMouseleave: handleMouseleave },
      h(
        "table",
        { ref: "body", class: `${mergedClsPrefix}-data-table-table`, style: {
          minWidth: formatLength(scrollX),
          tableLayout: mergedTableLayout
        } },
        h("colgroup", null, cols.map((col) => h("col", { key: col.key, style: col.style }))),
        theadVNode
      )
    );
  }
});
const Cell = defineComponent({
  name: "DataTableCell",
  props: {
    clsPrefix: {
      type: String,
      required: true
    },
    row: {
      type: Object,
      required: true
    },
    index: {
      type: Number,
      required: true
    },
    column: {
      type: Object,
      required: true
    },
    isSummary: Boolean,
    mergedTheme: {
      type: Object,
      required: true
    },
    renderCell: Function
  },
  render() {
    const { isSummary, column, row, renderCell } = this;
    let cell;
    const { render: render5, key, ellipsis } = column;
    if (render5 && !isSummary) {
      cell = render5(row, this.index);
    } else {
      if (isSummary) {
        cell = row[key].value;
      } else {
        cell = renderCell ? renderCell(get(row, key), row, column) : get(row, key);
      }
    }
    if (ellipsis) {
      if (typeof ellipsis === "object") {
        const { mergedTheme } = this;
        return h(NEllipsis, Object.assign({}, ellipsis, { theme: mergedTheme.peers.Ellipsis, themeOverrides: mergedTheme.peerOverrides.Ellipsis }), { default: () => cell });
      } else {
        return h("span", { class: `${this.clsPrefix}-data-table-td__ellipsis` }, cell);
      }
    }
    return cell;
  }
});
const ExpandTrigger = defineComponent({
  name: "DataTableExpandTrigger",
  props: {
    clsPrefix: {
      type: String,
      required: true
    },
    expanded: Boolean,
    loading: Boolean,
    onClick: {
      type: Function,
      required: true
    },
    renderExpandIcon: {
      type: Function
    }
  },
  render() {
    const { clsPrefix } = this;
    return h(
      "div",
      { class: [
        `${clsPrefix}-data-table-expand-trigger`,
        this.expanded && `${clsPrefix}-data-table-expand-trigger--expanded`
      ], onClick: this.onClick },
      h(NIconSwitchTransition, null, {
        default: () => {
          return this.loading ? h(NBaseLoading, { key: "loading", clsPrefix: this.clsPrefix, radius: 85, strokeWidth: 15, scale: 0.88 }) : this.renderExpandIcon ? this.renderExpandIcon({
            expanded: this.expanded
          }) : h(NBaseIcon, { clsPrefix, key: "base-icon" }, {
            default: () => h(ChevronRightIcon, null)
          });
        }
      })
    );
  }
});
const RenderSafeCheckbox = defineComponent({
  name: "DataTableBodyCheckbox",
  props: {
    rowKey: {
      type: [String, Number],
      required: true
    },
    disabled: {
      type: Boolean,
      required: true
    },
    onUpdateChecked: {
      type: Function,
      required: true
    }
  },
  setup(props) {
    const {
      mergedCheckedRowKeySetRef,
      mergedInderminateRowKeySetRef
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    } = inject(dataTableInjectionKey);
    return () => {
      const { rowKey } = props;
      return h(NCheckbox, { privateInsideTable: true, disabled: props.disabled, indeterminate: mergedInderminateRowKeySetRef.value.has(rowKey), checked: mergedCheckedRowKeySetRef.value.has(rowKey), onUpdateChecked: props.onUpdateChecked });
    };
  }
});
const RenderSafeRadio = defineComponent({
  name: "DataTableBodyRadio",
  props: {
    rowKey: {
      type: [String, Number],
      required: true
    },
    disabled: {
      type: Boolean,
      required: true
    },
    onUpdateChecked: {
      type: Function,
      required: true
    }
  },
  setup(props) {
    const {
      mergedCheckedRowKeySetRef,
      componentId
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    } = inject(dataTableInjectionKey);
    return () => {
      const { rowKey } = props;
      return h(NRadio, { name: componentId, disabled: props.disabled, checked: mergedCheckedRowKeySetRef.value.has(rowKey), onUpdateChecked: props.onUpdateChecked });
    };
  }
});
function flatten(rowInfos, expandedRowKeys) {
  const fRows = [];
  function traverse(rs, rootIndex) {
    rs.forEach((r) => {
      if (r.children && expandedRowKeys.has(r.key)) {
        fRows.push({
          tmNode: r,
          striped: false,
          key: r.key,
          index: rootIndex
        });
        traverse(r.children, rootIndex);
      } else {
        fRows.push({
          key: r.key,
          tmNode: r,
          striped: false,
          index: rootIndex
        });
      }
    });
  }
  rowInfos.forEach((rowInfo) => {
    fRows.push(rowInfo);
    const { children } = rowInfo.tmNode;
    if (children && expandedRowKeys.has(rowInfo.key)) {
      traverse(children, rowInfo.index);
    }
  });
  return fRows;
}
const VirtualListItemWrapper = defineComponent({
  props: {
    clsPrefix: {
      type: String,
      required: true
    },
    id: {
      type: String,
      required: true
    },
    cols: {
      type: Array,
      required: true
    },
    onMouseenter: Function,
    onMouseleave: Function
  },
  render() {
    const { clsPrefix, id, cols, onMouseenter, onMouseleave } = this;
    return h(
      "table",
      { style: { tableLayout: "fixed" }, class: `${clsPrefix}-data-table-table`, onMouseenter, onMouseleave },
      h("colgroup", null, cols.map((col) => h("col", { key: col.key, style: col.style }))),
      h("tbody", { "data-n-id": id, class: `${clsPrefix}-data-table-tbody` }, this.$slots)
    );
  }
});
const TableBody = defineComponent({
  name: "DataTableBody",
  props: {
    onResize: Function,
    showHeader: Boolean,
    flexHeight: Boolean,
    bodyStyle: Object
  },
  setup(props) {
    const {
      slots: dataTableSlots,
      bodyWidthRef,
      mergedExpandedRowKeysRef,
      mergedClsPrefixRef,
      mergedThemeRef,
      scrollXRef,
      colsRef,
      paginatedDataRef,
      rawPaginatedDataRef,
      fixedColumnLeftMapRef,
      fixedColumnRightMapRef,
      mergedCurrentPageRef,
      rowClassNameRef,
      leftActiveFixedColKeyRef,
      leftActiveFixedChildrenColKeysRef,
      rightActiveFixedColKeyRef,
      rightActiveFixedChildrenColKeysRef,
      renderExpandRef,
      hoverKeyRef,
      summaryRef,
      mergedSortStateRef,
      virtualScrollRef,
      componentId,
      scrollPartRef,
      mergedTableLayoutRef,
      childTriggerColIndexRef,
      indentRef,
      rowPropsRef,
      maxHeightRef,
      stripedRef,
      loadingRef,
      onLoadRef,
      loadingKeySetRef,
      expandableRef,
      stickyExpandedRowsRef,
      renderExpandIconRef,
      summaryPlacementRef,
      treeMateRef,
      scrollbarPropsRef,
      setHeaderScrollLeft,
      doUpdateExpandedRowKeys,
      handleTableBodyScroll,
      doCheck,
      doUncheck,
      renderCell
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    } = inject(dataTableInjectionKey);
    const scrollbarInstRef = ref(null);
    const virtualListRef = ref(null);
    const emptyElRef = ref(null);
    const emptyRef = useMemo(() => paginatedDataRef.value.length === 0);
    const shouldDisplaySomeTablePartRef = useMemo(() => props.showHeader || !emptyRef.value);
    const bodyShowHeaderOnlyRef = useMemo(() => {
      return props.showHeader || emptyRef.value;
    });
    let lastSelectedKey = "";
    const mergedExpandedRowKeySetRef = computed(() => {
      return new Set(mergedExpandedRowKeysRef.value);
    });
    function getRowInfo(key) {
      var _a;
      return (_a = treeMateRef.value.getNode(key)) === null || _a === void 0 ? void 0 : _a.rawNode;
    }
    function handleCheckboxUpdateChecked(tmNode, checked, shiftKey) {
      const rowInfo = getRowInfo(tmNode.key);
      if (!rowInfo) {
        warn("data-table", `fail to get row data with key ${tmNode.key}`);
        return;
      }
      if (shiftKey) {
        const lastIndex = paginatedDataRef.value.findIndex((item) => item.key === lastSelectedKey);
        if (lastIndex !== -1) {
          const currentIndex = paginatedDataRef.value.findIndex((item) => item.key === tmNode.key);
          const start = Math.min(lastIndex, currentIndex);
          const end = Math.max(lastIndex, currentIndex);
          const rowKeysToCheck = [];
          paginatedDataRef.value.slice(start, end + 1).forEach((r) => {
            if (!r.disabled) {
              rowKeysToCheck.push(r.key);
            }
          });
          if (checked) {
            doCheck(rowKeysToCheck, false, rowInfo);
          } else {
            doUncheck(rowKeysToCheck, rowInfo);
          }
          lastSelectedKey = tmNode.key;
          return;
        }
      }
      if (checked) {
        doCheck(tmNode.key, false, rowInfo);
      } else {
        doUncheck(tmNode.key, rowInfo);
      }
      lastSelectedKey = tmNode.key;
    }
    function handleRadioUpdateChecked(tmNode) {
      const rowInfo = getRowInfo(tmNode.key);
      if (!rowInfo) {
        warn("data-table", `fail to get row data with key ${tmNode.key}`);
        return;
      }
      doCheck(tmNode.key, true, rowInfo);
    }
    function getScrollContainer() {
      if (!shouldDisplaySomeTablePartRef.value) {
        const { value: emptyEl } = emptyElRef;
        if (emptyEl) {
          return emptyEl;
        } else {
          return null;
        }
      }
      if (virtualScrollRef.value) {
        return virtualListContainer();
      }
      const { value } = scrollbarInstRef;
      if (value)
        return value.containerRef;
      return null;
    }
    function handleUpdateExpanded(key, tmNode) {
      var _a;
      if (loadingKeySetRef.value.has(key))
        return;
      const { value: mergedExpandedRowKeys } = mergedExpandedRowKeysRef;
      const index = mergedExpandedRowKeys.indexOf(key);
      const nextExpandedKeys = Array.from(mergedExpandedRowKeys);
      if (~index) {
        nextExpandedKeys.splice(index, 1);
        doUpdateExpandedRowKeys(nextExpandedKeys);
      } else {
        if (tmNode && !tmNode.isLeaf && !tmNode.shallowLoaded) {
          loadingKeySetRef.value.add(key);
          void ((_a = onLoadRef.value) === null || _a === void 0 ? void 0 : _a.call(onLoadRef, tmNode.rawNode).then(() => {
            const { value: futureMergedExpandedRowKeys } = mergedExpandedRowKeysRef;
            const futureNextExpandedKeys = Array.from(futureMergedExpandedRowKeys);
            const index2 = futureNextExpandedKeys.indexOf(key);
            if (!~index2) {
              futureNextExpandedKeys.push(key);
            }
            doUpdateExpandedRowKeys(futureNextExpandedKeys);
          }).finally(() => {
            loadingKeySetRef.value.delete(key);
          }));
        } else {
          nextExpandedKeys.push(key);
          doUpdateExpandedRowKeys(nextExpandedKeys);
        }
      }
    }
    function handleMouseleaveTable() {
      hoverKeyRef.value = null;
    }
    function handleMouseenterTable() {
      scrollPartRef.value = "body";
    }
    function virtualListContainer() {
      const { value } = virtualListRef;
      return value === null || value === void 0 ? void 0 : value.listElRef;
    }
    function virtualListContent() {
      const { value } = virtualListRef;
      return value === null || value === void 0 ? void 0 : value.itemsElRef;
    }
    function handleVirtualListScroll(e) {
      var _a;
      handleTableBodyScroll(e);
      (_a = scrollbarInstRef.value) === null || _a === void 0 ? void 0 : _a.sync();
    }
    function handleVirtualListResize(e) {
      var _a;
      const { onResize } = props;
      if (onResize)
        onResize(e);
      (_a = scrollbarInstRef.value) === null || _a === void 0 ? void 0 : _a.sync();
    }
    const exposedMethods = {
      getScrollContainer,
      scrollTo(arg0, arg1) {
        var _a, _b;
        if (virtualScrollRef.value) {
          (_a = virtualListRef.value) === null || _a === void 0 ? void 0 : _a.scrollTo(arg0, arg1);
        } else {
          (_b = scrollbarInstRef.value) === null || _b === void 0 ? void 0 : _b.scrollTo(arg0, arg1);
        }
      }
    };
    const style2 = c([
      ({ props: cProps }) => {
        const createActiveLeftFixedStyle = (leftActiveFixedColKey) => {
          if (leftActiveFixedColKey === null)
            return null;
          return c(`[data-n-id="${cProps.componentId}"] [data-col-key="${leftActiveFixedColKey}"]::after`, { boxShadow: "var(--n-box-shadow-after)" });
        };
        const createActiveRightFixedStyle = (rightActiveFixedColKey) => {
          if (rightActiveFixedColKey === null)
            return null;
          return c(`[data-n-id="${cProps.componentId}"] [data-col-key="${rightActiveFixedColKey}"]::before`, { boxShadow: "var(--n-box-shadow-before)" });
        };
        return c([
          createActiveLeftFixedStyle(cProps.leftActiveFixedColKey),
          createActiveRightFixedStyle(cProps.rightActiveFixedColKey),
          cProps.leftActiveFixedChildrenColKeys.map((leftActiveFixedColKey) => createActiveLeftFixedStyle(leftActiveFixedColKey)),
          cProps.rightActiveFixedChildrenColKeys.map((rightActiveFixedColKey) => createActiveRightFixedStyle(rightActiveFixedColKey))
        ]);
      }
    ]);
    let fixedStyleMounted = false;
    watchEffect(() => {
      const { value: leftActiveFixedColKey } = leftActiveFixedColKeyRef;
      const { value: leftActiveFixedChildrenColKeys } = leftActiveFixedChildrenColKeysRef;
      const { value: rightActiveFixedColKey } = rightActiveFixedColKeyRef;
      const { value: rightActiveFixedChildrenColKeys } = rightActiveFixedChildrenColKeysRef;
      if (!fixedStyleMounted && leftActiveFixedColKey === null && rightActiveFixedColKey === null) {
        return;
      }
      const cProps = {
        leftActiveFixedColKey,
        leftActiveFixedChildrenColKeys,
        rightActiveFixedColKey,
        rightActiveFixedChildrenColKeys,
        componentId
      };
      style2.mount({
        id: `n-${componentId}`,
        force: true,
        props: cProps,
        anchorMetaName: cssrAnchorMetaName
      });
      fixedStyleMounted = true;
    });
    onUnmounted(() => {
      style2.unmount({
        id: `n-${componentId}`
      });
    });
    return Object.assign({
      bodyWidth: bodyWidthRef,
      summaryPlacement: summaryPlacementRef,
      dataTableSlots,
      componentId,
      scrollbarInstRef,
      virtualListRef,
      emptyElRef,
      summary: summaryRef,
      mergedClsPrefix: mergedClsPrefixRef,
      mergedTheme: mergedThemeRef,
      scrollX: scrollXRef,
      cols: colsRef,
      loading: loadingRef,
      bodyShowHeaderOnly: bodyShowHeaderOnlyRef,
      shouldDisplaySomeTablePart: shouldDisplaySomeTablePartRef,
      empty: emptyRef,
      paginatedDataAndInfo: computed(() => {
        const { value: striped } = stripedRef;
        let hasChildren = false;
        const data = paginatedDataRef.value.map(striped ? (tmNode, index) => {
          if (!tmNode.isLeaf)
            hasChildren = true;
          return {
            tmNode,
            key: tmNode.key,
            striped: index % 2 === 1,
            index
          };
        } : (tmNode, index) => {
          if (!tmNode.isLeaf)
            hasChildren = true;
          return {
            tmNode,
            key: tmNode.key,
            striped: false,
            index
          };
        });
        return {
          data,
          hasChildren
        };
      }),
      rawPaginatedData: rawPaginatedDataRef,
      fixedColumnLeftMap: fixedColumnLeftMapRef,
      fixedColumnRightMap: fixedColumnRightMapRef,
      currentPage: mergedCurrentPageRef,
      rowClassName: rowClassNameRef,
      renderExpand: renderExpandRef,
      mergedExpandedRowKeySet: mergedExpandedRowKeySetRef,
      hoverKey: hoverKeyRef,
      mergedSortState: mergedSortStateRef,
      virtualScroll: virtualScrollRef,
      mergedTableLayout: mergedTableLayoutRef,
      childTriggerColIndex: childTriggerColIndexRef,
      indent: indentRef,
      rowProps: rowPropsRef,
      maxHeight: maxHeightRef,
      loadingKeySet: loadingKeySetRef,
      expandable: expandableRef,
      stickyExpandedRows: stickyExpandedRowsRef,
      renderExpandIcon: renderExpandIconRef,
      scrollbarProps: scrollbarPropsRef,
      setHeaderScrollLeft,
      handleMouseenterTable,
      handleVirtualListScroll,
      handleVirtualListResize,
      handleMouseleaveTable,
      virtualListContainer,
      virtualListContent,
      handleTableBodyScroll,
      handleCheckboxUpdateChecked,
      handleRadioUpdateChecked,
      handleUpdateExpanded,
      renderCell
    }, exposedMethods);
  },
  render() {
    const { mergedTheme, scrollX, mergedClsPrefix, virtualScroll, maxHeight, mergedTableLayout, flexHeight, loadingKeySet, onResize, setHeaderScrollLeft } = this;
    const scrollable = scrollX !== void 0 || maxHeight !== void 0 || flexHeight;
    const isBasicAutoLayout = !scrollable && mergedTableLayout === "auto";
    const xScrollable = scrollX !== void 0 || isBasicAutoLayout;
    const contentStyle = {
      minWidth: formatLength(scrollX) || "100%"
    };
    if (scrollX)
      contentStyle.width = "100%";
    const tableNode = h(NScrollbar, Object.assign({}, this.scrollbarProps, { ref: "scrollbarInstRef", scrollable: scrollable || isBasicAutoLayout, class: `${mergedClsPrefix}-data-table-base-table-body`, style: this.bodyStyle, theme: mergedTheme.peers.Scrollbar, themeOverrides: mergedTheme.peerOverrides.Scrollbar, contentStyle, container: virtualScroll ? this.virtualListContainer : void 0, content: virtualScroll ? this.virtualListContent : void 0, horizontalRailStyle: { zIndex: 3 }, verticalRailStyle: { zIndex: 3 }, xScrollable, onScroll: virtualScroll ? void 0 : this.handleTableBodyScroll, internalOnUpdateScrollLeft: setHeaderScrollLeft, onResize }), {
      default: () => {
        const cordToPass = {};
        const cordKey = {};
        const { cols, paginatedDataAndInfo, mergedTheme: mergedTheme2, fixedColumnLeftMap, fixedColumnRightMap, currentPage, rowClassName, mergedSortState, mergedExpandedRowKeySet, stickyExpandedRows, componentId, childTriggerColIndex, expandable, rowProps, handleMouseenterTable, handleMouseleaveTable, renderExpand, summary, handleCheckboxUpdateChecked, handleRadioUpdateChecked, handleUpdateExpanded } = this;
        const { length: colCount } = cols;
        let mergedData;
        const { data: paginatedData, hasChildren } = paginatedDataAndInfo;
        const mergedPaginationData = hasChildren ? flatten(paginatedData, mergedExpandedRowKeySet) : paginatedData;
        if (summary) {
          const summaryRows = summary(this.rawPaginatedData);
          if (Array.isArray(summaryRows)) {
            const summaryRowData = summaryRows.map((row, i) => ({
              isSummaryRow: true,
              key: `__n_summary__${i}`,
              tmNode: {
                rawNode: row,
                disabled: true
              },
              index: -1
            }));
            mergedData = this.summaryPlacement === "top" ? [...summaryRowData, ...mergedPaginationData] : [...mergedPaginationData, ...summaryRowData];
          } else {
            const summaryRowData = {
              isSummaryRow: true,
              key: "__n_summary__",
              tmNode: {
                rawNode: summaryRows,
                disabled: true
              },
              index: -1
            };
            mergedData = this.summaryPlacement === "top" ? [summaryRowData, ...mergedPaginationData] : [...mergedPaginationData, summaryRowData];
          }
        } else {
          mergedData = mergedPaginationData;
        }
        const indentStyle = hasChildren ? { width: pxfy(this.indent) } : void 0;
        const displayedData = [];
        mergedData.forEach((rowInfo) => {
          if (renderExpand && mergedExpandedRowKeySet.has(rowInfo.key) && (!expandable || expandable(rowInfo.tmNode.rawNode))) {
            displayedData.push(rowInfo, {
              isExpandedRow: true,
              key: `${rowInfo.key}-expand`,
              tmNode: rowInfo.tmNode,
              index: rowInfo.index
            });
          } else {
            displayedData.push(rowInfo);
          }
        });
        const { length: rowCount } = displayedData;
        const rowIndexToKey = {};
        paginatedData.forEach(({ tmNode }, rowIndex) => {
          rowIndexToKey[rowIndex] = tmNode.key;
        });
        const bodyWidth = stickyExpandedRows ? this.bodyWidth : null;
        const bodyWidthPx = bodyWidth === null ? void 0 : `${bodyWidth}px`;
        const renderRow = (rowInfo, displayedRowIndex, isVirtual) => {
          const { index: actualRowIndex } = rowInfo;
          if ("isExpandedRow" in rowInfo) {
            const { tmNode: { key, rawNode } } = rowInfo;
            return h(
              "tr",
              { class: `${mergedClsPrefix}-data-table-tr`, key: `${key}__expand` },
              h("td", { class: [
                `${mergedClsPrefix}-data-table-td`,
                `${mergedClsPrefix}-data-table-td--last-col`,
                displayedRowIndex + 1 === rowCount && `${mergedClsPrefix}-data-table-td--last-row`
              ], colspan: colCount }, stickyExpandedRows ? h("div", { class: `${mergedClsPrefix}-data-table-expand`, style: {
                width: bodyWidthPx
              } }, renderExpand(rawNode, actualRowIndex)) : renderExpand(rawNode, actualRowIndex))
            );
          }
          const isSummary = "isSummaryRow" in rowInfo;
          const striped = !isSummary && rowInfo.striped;
          const { tmNode, key: rowKey } = rowInfo;
          const { rawNode: rowData } = tmNode;
          const expanded = mergedExpandedRowKeySet.has(rowKey);
          const props = rowProps ? rowProps(rowData, actualRowIndex) : void 0;
          const mergedRowClassName = typeof rowClassName === "string" ? rowClassName : createRowClassName(rowData, actualRowIndex, rowClassName);
          const row = h("tr", Object.assign({ onMouseenter: () => {
            this.hoverKey = rowKey;
          }, key: rowKey, class: [
            `${mergedClsPrefix}-data-table-tr`,
            isSummary && `${mergedClsPrefix}-data-table-tr--summary`,
            striped && `${mergedClsPrefix}-data-table-tr--striped`,
            mergedRowClassName
          ] }, props), cols.map((col, colIndex) => {
            var _a, _b, _c, _d, _e;
            if (displayedRowIndex in cordToPass) {
              const cordOfRowToPass = cordToPass[displayedRowIndex];
              const indexInCordOfRowToPass = cordOfRowToPass.indexOf(colIndex);
              if (~indexInCordOfRowToPass) {
                cordOfRowToPass.splice(indexInCordOfRowToPass, 1);
                return null;
              }
            }
            const { column } = col;
            const colKey = getColKey(col);
            const { rowSpan, colSpan } = column;
            const mergedColSpan = isSummary ? ((_a = rowInfo.tmNode.rawNode[colKey]) === null || _a === void 0 ? void 0 : _a.colSpan) || 1 : colSpan ? colSpan(rowData, actualRowIndex) : 1;
            const mergedRowSpan = isSummary ? ((_b = rowInfo.tmNode.rawNode[colKey]) === null || _b === void 0 ? void 0 : _b.rowSpan) || 1 : rowSpan ? rowSpan(rowData, actualRowIndex) : 1;
            const isLastCol = colIndex + mergedColSpan === colCount;
            const isLastRow = displayedRowIndex + mergedRowSpan === rowCount;
            const isCrossRowTd = mergedRowSpan > 1;
            if (isCrossRowTd) {
              cordKey[displayedRowIndex] = {
                [colIndex]: []
              };
            }
            if (mergedColSpan > 1 || isCrossRowTd) {
              for (let i = displayedRowIndex; i < displayedRowIndex + mergedRowSpan; ++i) {
                if (isCrossRowTd) {
                  cordKey[displayedRowIndex][colIndex].push(rowIndexToKey[i]);
                }
                for (let j = colIndex; j < colIndex + mergedColSpan; ++j) {
                  if (i === displayedRowIndex && j === colIndex) {
                    continue;
                  }
                  if (!(i in cordToPass)) {
                    cordToPass[i] = [j];
                  } else {
                    cordToPass[i].push(j);
                  }
                }
              }
            }
            const hoverKey = isCrossRowTd ? this.hoverKey : null;
            const { cellProps } = column;
            const resolvedCellProps = cellProps === null || cellProps === void 0 ? void 0 : cellProps(rowData, actualRowIndex);
            return h(
              "td",
              Object.assign({}, resolvedCellProps, { key: colKey, style: [
                {
                  textAlign: column.align || void 0,
                  left: pxfy((_c = fixedColumnLeftMap[colKey]) === null || _c === void 0 ? void 0 : _c.start),
                  right: pxfy((_d = fixedColumnRightMap[colKey]) === null || _d === void 0 ? void 0 : _d.start)
                },
                (resolvedCellProps === null || resolvedCellProps === void 0 ? void 0 : resolvedCellProps.style) || ""
              ], colspan: mergedColSpan, rowspan: isVirtual ? void 0 : mergedRowSpan, "data-col-key": colKey, class: [
                `${mergedClsPrefix}-data-table-td`,
                column.className,
                resolvedCellProps === null || resolvedCellProps === void 0 ? void 0 : resolvedCellProps.class,
                isSummary && `${mergedClsPrefix}-data-table-td--summary`,
                (hoverKey !== null && cordKey[displayedRowIndex][colIndex].includes(hoverKey) || isColumnSorting(column, mergedSortState)) && `${mergedClsPrefix}-data-table-td--hover`,
                column.fixed && `${mergedClsPrefix}-data-table-td--fixed-${column.fixed}`,
                column.align && `${mergedClsPrefix}-data-table-td--${column.align}-align`,
                column.type === "selection" && `${mergedClsPrefix}-data-table-td--selection`,
                column.type === "expand" && `${mergedClsPrefix}-data-table-td--expand`,
                isLastCol && `${mergedClsPrefix}-data-table-td--last-col`,
                isLastRow && `${mergedClsPrefix}-data-table-td--last-row`
              ] }),
              hasChildren && colIndex === childTriggerColIndex ? [
                repeat(isSummary ? 0 : rowInfo.tmNode.level, h("div", { class: `${mergedClsPrefix}-data-table-indent`, style: indentStyle })),
                isSummary || rowInfo.tmNode.isLeaf ? h("div", { class: `${mergedClsPrefix}-data-table-expand-placeholder` }) : h(ExpandTrigger, { class: `${mergedClsPrefix}-data-table-expand-trigger`, clsPrefix: mergedClsPrefix, expanded, renderExpandIcon: this.renderExpandIcon, loading: loadingKeySet.has(rowInfo.key), onClick: () => {
                  handleUpdateExpanded(rowKey, rowInfo.tmNode);
                } })
              ] : null,
              column.type === "selection" ? !isSummary ? column.multiple === false ? h(RenderSafeRadio, { key: currentPage, rowKey, disabled: rowInfo.tmNode.disabled, onUpdateChecked: () => {
                handleRadioUpdateChecked(rowInfo.tmNode);
              } }) : h(RenderSafeCheckbox, { key: currentPage, rowKey, disabled: rowInfo.tmNode.disabled, onUpdateChecked: (checked, e) => {
                handleCheckboxUpdateChecked(rowInfo.tmNode, checked, e.shiftKey);
              } }) : null : column.type === "expand" ? !isSummary ? !column.expandable || ((_e = column.expandable) === null || _e === void 0 ? void 0 : _e.call(column, rowData)) ? h(ExpandTrigger, { clsPrefix: mergedClsPrefix, expanded, renderExpandIcon: this.renderExpandIcon, onClick: () => {
                handleUpdateExpanded(rowKey, null);
              } }) : null : null : h(Cell, { clsPrefix: mergedClsPrefix, index: actualRowIndex, row: rowData, column, isSummary, mergedTheme: mergedTheme2, renderCell: this.renderCell })
            );
          }));
          return row;
        };
        if (!virtualScroll) {
          return h(
            "table",
            { class: `${mergedClsPrefix}-data-table-table`, onMouseleave: handleMouseleaveTable, onMouseenter: handleMouseenterTable, style: {
              tableLayout: this.mergedTableLayout
            } },
            h("colgroup", null, cols.map((col) => h("col", { key: col.key, style: col.style }))),
            this.showHeader ? h(TableHeader, { discrete: false }) : null,
            !this.empty ? h("tbody", { "data-n-id": componentId, class: `${mergedClsPrefix}-data-table-tbody` }, displayedData.map((rowInfo, displayedRowIndex) => {
              return renderRow(rowInfo, displayedRowIndex, false);
            })) : null
          );
        } else {
          return h(VVirtualList, { ref: "virtualListRef", items: displayedData, itemSize: 28, visibleItemsTag: VirtualListItemWrapper, visibleItemsProps: {
            clsPrefix: mergedClsPrefix,
            id: componentId,
            cols,
            onMouseenter: handleMouseenterTable,
            onMouseleave: handleMouseleaveTable
          }, showScrollbar: false, onResize: this.handleVirtualListResize, onScroll: this.handleVirtualListScroll, itemsStyle: contentStyle, itemResizable: true }, {
            default: ({ item, index }) => renderRow(item, index, true)
          });
        }
      }
    });
    if (this.empty) {
      const createEmptyNode = () => h("div", { class: [
        `${mergedClsPrefix}-data-table-empty`,
        this.loading && `${mergedClsPrefix}-data-table-empty--hide`
      ], style: this.bodyStyle, ref: "emptyElRef" }, resolveSlot(this.dataTableSlots.empty, () => [
        h(NEmpty, { theme: this.mergedTheme.peers.Empty, themeOverrides: this.mergedTheme.peerOverrides.Empty })
      ]));
      if (this.shouldDisplaySomeTablePart) {
        return h(
          Fragment,
          null,
          tableNode,
          createEmptyNode()
        );
      } else {
        return h(VResizeObserver, { onResize: this.onResize }, { default: createEmptyNode });
      }
    }
    return tableNode;
  }
});
const MainTable = defineComponent({
  setup() {
    const {
      mergedClsPrefixRef,
      rightFixedColumnsRef,
      leftFixedColumnsRef,
      bodyWidthRef,
      maxHeightRef,
      minHeightRef,
      flexHeightRef,
      syncScrollState
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    } = inject(dataTableInjectionKey);
    const headerInstRef = ref(null);
    const bodyInstRef = ref(null);
    const selfElRef = ref(null);
    const fixedStateInitializedRef = ref(!(leftFixedColumnsRef.value.length || rightFixedColumnsRef.value.length));
    const bodyStyleRef = computed(() => {
      return {
        maxHeight: formatLength(maxHeightRef.value),
        minHeight: formatLength(minHeightRef.value)
      };
    });
    function handleBodyResize(entry) {
      bodyWidthRef.value = entry.contentRect.width;
      syncScrollState();
      if (!fixedStateInitializedRef.value) {
        fixedStateInitializedRef.value = true;
      }
    }
    function getHeaderElement() {
      const { value } = headerInstRef;
      if (value) {
        return value.$el;
      }
      return null;
    }
    function getBodyElement() {
      const { value } = bodyInstRef;
      if (value) {
        return value.getScrollContainer();
      }
      return null;
    }
    const exposedMethods = {
      getBodyElement,
      getHeaderElement,
      scrollTo(arg0, arg1) {
        var _a;
        (_a = bodyInstRef.value) === null || _a === void 0 ? void 0 : _a.scrollTo(arg0, arg1);
      }
    };
    watchEffect(() => {
      const { value: selfEl } = selfElRef;
      if (!selfEl)
        return;
      const transitionDisabledClass = `${mergedClsPrefixRef.value}-data-table-base-table--transition-disabled`;
      if (fixedStateInitializedRef.value) {
        setTimeout(() => {
          selfEl.classList.remove(transitionDisabledClass);
        }, 0);
      } else {
        selfEl.classList.add(transitionDisabledClass);
      }
    });
    return Object.assign({
      maxHeight: maxHeightRef,
      mergedClsPrefix: mergedClsPrefixRef,
      selfElRef,
      headerInstRef,
      bodyInstRef,
      bodyStyle: bodyStyleRef,
      flexHeight: flexHeightRef,
      handleBodyResize
    }, exposedMethods);
  },
  render() {
    const { mergedClsPrefix, maxHeight, flexHeight } = this;
    const headerInBody = maxHeight === void 0 && !flexHeight;
    return h(
      "div",
      { class: `${mergedClsPrefix}-data-table-base-table`, ref: "selfElRef" },
      headerInBody ? null : h(TableHeader, { ref: "headerInstRef" }),
      h(TableBody, { ref: "bodyInstRef", bodyStyle: this.bodyStyle, showHeader: headerInBody, flexHeight, onResize: this.handleBodyResize })
    );
  }
});
function useCheck(props, data) {
  const { paginatedDataRef, treeMateRef, selectionColumnRef } = data;
  const uncontrolledCheckedRowKeysRef = ref(props.defaultCheckedRowKeys);
  const mergedCheckState = computed(() => {
    var _a;
    const { checkedRowKeys } = props;
    const sourceKeys = checkedRowKeys === void 0 ? uncontrolledCheckedRowKeysRef.value : checkedRowKeys;
    if (((_a = selectionColumnRef.value) === null || _a === void 0 ? void 0 : _a.multiple) === false) {
      return {
        checkedKeys: sourceKeys.slice(0, 1),
        indeterminateKeys: []
      };
    }
    return treeMateRef.value.getCheckedKeys(sourceKeys, {
      cascade: props.cascade,
      allowNotLoaded: props.allowCheckingNotLoaded
    });
  });
  const mergedCheckedRowKeysRef = computed(() => mergedCheckState.value.checkedKeys);
  const mergedInderminateRowKeysRef = computed(() => mergedCheckState.value.indeterminateKeys);
  const mergedCheckedRowKeySetRef = computed(() => {
    return new Set(mergedCheckedRowKeysRef.value);
  });
  const mergedInderminateRowKeySetRef = computed(() => {
    return new Set(mergedInderminateRowKeysRef.value);
  });
  const countOfCurrentPageCheckedRowsRef = computed(() => {
    const { value: mergedCheckedRowKeySet } = mergedCheckedRowKeySetRef;
    return paginatedDataRef.value.reduce((total, tmNode) => {
      const { key, disabled } = tmNode;
      return total + (!disabled && mergedCheckedRowKeySet.has(key) ? 1 : 0);
    }, 0);
  });
  const countOfCurrentPageDisabledRowsRef = computed(() => {
    return paginatedDataRef.value.filter((item) => item.disabled).length;
  });
  const someRowsCheckedRef = computed(() => {
    const { length } = paginatedDataRef.value;
    const { value: mergedInderminateRowKeySet } = mergedInderminateRowKeySetRef;
    return countOfCurrentPageCheckedRowsRef.value > 0 && countOfCurrentPageCheckedRowsRef.value < length - countOfCurrentPageDisabledRowsRef.value || paginatedDataRef.value.some((rowData) => mergedInderminateRowKeySet.has(rowData.key));
  });
  const allRowsCheckedRef = computed(() => {
    const { length } = paginatedDataRef.value;
    return countOfCurrentPageCheckedRowsRef.value !== 0 && countOfCurrentPageCheckedRowsRef.value === length - countOfCurrentPageDisabledRowsRef.value;
  });
  const headerCheckboxDisabledRef = computed(() => {
    return paginatedDataRef.value.length === 0;
  });
  function doUpdateCheckedRowKeys(keys, row, action) {
    const { "onUpdate:checkedRowKeys": _onUpdateCheckedRowKeys, onUpdateCheckedRowKeys, onCheckedRowKeysChange } = props;
    const rows = [];
    const { value: { getNode } } = treeMateRef;
    keys.forEach((key) => {
      var _a;
      const row2 = (_a = getNode(key)) === null || _a === void 0 ? void 0 : _a.rawNode;
      rows.push(row2);
    });
    if (_onUpdateCheckedRowKeys) {
      call(_onUpdateCheckedRowKeys, keys, rows, { row, action });
    }
    if (onUpdateCheckedRowKeys) {
      call(onUpdateCheckedRowKeys, keys, rows, { row, action });
    }
    if (onCheckedRowKeysChange) {
      call(onCheckedRowKeysChange, keys, rows, { row, action });
    }
    uncontrolledCheckedRowKeysRef.value = keys;
  }
  function doCheck(rowKey, single = false, rowInfo) {
    if (props.loading)
      return;
    if (single) {
      doUpdateCheckedRowKeys(Array.isArray(rowKey) ? rowKey.slice(0, 1) : [rowKey], rowInfo, "check");
      return;
    }
    doUpdateCheckedRowKeys(treeMateRef.value.check(rowKey, mergedCheckedRowKeysRef.value, {
      cascade: props.cascade,
      allowNotLoaded: props.allowCheckingNotLoaded
    }).checkedKeys, rowInfo, "check");
  }
  function doUncheck(rowKey, rowInfo) {
    if (props.loading)
      return;
    doUpdateCheckedRowKeys(treeMateRef.value.uncheck(rowKey, mergedCheckedRowKeysRef.value, {
      cascade: props.cascade,
      allowNotLoaded: props.allowCheckingNotLoaded
    }).checkedKeys, rowInfo, "uncheck");
  }
  function doCheckAll(checkWholeTable = false) {
    const { value: column } = selectionColumnRef;
    if (!column || props.loading)
      return;
    const rowKeysToCheck = [];
    (checkWholeTable ? treeMateRef.value.treeNodes : paginatedDataRef.value).forEach((tmNode) => {
      if (!tmNode.disabled) {
        rowKeysToCheck.push(tmNode.key);
      }
    });
    doUpdateCheckedRowKeys(treeMateRef.value.check(rowKeysToCheck, mergedCheckedRowKeysRef.value, {
      cascade: true,
      allowNotLoaded: props.allowCheckingNotLoaded
    }).checkedKeys, void 0, "checkAll");
  }
  function doUncheckAll(checkWholeTable = false) {
    const { value: column } = selectionColumnRef;
    if (!column || props.loading)
      return;
    const rowKeysToUncheck = [];
    (checkWholeTable ? treeMateRef.value.treeNodes : paginatedDataRef.value).forEach((tmNode) => {
      if (!tmNode.disabled) {
        rowKeysToUncheck.push(tmNode.key);
      }
    });
    doUpdateCheckedRowKeys(treeMateRef.value.uncheck(rowKeysToUncheck, mergedCheckedRowKeysRef.value, {
      cascade: true,
      allowNotLoaded: props.allowCheckingNotLoaded
    }).checkedKeys, void 0, "uncheckAll");
  }
  return {
    mergedCheckedRowKeySetRef,
    mergedCheckedRowKeysRef,
    mergedInderminateRowKeySetRef,
    someRowsCheckedRef,
    allRowsCheckedRef,
    headerCheckboxDisabledRef,
    doUpdateCheckedRowKeys,
    doCheckAll,
    doUncheckAll,
    doCheck,
    doUncheck
  };
}
function getMultiplePriority(sorter) {
  if (typeof sorter === "object" && typeof sorter.multiple === "number") {
    return sorter.multiple;
  }
  return false;
}
function getSortFunction(sorter, columnKey) {
  if (columnKey && (sorter === void 0 || sorter === "default" || typeof sorter === "object" && sorter.compare === "default")) {
    return getDefaultSorterFn(columnKey);
  }
  if (typeof sorter === "function") {
    return sorter;
  }
  if (sorter && typeof sorter === "object" && sorter.compare && sorter.compare !== "default") {
    return sorter.compare;
  }
  return false;
}
function getDefaultSorterFn(columnKey) {
  return (row1, row2) => {
    const value1 = row1[columnKey];
    const value2 = row2[columnKey];
    if (typeof value1 === "number" && typeof value2 === "number") {
      return value1 - value2;
    } else if (typeof value1 === "string" && typeof value2 === "string") {
      return value1.localeCompare(value2);
    }
    return 0;
  };
}
function useSorter(props, { dataRelatedColsRef, filteredDataRef }) {
  const defaultSortState = [];
  dataRelatedColsRef.value.forEach((column) => {
    var _a;
    if (column.sorter !== void 0) {
      updateSortStatesByNewSortState(defaultSortState, {
        columnKey: column.key,
        sorter: column.sorter,
        order: (_a = column.defaultSortOrder) !== null && _a !== void 0 ? _a : false
      });
    }
  });
  const uncontrolledSortStateRef = ref(defaultSortState);
  const mergedSortStateRef = computed(() => {
    const columnsWithControlledSortOrder = dataRelatedColsRef.value.filter((column) => column.type !== "selection" && column.sorter !== void 0 && (column.sortOrder === "ascend" || column.sortOrder === "descend" || column.sortOrder === false));
    const columnToSort = columnsWithControlledSortOrder.filter((col) => col.sortOrder !== false);
    if (columnToSort.length) {
      return columnToSort.map((column) => {
        return {
          columnKey: column.key,
          // column to sort has controlled sorter
          // sorter && sort order won't be undefined
          // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
          order: column.sortOrder,
          // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
          sorter: column.sorter
        };
      });
    }
    if (columnsWithControlledSortOrder.length)
      return [];
    const { value: uncontrolledSortState } = uncontrolledSortStateRef;
    if (Array.isArray(uncontrolledSortState)) {
      return uncontrolledSortState;
    } else if (uncontrolledSortState) {
      return [uncontrolledSortState];
    } else {
      return [];
    }
  });
  const sortedDataRef = computed(() => {
    const activeSorters = mergedSortStateRef.value.slice().sort((a, b) => {
      const item1Priority = getMultiplePriority(a.sorter) || 0;
      const item2Priority = getMultiplePriority(b.sorter) || 0;
      return item2Priority - item1Priority;
    });
    if (activeSorters.length) {
      const filteredData = filteredDataRef.value.slice();
      return filteredData.sort((tmNode1, tmNode2) => {
        let compareResult = 0;
        activeSorters.some((sorterState) => {
          const { columnKey, sorter, order } = sorterState;
          const compareFn = getSortFunction(sorter, columnKey);
          if (compareFn && order) {
            compareResult = compareFn(tmNode1.rawNode, tmNode2.rawNode);
            if (compareResult !== 0) {
              compareResult = compareResult * getFlagOfOrder(order);
              return true;
            }
          }
          return false;
        });
        return compareResult;
      });
    }
    return filteredDataRef.value;
  });
  function getUpdatedSorterState(sortState) {
    let currentSortState = mergedSortStateRef.value.slice();
    if (sortState && getMultiplePriority(sortState.sorter) !== false) {
      currentSortState = currentSortState.filter((sortState2) => getMultiplePriority(sortState2.sorter) !== false);
      updateSortStatesByNewSortState(currentSortState, sortState);
      return currentSortState;
    } else if (sortState) {
      return sortState;
    }
    return null;
  }
  function deriveNextSorter(sortState) {
    const nextSorterState = getUpdatedSorterState(sortState);
    doUpdateSorter(nextSorterState);
  }
  function doUpdateSorter(sortState) {
    const { "onUpdate:sorter": _onUpdateSorter, onUpdateSorter, onSorterChange } = props;
    if (_onUpdateSorter) {
      call(_onUpdateSorter, sortState);
    }
    if (onUpdateSorter) {
      call(onUpdateSorter, sortState);
    }
    if (onSorterChange) {
      call(onSorterChange, sortState);
    }
    uncontrolledSortStateRef.value = sortState;
  }
  function sort(columnKey, order = "ascend") {
    if (!columnKey) {
      clearSorter();
    } else {
      const columnToSort = dataRelatedColsRef.value.find((column) => column.type !== "selection" && column.type !== "expand" && column.key === columnKey);
      if (!(columnToSort === null || columnToSort === void 0 ? void 0 : columnToSort.sorter))
        return;
      const sorter = columnToSort.sorter;
      deriveNextSorter({
        columnKey,
        sorter,
        order
      });
    }
  }
  function clearSorter() {
    doUpdateSorter(null);
  }
  function updateSortStatesByNewSortState(sortStates, sortState) {
    const index = sortStates.findIndex((state) => (sortState === null || sortState === void 0 ? void 0 : sortState.columnKey) && state.columnKey === sortState.columnKey);
    if (index !== void 0 && index >= 0) {
      sortStates[index] = sortState;
    } else {
      sortStates.push(sortState);
    }
  }
  return {
    clearSorter,
    sort,
    sortedDataRef,
    mergedSortStateRef,
    deriveNextSorter
  };
}
function useTableData(props, { dataRelatedColsRef }) {
  const selectionColumnRef = computed(() => {
    const getSelectionColumn = (cols) => {
      for (let i = 0; i < cols.length; ++i) {
        const col = cols[i];
        if ("children" in col) {
          return getSelectionColumn(col.children);
        } else if (col.type === "selection") {
          return col;
        }
      }
      return null;
    };
    return getSelectionColumn(props.columns);
  });
  const treeMateRef = computed(() => {
    const { childrenKey } = props;
    return createTreeMate(props.data, {
      ignoreEmptyChildren: true,
      getKey: props.rowKey,
      getChildren: (rowData) => rowData[childrenKey],
      getDisabled: (rowData) => {
        var _a, _b;
        if ((_b = (_a = selectionColumnRef.value) === null || _a === void 0 ? void 0 : _a.disabled) === null || _b === void 0 ? void 0 : _b.call(_a, rowData)) {
          return true;
        }
        return false;
      }
    });
  });
  const childTriggerColIndexRef = useMemo(() => {
    const { columns } = props;
    const { length } = columns;
    let firstContentfulColIndex = null;
    for (let i = 0; i < length; ++i) {
      const col = columns[i];
      if (!col.type && firstContentfulColIndex === null) {
        firstContentfulColIndex = i;
      }
      if ("tree" in col && col.tree) {
        return i;
      }
    }
    return firstContentfulColIndex || 0;
  });
  const uncontrolledFilterStateRef = ref({});
  const uncontrolledCurrentPageRef = ref(1);
  const uncontrolledPageSizeRef = ref(10);
  const mergedFilterStateRef = computed(() => {
    const columnsWithControlledFilter = dataRelatedColsRef.value.filter((column) => {
      return column.filterOptionValues !== void 0 || column.filterOptionValue !== void 0;
    });
    const controlledFilterState = {};
    columnsWithControlledFilter.forEach((column) => {
      var _a;
      if (column.type === "selection" || column.type === "expand")
        return;
      if (column.filterOptionValues === void 0) {
        controlledFilterState[column.key] = (_a = column.filterOptionValue) !== null && _a !== void 0 ? _a : null;
      } else {
        controlledFilterState[column.key] = column.filterOptionValues;
      }
    });
    const activeFilters = Object.assign(createShallowClonedObject(uncontrolledFilterStateRef.value), controlledFilterState);
    return activeFilters;
  });
  const filteredDataRef = computed(() => {
    const mergedFilterState = mergedFilterStateRef.value;
    const { columns } = props;
    function createDefaultFilter(columnKey) {
      return (filterOptionValue, row) => !!~String(row[columnKey]).indexOf(String(filterOptionValue));
    }
    const { value: { treeNodes: data } } = treeMateRef;
    const columnEntries = [];
    columns.forEach((column) => {
      if (column.type === "selection" || column.type === "expand" || "children" in column) {
        return;
      }
      columnEntries.push([column.key, column]);
    });
    return data ? data.filter((tmNode) => {
      const { rawNode: row } = tmNode;
      for (const [columnKey, column] of columnEntries) {
        let activeFilterOptionValues = mergedFilterState[columnKey];
        if (activeFilterOptionValues == null)
          continue;
        if (!Array.isArray(activeFilterOptionValues)) {
          activeFilterOptionValues = [activeFilterOptionValues];
        }
        if (!activeFilterOptionValues.length)
          continue;
        const filter2 = column.filter === "default" ? createDefaultFilter(columnKey) : column.filter;
        if (column && typeof filter2 === "function") {
          if (column.filterMode === "and") {
            if (activeFilterOptionValues.some((filterOptionValue) => !filter2(filterOptionValue, row))) {
              return false;
            }
          } else {
            if (activeFilterOptionValues.some((filterOptionValue) => filter2(filterOptionValue, row))) {
              continue;
            } else {
              return false;
            }
          }
        }
      }
      return true;
    }) : [];
  });
  const { sortedDataRef, deriveNextSorter, mergedSortStateRef, sort, clearSorter } = useSorter(props, {
    dataRelatedColsRef,
    filteredDataRef
  });
  dataRelatedColsRef.value.forEach((column) => {
    var _a;
    if (column.filter) {
      const defaultFilterOptionValues = column.defaultFilterOptionValues;
      if (column.filterMultiple) {
        uncontrolledFilterStateRef.value[column.key] = defaultFilterOptionValues || [];
      } else if (defaultFilterOptionValues !== void 0) {
        uncontrolledFilterStateRef.value[column.key] = defaultFilterOptionValues === null ? [] : defaultFilterOptionValues;
      } else {
        uncontrolledFilterStateRef.value[column.key] = (_a = column.defaultFilterOptionValue) !== null && _a !== void 0 ? _a : null;
      }
    }
  });
  const controlledCurrentPageRef = computed(() => {
    const { pagination } = props;
    if (pagination === false)
      return void 0;
    return pagination.page;
  });
  const controlledPageSizeRef = computed(() => {
    const { pagination } = props;
    if (pagination === false)
      return void 0;
    return pagination.pageSize;
  });
  const _mergedCurrentPageRef = useMergedState(controlledCurrentPageRef, uncontrolledCurrentPageRef);
  const mergedPageSizeRef = useMergedState(controlledPageSizeRef, uncontrolledPageSizeRef);
  const boundedMergedCurrentPageRef = useMemo(() => {
    const page2 = _mergedCurrentPageRef.value;
    return props.remote ? page2 : Math.max(1, Math.min(Math.ceil(filteredDataRef.value.length / mergedPageSizeRef.value), page2));
  });
  const mergedPageCountRef = computed(() => {
    const { pagination } = props;
    if (pagination) {
      const { pageCount } = pagination;
      if (pageCount !== void 0)
        return pageCount;
    }
    return void 0;
  });
  const paginatedDataRef = computed(() => {
    if (props.remote)
      return treeMateRef.value.treeNodes;
    if (!props.pagination)
      return sortedDataRef.value;
    const pageSize = mergedPageSizeRef.value;
    const startIndex = (boundedMergedCurrentPageRef.value - 1) * pageSize;
    return sortedDataRef.value.slice(startIndex, startIndex + pageSize);
  });
  const rawPaginatedDataRef = computed(() => {
    return paginatedDataRef.value.map((tmNode) => tmNode.rawNode);
  });
  function mergedOnUpdatePage(page2) {
    const { pagination } = props;
    if (pagination) {
      const { onChange, "onUpdate:page": _onUpdatePage, onUpdatePage } = pagination;
      if (onChange)
        call(onChange, page2);
      if (onUpdatePage)
        call(onUpdatePage, page2);
      if (_onUpdatePage)
        call(_onUpdatePage, page2);
      doUpdatePage(page2);
    }
  }
  function mergedOnUpdatePageSize(pageSize) {
    const { pagination } = props;
    if (pagination) {
      const { onPageSizeChange, "onUpdate:pageSize": _onUpdatePageSize, onUpdatePageSize } = pagination;
      if (onPageSizeChange)
        call(onPageSizeChange, pageSize);
      if (onUpdatePageSize)
        call(onUpdatePageSize, pageSize);
      if (_onUpdatePageSize)
        call(_onUpdatePageSize, pageSize);
      doUpdatePageSize(pageSize);
    }
  }
  const mergedItemCountRef = computed(() => {
    if (props.remote) {
      const { pagination } = props;
      if (pagination) {
        const { itemCount } = pagination;
        if (itemCount !== void 0)
          return itemCount;
      }
      return void 0;
    }
    return filteredDataRef.value.length;
  });
  const mergedPaginationRef = computed(() => {
    return Object.assign(Object.assign({}, props.pagination), {
      // reset deprecated methods
      onChange: void 0,
      onUpdatePage: void 0,
      onUpdatePageSize: void 0,
      onPageSizeChange: void 0,
      "onUpdate:page": mergedOnUpdatePage,
      "onUpdate:pageSize": mergedOnUpdatePageSize,
      // writing merged props after pagination to avoid
      // pagination[key] === undefined
      // key still exists but value is undefined
      page: boundedMergedCurrentPageRef.value,
      pageSize: mergedPageSizeRef.value,
      pageCount: mergedItemCountRef.value === void 0 ? mergedPageCountRef.value : void 0,
      itemCount: mergedItemCountRef.value
    });
  });
  function doUpdatePage(page2) {
    const { "onUpdate:page": _onUpdatePage, onPageChange, onUpdatePage } = props;
    if (onUpdatePage)
      call(onUpdatePage, page2);
    if (_onUpdatePage)
      call(_onUpdatePage, page2);
    if (onPageChange)
      call(onPageChange, page2);
    uncontrolledCurrentPageRef.value = page2;
  }
  function doUpdatePageSize(pageSize) {
    const { "onUpdate:pageSize": _onUpdatePageSize, onPageSizeChange, onUpdatePageSize } = props;
    if (onPageSizeChange)
      call(onPageSizeChange, pageSize);
    if (onUpdatePageSize)
      call(onUpdatePageSize, pageSize);
    if (_onUpdatePageSize)
      call(_onUpdatePageSize, pageSize);
    uncontrolledPageSizeRef.value = pageSize;
  }
  function doUpdateFilters(filters2, sourceColumn) {
    const { onUpdateFilters, "onUpdate:filters": _onUpdateFilters, onFiltersChange } = props;
    if (onUpdateFilters)
      call(onUpdateFilters, filters2, sourceColumn);
    if (_onUpdateFilters)
      call(_onUpdateFilters, filters2, sourceColumn);
    if (onFiltersChange)
      call(onFiltersChange, filters2, sourceColumn);
    uncontrolledFilterStateRef.value = filters2;
  }
  function onUnstableColumnResize(resizedWidth, limitedWidth, column, getColumnWidth) {
    var _a;
    (_a = props.onUnstableColumnResize) === null || _a === void 0 ? void 0 : _a.call(props, resizedWidth, limitedWidth, column, getColumnWidth);
  }
  function page(page2) {
    doUpdatePage(page2);
  }
  function clearFilter() {
    clearFilters();
  }
  function clearFilters() {
    filters({});
  }
  function filters(filters2) {
    filter(filters2);
  }
  function filter(filters2) {
    if (!filters2) {
      uncontrolledFilterStateRef.value = {};
    } else if (filters2) {
      uncontrolledFilterStateRef.value = createShallowClonedObject(filters2);
    } else
      ;
  }
  return {
    treeMateRef,
    mergedCurrentPageRef: boundedMergedCurrentPageRef,
    mergedPaginationRef,
    paginatedDataRef,
    rawPaginatedDataRef,
    mergedFilterStateRef,
    mergedSortStateRef,
    hoverKeyRef: ref(null),
    selectionColumnRef,
    childTriggerColIndexRef,
    doUpdateFilters,
    deriveNextSorter,
    doUpdatePageSize,
    doUpdatePage,
    onUnstableColumnResize,
    // exported methods
    filter,
    filters,
    clearFilter,
    clearFilters,
    clearSorter,
    page,
    sort
  };
}
function useScroll(props, { mainTableInstRef, mergedCurrentPageRef, bodyWidthRef, scrollPartRef }) {
  let scrollLeft = 0;
  const leftActiveFixedColKeyRef = ref(null);
  const leftActiveFixedChildrenColKeysRef = ref([]);
  const rightActiveFixedColKeyRef = ref(null);
  const rightActiveFixedChildrenColKeysRef = ref([]);
  const styleScrollXRef = computed(() => {
    return formatLength(props.scrollX);
  });
  const leftFixedColumnsRef = computed(() => {
    return props.columns.filter((column) => column.fixed === "left");
  });
  const rightFixedColumnsRef = computed(() => {
    return props.columns.filter((column) => column.fixed === "right");
  });
  const fixedColumnLeftMapRef = computed(() => {
    const columns = {};
    let left = 0;
    function traverse(cols) {
      cols.forEach((col) => {
        const positionInfo = { start: left, end: 0 };
        columns[getColKey(col)] = positionInfo;
        if ("children" in col) {
          traverse(col.children);
          positionInfo.end = left;
        } else {
          left += getNumberColWidth(col) || 0;
          positionInfo.end = left;
        }
      });
    }
    traverse(leftFixedColumnsRef.value);
    return columns;
  });
  const fixedColumnRightMapRef = computed(() => {
    const columns = {};
    let right = 0;
    function traverse(cols) {
      for (let i = cols.length - 1; i >= 0; --i) {
        const col = cols[i];
        const positionInfo = { start: right, end: 0 };
        columns[getColKey(col)] = positionInfo;
        if ("children" in col) {
          traverse(col.children);
          positionInfo.end = right;
        } else {
          right += getNumberColWidth(col) || 0;
          positionInfo.end = right;
        }
      }
    }
    traverse(rightFixedColumnsRef.value);
    return columns;
  });
  function deriveActiveLeftFixedColumn() {
    var _a, _b;
    const { value: leftFixedColumns } = leftFixedColumnsRef;
    let leftWidth = 0;
    const { value: fixedColumnLeftMap } = fixedColumnLeftMapRef;
    let leftActiveFixedColKey = null;
    for (let i = 0; i < leftFixedColumns.length; ++i) {
      const key = getColKey(leftFixedColumns[i]);
      if (scrollLeft > (((_a = fixedColumnLeftMap[key]) === null || _a === void 0 ? void 0 : _a.start) || 0) - leftWidth) {
        leftActiveFixedColKey = key;
        leftWidth = ((_b = fixedColumnLeftMap[key]) === null || _b === void 0 ? void 0 : _b.end) || 0;
      } else {
        break;
      }
    }
    leftActiveFixedColKeyRef.value = leftActiveFixedColKey;
  }
  function deriveActiveLeftFixedChildrenColumns() {
    leftActiveFixedChildrenColKeysRef.value = [];
    let activeLeftFixedColumn = props.columns.find((col) => getColKey(col) === leftActiveFixedColKeyRef.value);
    while (activeLeftFixedColumn && "children" in activeLeftFixedColumn) {
      const length = activeLeftFixedColumn.children.length;
      if (length === 0)
        break;
      const nextActiveLeftFixedColumn = activeLeftFixedColumn.children[length - 1];
      leftActiveFixedChildrenColKeysRef.value.push(getColKey(nextActiveLeftFixedColumn));
      activeLeftFixedColumn = nextActiveLeftFixedColumn;
    }
  }
  function deriveActiveRightFixedColumn() {
    var _a, _b;
    const { value: rightFixedColumns } = rightFixedColumnsRef;
    const scrollWidth = Number(props.scrollX);
    const { value: tableWidth } = bodyWidthRef;
    if (tableWidth === null)
      return;
    let rightWidth = 0;
    let rightActiveFixedColKey = null;
    const { value: fixedColumnRightMap } = fixedColumnRightMapRef;
    for (let i = rightFixedColumns.length - 1; i >= 0; --i) {
      const key = getColKey(rightFixedColumns[i]);
      if (Math.round(scrollLeft + (((_a = fixedColumnRightMap[key]) === null || _a === void 0 ? void 0 : _a.start) || 0) + tableWidth - rightWidth) < scrollWidth) {
        rightActiveFixedColKey = key;
        rightWidth = ((_b = fixedColumnRightMap[key]) === null || _b === void 0 ? void 0 : _b.end) || 0;
      } else {
        break;
      }
    }
    rightActiveFixedColKeyRef.value = rightActiveFixedColKey;
  }
  function deriveActiveRightFixedChildrenColumns() {
    rightActiveFixedChildrenColKeysRef.value = [];
    let activeRightFixedColumn = props.columns.find((col) => getColKey(col) === rightActiveFixedColKeyRef.value);
    while (activeRightFixedColumn && "children" in activeRightFixedColumn && activeRightFixedColumn.children.length) {
      const nextActiveRightFixedColumn = activeRightFixedColumn.children[0];
      rightActiveFixedChildrenColKeysRef.value.push(getColKey(nextActiveRightFixedColumn));
      activeRightFixedColumn = nextActiveRightFixedColumn;
    }
  }
  function getScrollElements() {
    const header = mainTableInstRef.value ? mainTableInstRef.value.getHeaderElement() : null;
    const body = mainTableInstRef.value ? mainTableInstRef.value.getBodyElement() : null;
    return {
      header,
      body
    };
  }
  function scrollMainTableBodyToTop() {
    const { body } = getScrollElements();
    if (body) {
      body.scrollTop = 0;
    }
  }
  function handleTableHeaderScroll() {
    if (scrollPartRef.value === "head") {
      beforeNextFrameOnce(syncScrollState);
    }
  }
  function handleTableBodyScroll(e) {
    var _a;
    (_a = props.onScroll) === null || _a === void 0 ? void 0 : _a.call(props, e);
    if (scrollPartRef.value === "body") {
      beforeNextFrameOnce(syncScrollState);
    }
  }
  function syncScrollState() {
    const { header, body } = getScrollElements();
    if (!body)
      return;
    const { value: tableWidth } = bodyWidthRef;
    if (tableWidth === null)
      return;
    const { value: scrollPart } = scrollPartRef;
    if (props.maxHeight || props.flexHeight) {
      if (!header)
        return;
      if (scrollPart === "head") {
        scrollLeft = header.scrollLeft;
        body.scrollLeft = scrollLeft;
      } else {
        scrollLeft = body.scrollLeft;
        header.scrollLeft = scrollLeft;
      }
    } else {
      scrollLeft = body.scrollLeft;
    }
    deriveActiveLeftFixedColumn();
    deriveActiveLeftFixedChildrenColumns();
    deriveActiveRightFixedColumn();
    deriveActiveRightFixedChildrenColumns();
  }
  function setHeaderScrollLeft(left) {
    const { header } = getScrollElements();
    if (!header)
      return;
    header.scrollLeft = left;
    syncScrollState();
  }
  watch(mergedCurrentPageRef, () => {
    scrollMainTableBodyToTop();
  });
  return {
    styleScrollXRef,
    fixedColumnLeftMapRef,
    fixedColumnRightMapRef,
    leftFixedColumnsRef,
    rightFixedColumnsRef,
    leftActiveFixedColKeyRef,
    leftActiveFixedChildrenColKeysRef,
    rightActiveFixedColKeyRef,
    rightActiveFixedChildrenColKeysRef,
    syncScrollState,
    handleTableBodyScroll,
    handleTableHeaderScroll,
    setHeaderScrollLeft
  };
}
function useResizable() {
  const resizableWidthsRef = ref({});
  function getResizableWidth(key) {
    return resizableWidthsRef.value[key];
  }
  function doUpdateResizableWidth(column, width) {
    if (isColumnResizable(column) && "key" in column) {
      resizableWidthsRef.value[column.key] = width;
    }
  }
  function clearResizableWidth() {
    resizableWidthsRef.value = {};
  }
  return {
    getResizableWidth,
    doUpdateResizableWidth,
    clearResizableWidth
  };
}
function getRowsAndCols(columns, getResizableWidth) {
  const rows = [];
  const cols = [];
  const dataRelatedCols = [];
  const rowItemMap = /* @__PURE__ */ new WeakMap();
  let maxDepth = -1;
  let totalRowSpan = 0;
  let hasEllipsis = false;
  function ensureMaxDepth(columns2, currentDepth) {
    if (currentDepth > maxDepth) {
      rows[currentDepth] = [];
      maxDepth = currentDepth;
    }
    for (const column of columns2) {
      if ("children" in column) {
        ensureMaxDepth(column.children, currentDepth + 1);
      } else {
        const key = "key" in column ? column.key : void 0;
        cols.push({
          key: getColKey(column),
          style: createCustomWidthStyle(column, key !== void 0 ? formatLength(getResizableWidth(key)) : void 0),
          column
        });
        totalRowSpan += 1;
        if (!hasEllipsis) {
          hasEllipsis = !!column.ellipsis;
        }
        dataRelatedCols.push(column);
      }
    }
  }
  ensureMaxDepth(columns, 0);
  let currentLeafIndex = 0;
  function ensureColLayout(columns2, currentDepth) {
    let hideUntilIndex = 0;
    columns2.forEach((column, index) => {
      var _a;
      if ("children" in column) {
        const cachedCurrentLeafIndex = currentLeafIndex;
        const rowItem = {
          column,
          colSpan: 0,
          rowSpan: 1,
          isLast: false
        };
        ensureColLayout(column.children, currentDepth + 1);
        column.children.forEach((childColumn) => {
          var _a2, _b;
          rowItem.colSpan += (_b = (_a2 = rowItemMap.get(childColumn)) === null || _a2 === void 0 ? void 0 : _a2.colSpan) !== null && _b !== void 0 ? _b : 0;
        });
        if (cachedCurrentLeafIndex + rowItem.colSpan === totalRowSpan) {
          rowItem.isLast = true;
        }
        rowItemMap.set(column, rowItem);
        rows[currentDepth].push(rowItem);
      } else {
        if (currentLeafIndex < hideUntilIndex) {
          currentLeafIndex += 1;
          return;
        }
        let colSpan = 1;
        if ("titleColSpan" in column) {
          colSpan = (_a = column.titleColSpan) !== null && _a !== void 0 ? _a : 1;
        }
        if (colSpan > 1) {
          hideUntilIndex = currentLeafIndex + colSpan;
        }
        const isLast = currentLeafIndex + colSpan === totalRowSpan;
        const rowItem = {
          column,
          colSpan,
          rowSpan: maxDepth - currentDepth + 1,
          isLast
        };
        rowItemMap.set(column, rowItem);
        rows[currentDepth].push(rowItem);
        currentLeafIndex += 1;
      }
    });
  }
  ensureColLayout(columns, 0);
  return {
    hasEllipsis,
    rows,
    cols,
    dataRelatedCols
  };
}
function useGroupHeader(props, getResizableWidth) {
  const rowsAndCols = computed(() => getRowsAndCols(props.columns, getResizableWidth));
  return {
    rowsRef: computed(() => rowsAndCols.value.rows),
    colsRef: computed(() => rowsAndCols.value.cols),
    hasEllipsisRef: computed(() => rowsAndCols.value.hasEllipsis),
    dataRelatedColsRef: computed(() => rowsAndCols.value.dataRelatedCols)
  };
}
function useExpand(props, treeMateRef) {
  const renderExpandRef = useMemo(() => {
    for (const col of props.columns) {
      if (col.type === "expand") {
        return col.renderExpand;
      }
    }
  });
  const expandableRef = useMemo(() => {
    let expandable;
    for (const col of props.columns) {
      if (col.type === "expand") {
        expandable = col.expandable;
        break;
      }
    }
    return expandable;
  });
  const uncontrolledExpandedRowKeysRef = ref(props.defaultExpandAll ? (renderExpandRef === null || renderExpandRef === void 0 ? void 0 : renderExpandRef.value) ? (() => {
    const expandedKeys = [];
    treeMateRef.value.treeNodes.forEach((tmNode) => {
      var _a;
      if ((_a = expandableRef.value) === null || _a === void 0 ? void 0 : _a.call(expandableRef, tmNode.rawNode)) {
        expandedKeys.push(tmNode.key);
      }
    });
    return expandedKeys;
  })() : treeMateRef.value.getNonLeafKeys() : props.defaultExpandedRowKeys);
  const controlledExpandedRowKeysRef = toRef(props, "expandedRowKeys");
  const stickyExpandedRowsRef = toRef(props, "stickyExpandedRows");
  const mergedExpandedRowKeysRef = useMergedState(controlledExpandedRowKeysRef, uncontrolledExpandedRowKeysRef);
  function doUpdateExpandedRowKeys(expandedKeys) {
    const { onUpdateExpandedRowKeys, "onUpdate:expandedRowKeys": _onUpdateExpandedRowKeys } = props;
    if (onUpdateExpandedRowKeys) {
      call(onUpdateExpandedRowKeys, expandedKeys);
    }
    if (_onUpdateExpandedRowKeys) {
      call(_onUpdateExpandedRowKeys, expandedKeys);
    }
    uncontrolledExpandedRowKeysRef.value = expandedKeys;
  }
  return {
    stickyExpandedRowsRef,
    mergedExpandedRowKeysRef,
    renderExpandRef,
    expandableRef,
    doUpdateExpandedRowKeys
  };
}
const fixedColumnStyle = createFixedColumnStyle();
const style$1 = c([cB("data-table", `
 width: 100%;
 font-size: var(--n-font-size);
 display: flex;
 flex-direction: column;
 position: relative;
 --n-merged-th-color: var(--n-th-color);
 --n-merged-td-color: var(--n-td-color);
 --n-merged-border-color: var(--n-border-color);
 --n-merged-th-color-hover: var(--n-th-color-hover);
 --n-merged-td-color-hover: var(--n-td-color-hover);
 --n-merged-td-color-striped: var(--n-td-color-striped);
 `, [cB("data-table-wrapper", `
 flex-grow: 1;
 display: flex;
 flex-direction: column;
 `), cM("flex-height", [c(">", [cB("data-table-wrapper", [c(">", [cB("data-table-base-table", `
 display: flex;
 flex-direction: column;
 flex-grow: 1;
 `, [c(">", [cB("data-table-base-table-body", "flex-basis: 0;", [
  // last-child means there is no empty icon
  // body is a scrollbar, we need to override height 100%
  c("&:last-child", "flex-grow: 1;")
])])])])])])]), c(">", [cB("data-table-loading-wrapper", `
 color: var(--n-loading-color);
 font-size: var(--n-loading-size);
 position: absolute;
 left: 50%;
 top: 50%;
 transform: translateX(-50%) translateY(-50%);
 transition: color .3s var(--n-bezier);
 display: flex;
 align-items: center;
 justify-content: center;
 `, [fadeInScaleUpTransition({
  originalTransform: "translateX(-50%) translateY(-50%)"
})])]), cB("data-table-expand-placeholder", `
 margin-right: 8px;
 display: inline-block;
 width: 16px;
 height: 1px;
 `), cB("data-table-indent", `
 display: inline-block;
 height: 1px;
 `), cB("data-table-expand-trigger", `
 display: inline-flex;
 margin-right: 8px;
 cursor: pointer;
 font-size: 16px;
 vertical-align: -0.2em;
 position: relative;
 width: 16px;
 height: 16px;
 color: var(--n-td-text-color);
 transition: color .3s var(--n-bezier);
 `, [cM("expanded", [cB("icon", "transform: rotate(90deg);", [iconSwitchTransition({
  originalTransform: "rotate(90deg)"
})]), cB("base-icon", "transform: rotate(90deg);", [iconSwitchTransition({
  originalTransform: "rotate(90deg)"
})])]), cB("base-loading", `
 color: var(--n-loading-color);
 transition: color .3s var(--n-bezier);
 position: absolute;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 `, [iconSwitchTransition()]), cB("icon", `
 position: absolute;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 `, [iconSwitchTransition()]), cB("base-icon", `
 position: absolute;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 `, [iconSwitchTransition()])]), cB("data-table-thead", `
 transition: background-color .3s var(--n-bezier);
 background-color: var(--n-merged-th-color);
 `), cB("data-table-tr", `
 box-sizing: border-box;
 background-clip: padding-box;
 transition: background-color .3s var(--n-bezier);
 `, [cB("data-table-expand", `
 position: sticky;
 left: 0;
 overflow: hidden;
 margin: calc(var(--n-th-padding) * -1);
 padding: var(--n-th-padding);
 box-sizing: border-box;
 `), cM("striped", "background-color: var(--n-merged-td-color-striped);", [cB("data-table-td", "background-color: var(--n-merged-td-color-striped);")]), cNotM("summary", [c("&:hover", "background-color: var(--n-merged-td-color-hover);", [c(">", [cB("data-table-td", "background-color: var(--n-merged-td-color-hover);")])])])]), cB("data-table-th", `
 padding: var(--n-th-padding);
 position: relative;
 text-align: start;
 box-sizing: border-box;
 background-color: var(--n-merged-th-color);
 border-color: var(--n-merged-border-color);
 border-bottom: 1px solid var(--n-merged-border-color);
 color: var(--n-th-text-color);
 transition:
 border-color .3s var(--n-bezier),
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 font-weight: var(--n-th-font-weight);
 `, [cM("filterable", `
 padding-right: 36px;
 `, [cM("sortable", `
 padding-right: calc(var(--n-th-padding) + 36px);
 `)]), fixedColumnStyle, cM("selection", `
 padding: 0;
 text-align: center;
 line-height: 0;
 z-index: 3;
 `), cE("title-wrapper", `
 display: flex;
 align-items: center;
 flex-wrap: nowrap;
 max-width: 100%;
 `, [cE("title", `
 flex: 1;
 min-width: 0;
 `)]), cE("ellipsis", `
 display: inline-block;
 vertical-align: bottom;
 text-overflow: ellipsis;
 overflow: hidden;
 white-space: nowrap;
 max-width: 100%;
 `), cM("hover", `
 background-color: var(--n-merged-th-color-hover);
 `), cM("sortable", `
 cursor: pointer;
 `, [cE("ellipsis", `
 max-width: calc(100% - 18px);
 `), c("&:hover", `
 background-color: var(--n-merged-th-color-hover);
 `)]), cB("data-table-sorter", `
 height: var(--n-sorter-size);
 width: var(--n-sorter-size);
 margin-left: 4px;
 position: relative;
 display: inline-flex;
 align-items: center;
 justify-content: center;
 vertical-align: -0.2em;
 color: var(--n-th-icon-color);
 transition: color .3s var(--n-bezier);
 `, [cB("base-icon", "transition: transform .3s var(--n-bezier)"), cM("desc", [cB("base-icon", `
 transform: rotate(0deg);
 `)]), cM("asc", [cB("base-icon", `
 transform: rotate(-180deg);
 `)]), cM("asc, desc", `
 color: var(--n-th-icon-color-active);
 `)]), cB("data-table-resize-button", `
 width: var(--n-resizable-container-size);
 position: absolute;
 top: 0;
 right: calc(var(--n-resizable-container-size) / 2);
 bottom: 0;
 cursor: col-resize;
 user-select: none;
 `, [c("&::after", `
 width: var(--n-resizable-size);
 height: 50%;
 position: absolute;
 top: 50%;
 left: calc(var(--n-resizable-container-size) / 2);
 bottom: 0;
 background-color: var(--n-merged-border-color);
 transform: translateY(-50%);
 transition: background-color .3s var(--n-bezier);
 z-index: 1;
 content: '';
 `), cM("active", [c("&::after", ` 
 background-color: var(--n-th-icon-color-active);
 `)]), c("&:hover::after", `
 background-color: var(--n-th-icon-color-active);
 `)]), cB("data-table-filter", `
 position: absolute;
 z-index: auto;
 right: 0;
 width: 36px;
 top: 0;
 bottom: 0;
 cursor: pointer;
 display: flex;
 justify-content: center;
 align-items: center;
 transition:
 background-color .3s var(--n-bezier),
 color .3s var(--n-bezier);
 font-size: var(--n-filter-size);
 color: var(--n-th-icon-color);
 `, [c("&:hover", `
 background-color: var(--n-th-button-color-hover);
 `), cM("show", `
 background-color: var(--n-th-button-color-hover);
 `), cM("active", `
 background-color: var(--n-th-button-color-hover);
 color: var(--n-th-icon-color-active);
 `)])]), cB("data-table-td", `
 padding: var(--n-td-padding);
 text-align: start;
 box-sizing: border-box;
 border: none;
 background-color: var(--n-merged-td-color);
 color: var(--n-td-text-color);
 border-bottom: 1px solid var(--n-merged-border-color);
 transition:
 box-shadow .3s var(--n-bezier),
 background-color .3s var(--n-bezier),
 border-color .3s var(--n-bezier),
 color .3s var(--n-bezier);
 `, [cM("expand", [cB("data-table-expand-trigger", `
 margin-right: 0;
 `)]), cM("last-row", `
 border-bottom: 0 solid var(--n-merged-border-color);
 `, [
  // make sure there is no overlap between bottom border and
  // fixed column box shadow
  c("&::after", `
 bottom: 0 !important;
 `),
  c("&::before", `
 bottom: 0 !important;
 `)
]), cM("summary", `
 background-color: var(--n-merged-th-color);
 `), cM("hover", `
 background-color: var(--n-merged-td-color-hover);
 `), cE("ellipsis", `
 display: inline-block;
 text-overflow: ellipsis;
 overflow: hidden;
 white-space: nowrap;
 max-width: 100%;
 vertical-align: bottom;
 `), cM("selection, expand", `
 text-align: center;
 padding: 0;
 line-height: 0;
 `), fixedColumnStyle]), cB("data-table-empty", `
 box-sizing: border-box;
 padding: var(--n-empty-padding);
 flex-grow: 1;
 flex-shrink: 0;
 opacity: 1;
 display: flex;
 align-items: center;
 justify-content: center;
 transition: opacity .3s var(--n-bezier);
 `, [cM("hide", `
 opacity: 0;
 `)]), cE("pagination", `
 margin: var(--n-pagination-margin);
 display: flex;
 justify-content: flex-end;
 `), cB("data-table-wrapper", `
 position: relative;
 opacity: 1;
 transition: opacity .3s var(--n-bezier), border-color .3s var(--n-bezier);
 border-top-left-radius: var(--n-border-radius);
 border-top-right-radius: var(--n-border-radius);
 line-height: var(--n-line-height);
 `), cM("loading", [cB("data-table-wrapper", `
 opacity: var(--n-opacity-loading);
 pointer-events: none;
 `)]), cM("single-column", [cB("data-table-td", `
 border-bottom: 0 solid var(--n-merged-border-color);
 `, [c("&::after, &::before", `
 bottom: 0 !important;
 `)])]), cNotM("single-line", [cB("data-table-th", `
 border-right: 1px solid var(--n-merged-border-color);
 `, [cM("last", `
 border-right: 0 solid var(--n-merged-border-color);
 `)]), cB("data-table-td", `
 border-right: 1px solid var(--n-merged-border-color);
 `, [cM("last-col", `
 border-right: 0 solid var(--n-merged-border-color);
 `)])]), cM("bordered", [cB("data-table-wrapper", `
 border: 1px solid var(--n-merged-border-color);
 border-bottom-left-radius: var(--n-border-radius);
 border-bottom-right-radius: var(--n-border-radius);
 overflow: hidden;
 `)]), cB("data-table-base-table", [cM("transition-disabled", [cB("data-table-th", [c("&::after, &::before", "transition: none;")]), cB("data-table-td", [c("&::after, &::before", "transition: none;")])])]), cM("bottom-bordered", [cB("data-table-td", [cM("last-row", `
 border-bottom: 1px solid var(--n-merged-border-color);
 `)])]), cB("data-table-table", `
 font-variant-numeric: tabular-nums;
 width: 100%;
 word-break: break-word;
 transition: background-color .3s var(--n-bezier);
 border-collapse: separate;
 border-spacing: 0;
 background-color: var(--n-merged-td-color);
 `), cB("data-table-base-table-header", `
 border-top-left-radius: calc(var(--n-border-radius) - 1px);
 border-top-right-radius: calc(var(--n-border-radius) - 1px);
 z-index: 3;
 overflow: scroll;
 flex-shrink: 0;
 transition: border-color .3s var(--n-bezier);
 scrollbar-width: none;
 `, [c("&::-webkit-scrollbar", `
 width: 0;
 height: 0;
 `)]), cB("data-table-check-extra", `
 transition: color .3s var(--n-bezier);
 color: var(--n-th-icon-color);
 position: absolute;
 font-size: 14px;
 right: -4px;
 top: 50%;
 transform: translateY(-50%);
 z-index: 1;
 `)]), cB("data-table-filter-menu", [cB("scrollbar", `
 max-height: 240px;
 `), cE("group", `
 display: flex;
 flex-direction: column;
 padding: 12px 12px 0 12px;
 `, [cB("checkbox", `
 margin-bottom: 12px;
 margin-right: 0;
 `), cB("radio", `
 margin-bottom: 12px;
 margin-right: 0;
 `)]), cE("action", `
 padding: var(--n-action-padding);
 display: flex;
 flex-wrap: nowrap;
 justify-content: space-evenly;
 border-top: 1px solid var(--n-action-divider-color);
 `, [cB("button", [c("&:not(:last-child)", `
 margin: var(--n-action-button-margin);
 `), c("&:last-child", `
 margin-right: 0;
 `)])]), cB("divider", `
 margin: 0 !important;
 `)]), insideModal(cB("data-table", `
 --n-merged-th-color: var(--n-th-color-modal);
 --n-merged-td-color: var(--n-td-color-modal);
 --n-merged-border-color: var(--n-border-color-modal);
 --n-merged-th-color-hover: var(--n-th-color-hover-modal);
 --n-merged-td-color-hover: var(--n-td-color-hover-modal);
 --n-merged-td-color-striped: var(--n-td-color-striped-modal);
 `)), insidePopover(cB("data-table", `
 --n-merged-th-color: var(--n-th-color-popover);
 --n-merged-td-color: var(--n-td-color-popover);
 --n-merged-border-color: var(--n-border-color-popover);
 --n-merged-th-color-hover: var(--n-th-color-hover-popover);
 --n-merged-td-color-hover: var(--n-td-color-hover-popover);
 --n-merged-td-color-striped: var(--n-td-color-striped-popover);
 `))]);
function createFixedColumnStyle() {
  return [cM("fixed-left", `
 left: 0;
 position: sticky;
 z-index: 2;
 `, [c("&::after", `
 pointer-events: none;
 content: "";
 width: 36px;
 display: inline-block;
 position: absolute;
 top: 0;
 bottom: -1px;
 transition: box-shadow .2s var(--n-bezier);
 right: -36px;
 `)]), cM("fixed-right", `
 right: 0;
 position: sticky;
 z-index: 1;
 `, [c("&::before", `
 pointer-events: none;
 content: "";
 width: 36px;
 display: inline-block;
 position: absolute;
 top: 0;
 bottom: -1px;
 transition: box-shadow .2s var(--n-bezier);
 left: -36px;
 `)])];
}
const NDataTable = defineComponent({
  name: "DataTable",
  alias: ["AdvancedTable"],
  props: dataTableProps,
  setup(props, { slots }) {
    const { mergedBorderedRef, mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const mergedBottomBorderedRef = computed(() => {
      const { bottomBordered } = props;
      if (mergedBorderedRef.value)
        return false;
      if (bottomBordered !== void 0)
        return bottomBordered;
      return true;
    });
    const themeRef = useTheme("DataTable", "-data-table", style$1, dataTableLight, props, mergedClsPrefixRef);
    const bodyWidthRef = ref(null);
    const scrollPartRef = ref("body");
    onDeactivated(() => {
      scrollPartRef.value = "body";
    });
    const mainTableInstRef = ref(null);
    const { getResizableWidth, clearResizableWidth, doUpdateResizableWidth } = useResizable();
    const { rowsRef, colsRef, dataRelatedColsRef, hasEllipsisRef } = useGroupHeader(props, getResizableWidth);
    const { treeMateRef, mergedCurrentPageRef, paginatedDataRef, rawPaginatedDataRef, selectionColumnRef, hoverKeyRef, mergedPaginationRef, mergedFilterStateRef, mergedSortStateRef, childTriggerColIndexRef, doUpdatePage, doUpdateFilters, onUnstableColumnResize, deriveNextSorter, filter, filters, clearFilter, clearFilters, clearSorter, page, sort } = useTableData(props, { dataRelatedColsRef });
    const { doCheckAll, doUncheckAll, doCheck, doUncheck, headerCheckboxDisabledRef, someRowsCheckedRef, allRowsCheckedRef, mergedCheckedRowKeySetRef, mergedInderminateRowKeySetRef } = useCheck(props, {
      selectionColumnRef,
      treeMateRef,
      paginatedDataRef
    });
    const { stickyExpandedRowsRef, mergedExpandedRowKeysRef, renderExpandRef, expandableRef, doUpdateExpandedRowKeys } = useExpand(props, treeMateRef);
    const { handleTableBodyScroll, handleTableHeaderScroll, syncScrollState, setHeaderScrollLeft, leftActiveFixedColKeyRef, leftActiveFixedChildrenColKeysRef, rightActiveFixedColKeyRef, rightActiveFixedChildrenColKeysRef, leftFixedColumnsRef, rightFixedColumnsRef, fixedColumnLeftMapRef, fixedColumnRightMapRef } = useScroll(props, {
      scrollPartRef,
      bodyWidthRef,
      mainTableInstRef,
      mergedCurrentPageRef
    });
    const { localeRef } = useLocale("DataTable");
    const mergedTableLayoutRef = computed(() => {
      if (props.virtualScroll || props.flexHeight || props.maxHeight !== void 0 || hasEllipsisRef.value) {
        return "fixed";
      }
      return props.tableLayout;
    });
    provide(dataTableInjectionKey, {
      props,
      treeMateRef,
      renderExpandIconRef: toRef(props, "renderExpandIcon"),
      loadingKeySetRef: ref(/* @__PURE__ */ new Set()),
      slots,
      indentRef: toRef(props, "indent"),
      childTriggerColIndexRef,
      bodyWidthRef,
      componentId: createId(),
      hoverKeyRef,
      mergedClsPrefixRef,
      mergedThemeRef: themeRef,
      scrollXRef: computed(() => props.scrollX),
      rowsRef,
      colsRef,
      paginatedDataRef,
      leftActiveFixedColKeyRef,
      leftActiveFixedChildrenColKeysRef,
      rightActiveFixedColKeyRef,
      rightActiveFixedChildrenColKeysRef,
      leftFixedColumnsRef,
      rightFixedColumnsRef,
      fixedColumnLeftMapRef,
      fixedColumnRightMapRef,
      mergedCurrentPageRef,
      someRowsCheckedRef,
      allRowsCheckedRef,
      mergedSortStateRef,
      mergedFilterStateRef,
      loadingRef: toRef(props, "loading"),
      rowClassNameRef: toRef(props, "rowClassName"),
      mergedCheckedRowKeySetRef,
      mergedExpandedRowKeysRef,
      mergedInderminateRowKeySetRef,
      localeRef,
      scrollPartRef,
      expandableRef,
      stickyExpandedRowsRef,
      rowKeyRef: toRef(props, "rowKey"),
      renderExpandRef,
      summaryRef: toRef(props, "summary"),
      virtualScrollRef: toRef(props, "virtualScroll"),
      rowPropsRef: toRef(props, "rowProps"),
      stripedRef: toRef(props, "striped"),
      checkOptionsRef: computed(() => {
        const { value: selectionColumn } = selectionColumnRef;
        return selectionColumn === null || selectionColumn === void 0 ? void 0 : selectionColumn.options;
      }),
      rawPaginatedDataRef,
      filterMenuCssVarsRef: computed(() => {
        const { self: { actionDividerColor, actionPadding, actionButtonMargin } } = themeRef.value;
        return {
          "--n-action-padding": actionPadding,
          "--n-action-button-margin": actionButtonMargin,
          "--n-action-divider-color": actionDividerColor
        };
      }),
      onLoadRef: toRef(props, "onLoad"),
      mergedTableLayoutRef,
      maxHeightRef: toRef(props, "maxHeight"),
      minHeightRef: toRef(props, "minHeight"),
      flexHeightRef: toRef(props, "flexHeight"),
      headerCheckboxDisabledRef,
      paginationBehaviorOnFilterRef: toRef(props, "paginationBehaviorOnFilter"),
      summaryPlacementRef: toRef(props, "summaryPlacement"),
      scrollbarPropsRef: toRef(props, "scrollbarProps"),
      syncScrollState,
      doUpdatePage,
      doUpdateFilters,
      getResizableWidth,
      onUnstableColumnResize,
      clearResizableWidth,
      doUpdateResizableWidth,
      deriveNextSorter,
      doCheck,
      doUncheck,
      doCheckAll,
      doUncheckAll,
      doUpdateExpandedRowKeys,
      handleTableHeaderScroll,
      handleTableBodyScroll,
      setHeaderScrollLeft,
      renderCell: toRef(props, "renderCell")
    });
    const exposedMethods = {
      filter,
      filters,
      clearFilters,
      clearSorter,
      page,
      sort,
      clearFilter,
      scrollTo: (arg0, arg1) => {
        var _a;
        (_a = mainTableInstRef.value) === null || _a === void 0 ? void 0 : _a.scrollTo(arg0, arg1);
      }
    };
    const cssVarsRef = computed(() => {
      const { size } = props;
      const { common: { cubicBezierEaseInOut }, self: { borderColor, tdColorHover, thColor, thColorHover, tdColor, tdTextColor, thTextColor, thFontWeight, thButtonColorHover, thIconColor, thIconColorActive, filterSize, borderRadius, lineHeight, tdColorModal, thColorModal, borderColorModal, thColorHoverModal, tdColorHoverModal, borderColorPopover, thColorPopover, tdColorPopover, tdColorHoverPopover, thColorHoverPopover, paginationMargin, emptyPadding, boxShadowAfter, boxShadowBefore, sorterSize, resizableContainerSize, resizableSize, loadingColor, loadingSize, opacityLoading, tdColorStriped, tdColorStripedModal, tdColorStripedPopover, [createKey("fontSize", size)]: fontSize, [createKey("thPadding", size)]: thPadding, [createKey("tdPadding", size)]: tdPadding } } = themeRef.value;
      return {
        "--n-font-size": fontSize,
        "--n-th-padding": thPadding,
        "--n-td-padding": tdPadding,
        "--n-bezier": cubicBezierEaseInOut,
        "--n-border-radius": borderRadius,
        "--n-line-height": lineHeight,
        "--n-border-color": borderColor,
        "--n-border-color-modal": borderColorModal,
        "--n-border-color-popover": borderColorPopover,
        "--n-th-color": thColor,
        "--n-th-color-hover": thColorHover,
        "--n-th-color-modal": thColorModal,
        "--n-th-color-hover-modal": thColorHoverModal,
        "--n-th-color-popover": thColorPopover,
        "--n-th-color-hover-popover": thColorHoverPopover,
        "--n-td-color": tdColor,
        "--n-td-color-hover": tdColorHover,
        "--n-td-color-modal": tdColorModal,
        "--n-td-color-hover-modal": tdColorHoverModal,
        "--n-td-color-popover": tdColorPopover,
        "--n-td-color-hover-popover": tdColorHoverPopover,
        "--n-th-text-color": thTextColor,
        "--n-td-text-color": tdTextColor,
        "--n-th-font-weight": thFontWeight,
        "--n-th-button-color-hover": thButtonColorHover,
        "--n-th-icon-color": thIconColor,
        "--n-th-icon-color-active": thIconColorActive,
        "--n-filter-size": filterSize,
        "--n-pagination-margin": paginationMargin,
        "--n-empty-padding": emptyPadding,
        "--n-box-shadow-before": boxShadowBefore,
        "--n-box-shadow-after": boxShadowAfter,
        "--n-sorter-size": sorterSize,
        "--n-resizable-container-size": resizableContainerSize,
        "--n-resizable-size": resizableSize,
        "--n-loading-size": loadingSize,
        "--n-loading-color": loadingColor,
        "--n-opacity-loading": opacityLoading,
        "--n-td-color-striped": tdColorStriped,
        "--n-td-color-striped-modal": tdColorStripedModal,
        "--n-td-color-striped-popover": tdColorStripedPopover
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("data-table", computed(() => props.size[0]), cssVarsRef, props) : void 0;
    const mergedShowPaginationRef = computed(() => {
      if (!props.pagination)
        return false;
      if (props.paginateSinglePage)
        return true;
      const mergedPagination = mergedPaginationRef.value;
      const { pageCount } = mergedPagination;
      if (pageCount !== void 0)
        return pageCount > 1;
      return mergedPagination.itemCount && mergedPagination.pageSize && mergedPagination.itemCount > mergedPagination.pageSize;
    });
    return Object.assign({ mainTableInstRef, mergedClsPrefix: mergedClsPrefixRef, mergedTheme: themeRef, paginatedData: paginatedDataRef, mergedBordered: mergedBorderedRef, mergedBottomBordered: mergedBottomBorderedRef, mergedPagination: mergedPaginationRef, mergedShowPagination: mergedShowPaginationRef, cssVars: inlineThemeDisabled ? void 0 : cssVarsRef, themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass, onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender }, exposedMethods);
  },
  render() {
    const { mergedClsPrefix, themeClass, onRender, $slots, spinProps } = this;
    onRender === null || onRender === void 0 ? void 0 : onRender();
    return h(
      "div",
      { class: [
        `${mergedClsPrefix}-data-table`,
        themeClass,
        {
          [`${mergedClsPrefix}-data-table--bordered`]: this.mergedBordered,
          [`${mergedClsPrefix}-data-table--bottom-bordered`]: this.mergedBottomBordered,
          [`${mergedClsPrefix}-data-table--single-line`]: this.singleLine,
          [`${mergedClsPrefix}-data-table--single-column`]: this.singleColumn,
          [`${mergedClsPrefix}-data-table--loading`]: this.loading,
          [`${mergedClsPrefix}-data-table--flex-height`]: this.flexHeight
        }
      ], style: this.cssVars },
      h(
        "div",
        { class: `${mergedClsPrefix}-data-table-wrapper` },
        h(MainTable, { ref: "mainTableInstRef" })
      ),
      this.mergedShowPagination ? h(
        "div",
        { class: `${mergedClsPrefix}-data-table__pagination` },
        h(NPagination, Object.assign({ theme: this.mergedTheme.peers.Pagination, themeOverrides: this.mergedTheme.peerOverrides.Pagination, disabled: this.loading }, this.mergedPagination))
      ) : null,
      h(Transition, { name: "fade-in-scale-up-transition" }, {
        default: () => {
          return this.loading ? h("div", { class: `${mergedClsPrefix}-data-table-loading-wrapper` }, resolveSlot($slots.loading, () => [
            h(NBaseLoading, Object.assign({ clsPrefix: mergedClsPrefix, strokeWidth: 20 }, spinProps))
          ])) : null;
        }
      })
    );
  }
});
const uploadInjectionKey = createInjectionKey("n-upload");
const uploadDraggerKey = "__UPLOAD_DRAGGER__";
const NUploadDragger = defineComponent({
  name: "UploadDragger",
  [uploadDraggerKey]: true,
  setup(_, { slots }) {
    const NUpload2 = inject(uploadInjectionKey, null);
    if (!NUpload2) {
      throwError("upload-dragger", "`n-upload-dragger` must be placed inside `n-upload`.");
    }
    return () => {
      const { mergedClsPrefixRef: { value: mergedClsPrefix }, mergedDisabledRef: { value: mergedDisabled }, maxReachedRef: { value: maxReached } } = NUpload2;
      return h("div", { class: [
        `${mergedClsPrefix}-upload-dragger`,
        (mergedDisabled || maxReached) && `${mergedClsPrefix}-upload-dragger--disabled`
      ] }, slots);
    };
  }
});
var __awaiter$2 = globalThis && globalThis.__awaiter || function(thisArg, _arguments, P, generator) {
  function adopt(value) {
    return value instanceof P ? value : new P(function(resolve) {
      resolve(value);
    });
  }
  return new (P || (P = Promise))(function(resolve, reject) {
    function fulfilled(value) {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    }
    function rejected(value) {
      try {
        step(generator["throw"](value));
      } catch (e) {
        reject(e);
      }
    }
    function step(result) {
      result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
    }
    step((generator = generator.apply(thisArg, _arguments || [])).next());
  });
};
const isImageFileType = (type) => type.includes("image/");
const getExtname = (url = "") => {
  const temp = url.split("/");
  const filename = temp[temp.length - 1];
  const filenameWithoutSuffix = filename.split(/#|\?/)[0];
  return (/\.[^./\\]*$/.exec(filenameWithoutSuffix) || [""])[0];
};
const imageExtensionRegex = /(webp|svg|png|gif|jpg|jpeg|jfif|bmp|dpg|ico)$/i;
const isImageFile = (file) => {
  if (file.type) {
    return isImageFileType(file.type);
  }
  const fileNameExtension = getExtname(file.name || "");
  if (imageExtensionRegex.test(fileNameExtension)) {
    return true;
  }
  const url = file.thumbnailUrl || file.url || "";
  const urlExtension = getExtname(url);
  if (/^data:image\//.test(url) || imageExtensionRegex.test(urlExtension)) {
    return true;
  }
  return false;
};
function createImageDataUrl(file) {
  return __awaiter$2(this, void 0, void 0, function* () {
    return yield new Promise((resolve) => {
      if (!file.type || !isImageFileType(file.type)) {
        resolve("");
        return;
      }
      resolve(window.URL.createObjectURL(file));
    });
  });
}
const environmentSupportFile = isBrowser && window.FileReader && window.File;
function isFileSystemDirectoryEntry(item) {
  return item.isDirectory;
}
function isFileSystemFileEntry(item) {
  return item.isFile;
}
function getFilesFromEntries(entries, directory) {
  return __awaiter$2(this, void 0, void 0, function* () {
    const fileAndEntries = [];
    let _resolve;
    let requestCallbackCount = 0;
    function lock() {
      requestCallbackCount++;
    }
    function unlock() {
      requestCallbackCount--;
      if (!requestCallbackCount) {
        _resolve(fileAndEntries);
      }
    }
    function _getFilesFromEntries(entries2) {
      entries2.forEach((entry) => {
        if (!entry)
          return;
        lock();
        if (directory && isFileSystemDirectoryEntry(entry)) {
          const directoryReader = entry.createReader();
          lock();
          directoryReader.readEntries((entries3) => {
            _getFilesFromEntries(entries3);
            unlock();
          }, () => {
            unlock();
          });
        } else if (isFileSystemFileEntry(entry)) {
          lock();
          entry.file((file) => {
            fileAndEntries.push({ file, entry, source: "dnd" });
            unlock();
          }, () => {
            unlock();
          });
        }
        unlock();
      });
    }
    yield new Promise((resolve) => {
      _resolve = resolve;
      _getFilesFromEntries(entries);
    });
    return fileAndEntries;
  });
}
function createSettledFileInfo(fileInfo) {
  const { id, name, percentage, status, url, file, thumbnailUrl, type, fullPath, batchId } = fileInfo;
  return {
    id,
    name,
    percentage: percentage !== null && percentage !== void 0 ? percentage : null,
    status,
    url: url !== null && url !== void 0 ? url : null,
    file: file !== null && file !== void 0 ? file : null,
    thumbnailUrl: thumbnailUrl !== null && thumbnailUrl !== void 0 ? thumbnailUrl : null,
    type: type !== null && type !== void 0 ? type : null,
    fullPath: fullPath !== null && fullPath !== void 0 ? fullPath : null,
    batchId: batchId !== null && batchId !== void 0 ? batchId : null
  };
}
function matchType(name, mimeType, accept) {
  name = name.toLowerCase();
  mimeType = mimeType.toLocaleLowerCase();
  accept = accept.toLocaleLowerCase();
  const acceptAtoms = accept.split(",").map((acceptAtom) => acceptAtom.trim()).filter(Boolean);
  return acceptAtoms.some((acceptAtom) => {
    if (acceptAtom.startsWith(".")) {
      if (name.endsWith(acceptAtom))
        return true;
    } else if (acceptAtom.includes("/")) {
      const [type, subtype] = mimeType.split("/");
      const [acceptType, acceptSubtype] = acceptAtom.split("/");
      if (acceptType === "*" || type && acceptType && acceptType === type) {
        if (acceptSubtype === "*" || subtype && acceptSubtype && acceptSubtype === subtype) {
          return true;
        }
      }
    } else {
      return true;
    }
    return false;
  });
}
const download = (url, name) => {
  if (!url)
    return;
  const a = document.createElement("a");
  a.href = url;
  if (name !== void 0) {
    a.download = name;
  }
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};
const NUploadTrigger = defineComponent({
  name: "UploadTrigger",
  props: {
    abstract: Boolean
  },
  setup(props, { slots }) {
    const NUpload2 = inject(uploadInjectionKey, null);
    if (!NUpload2) {
      throwError("upload-trigger", "`n-upload-trigger` must be placed inside `n-upload`.");
    }
    const { mergedClsPrefixRef, mergedDisabledRef, maxReachedRef, listTypeRef, dragOverRef, openOpenFileDialog, draggerInsideRef, handleFileAddition, mergedDirectoryDndRef, triggerStyleRef } = NUpload2;
    const isImageCardTypeRef = computed(() => listTypeRef.value === "image-card");
    function handleTriggerClick() {
      if (mergedDisabledRef.value || maxReachedRef.value)
        return;
      openOpenFileDialog();
    }
    function handleTriggerDragOver(e) {
      e.preventDefault();
      dragOverRef.value = true;
    }
    function handleTriggerDragEnter(e) {
      e.preventDefault();
      dragOverRef.value = true;
    }
    function handleTriggerDragLeave(e) {
      e.preventDefault();
      dragOverRef.value = false;
    }
    function handleTriggerDrop(e) {
      var _a;
      e.preventDefault();
      if (!draggerInsideRef.value || mergedDisabledRef.value || maxReachedRef.value) {
        dragOverRef.value = false;
        return;
      }
      const dataTransferItems = (_a = e.dataTransfer) === null || _a === void 0 ? void 0 : _a.items;
      if (dataTransferItems === null || dataTransferItems === void 0 ? void 0 : dataTransferItems.length) {
        void getFilesFromEntries(Array.from(dataTransferItems).map((item) => item.webkitGetAsEntry()), mergedDirectoryDndRef.value).then((files) => {
          handleFileAddition(files);
        }).finally(() => {
          dragOverRef.value = false;
        });
      } else {
        dragOverRef.value = false;
      }
    }
    return () => {
      var _a;
      const { value: mergedClsPrefix } = mergedClsPrefixRef;
      return props.abstract ? (_a = slots.default) === null || _a === void 0 ? void 0 : _a.call(slots, {
        handleClick: handleTriggerClick,
        handleDrop: handleTriggerDrop,
        handleDragOver: handleTriggerDragOver,
        handleDragEnter: handleTriggerDragEnter,
        handleDragLeave: handleTriggerDragLeave
      }) : h("div", { class: [
        `${mergedClsPrefix}-upload-trigger`,
        (mergedDisabledRef.value || maxReachedRef.value) && `${mergedClsPrefix}-upload-trigger--disabled`,
        isImageCardTypeRef.value && `${mergedClsPrefix}-upload-trigger--image-card`
      ], style: triggerStyleRef.value, onClick: handleTriggerClick, onDrop: handleTriggerDrop, onDragover: handleTriggerDragOver, onDragenter: handleTriggerDragEnter, onDragleave: handleTriggerDragLeave }, isImageCardTypeRef.value ? h(NUploadDragger, null, {
        default: () => resolveSlot(slots.default, () => [
          h(NBaseIcon, { clsPrefix: mergedClsPrefix }, { default: () => h(AddIcon, null) })
        ])
      }) : slots);
    };
  }
});
const NUploadProgress = defineComponent({
  name: "UploadProgress",
  props: {
    show: Boolean,
    percentage: {
      type: Number,
      required: true
    },
    status: {
      type: String,
      required: true
    }
  },
  setup() {
    const NUpload2 = inject(uploadInjectionKey);
    return {
      mergedTheme: NUpload2.mergedThemeRef
    };
  },
  render() {
    return h(NFadeInExpandTransition, null, {
      default: () => this.show ? h(NProgress, { type: "line", showIndicator: false, percentage: this.percentage, status: this.status, height: 2, theme: this.mergedTheme.peers.Progress, themeOverrides: this.mergedTheme.peerOverrides.Progress }) : null
    });
  }
});
const imageIcon = h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 28 28" },
  h(
    "g",
    { fill: "none" },
    h("path", { d: "M21.75 3A3.25 3.25 0 0 1 25 6.25v15.5A3.25 3.25 0 0 1 21.75 25H6.25A3.25 3.25 0 0 1 3 21.75V6.25A3.25 3.25 0 0 1 6.25 3h15.5zm.583 20.4l-7.807-7.68a.75.75 0 0 0-.968-.07l-.084.07l-7.808 7.68c.183.065.38.1.584.1h15.5c.204 0 .4-.035.583-.1l-7.807-7.68l7.807 7.68zM21.75 4.5H6.25A1.75 1.75 0 0 0 4.5 6.25v15.5c0 .208.036.408.103.593l7.82-7.692a2.25 2.25 0 0 1 3.026-.117l.129.117l7.82 7.692c.066-.185.102-.385.102-.593V6.25a1.75 1.75 0 0 0-1.75-1.75zm-3.25 3a2.5 2.5 0 1 1 0 5a2.5 2.5 0 0 1 0-5zm0 1.5a1 1 0 1 0 0 2a1 1 0 0 0 0-2z", fill: "currentColor" })
  )
);
const documentIcon = h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 28 28" },
  h(
    "g",
    { fill: "none" },
    h("path", { d: "M6.4 2A2.4 2.4 0 0 0 4 4.4v19.2A2.4 2.4 0 0 0 6.4 26h15.2a2.4 2.4 0 0 0 2.4-2.4V11.578c0-.729-.29-1.428-.805-1.944l-6.931-6.931A2.4 2.4 0 0 0 14.567 2H6.4zm-.9 2.4a.9.9 0 0 1 .9-.9H14V10a2 2 0 0 0 2 2h6.5v11.6a.9.9 0 0 1-.9.9H6.4a.9.9 0 0 1-.9-.9V4.4zm16.44 6.1H16a.5.5 0 0 1-.5-.5V4.06l6.44 6.44z", fill: "currentColor" })
  )
);
var __awaiter$1 = globalThis && globalThis.__awaiter || function(thisArg, _arguments, P, generator) {
  function adopt(value) {
    return value instanceof P ? value : new P(function(resolve) {
      resolve(value);
    });
  }
  return new (P || (P = Promise))(function(resolve, reject) {
    function fulfilled(value) {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    }
    function rejected(value) {
      try {
        step(generator["throw"](value));
      } catch (e) {
        reject(e);
      }
    }
    function step(result) {
      result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
    }
    step((generator = generator.apply(thisArg, _arguments || [])).next());
  });
};
const buttonThemeOverrides = {
  paddingMedium: "0 3px",
  heightMedium: "24px",
  iconSizeMedium: "18px"
};
const NUploadFile = defineComponent({
  name: "UploadFile",
  props: {
    clsPrefix: {
      type: String,
      required: true
    },
    file: {
      type: Object,
      required: true
    },
    listType: {
      type: String,
      required: true
    }
  },
  setup(props) {
    const NUpload2 = inject(uploadInjectionKey);
    const imageRef = ref(null);
    const thumbnailUrlRef = ref("");
    const progressStatusRef = computed(() => {
      const { file } = props;
      if (file.status === "finished")
        return "success";
      if (file.status === "error")
        return "error";
      return "info";
    });
    const buttonTypeRef = computed(() => {
      const { file } = props;
      if (file.status === "error")
        return "error";
      return void 0;
    });
    const showProgressRef = computed(() => {
      const { file } = props;
      return file.status === "uploading";
    });
    const showCancelButtonRef = computed(() => {
      if (!NUpload2.showCancelButtonRef.value)
        return false;
      const { file } = props;
      return ["uploading", "pending", "error"].includes(file.status);
    });
    const showRemoveButtonRef = computed(() => {
      if (!NUpload2.showRemoveButtonRef.value)
        return false;
      const { file } = props;
      return ["finished"].includes(file.status);
    });
    const showDownloadButtonRef = computed(() => {
      if (!NUpload2.showDownloadButtonRef.value)
        return false;
      const { file } = props;
      return ["finished"].includes(file.status);
    });
    const showRetryButtonRef = computed(() => {
      if (!NUpload2.showRetryButtonRef.value)
        return false;
      const { file } = props;
      return ["error"].includes(file.status);
    });
    const mergedThumbnailUrlRef = useMemo(() => {
      return thumbnailUrlRef.value || props.file.thumbnailUrl || props.file.url;
    });
    const showPreviewButtonRef = computed(() => {
      if (!NUpload2.showPreviewButtonRef.value)
        return false;
      const { file: { status }, listType } = props;
      return ["finished"].includes(status) && mergedThumbnailUrlRef.value && listType === "image-card";
    });
    function handleRetryClick() {
      NUpload2.submit(props.file.id);
    }
    function handleRemoveOrCancelClick(e) {
      e.preventDefault();
      const { file } = props;
      if (["finished", "pending", "error"].includes(file.status)) {
        handleRemove(file);
      } else if (["uploading"].includes(file.status)) {
        handleAbort(file);
      } else {
        warn("upload", "The button clicked type is unknown.");
      }
    }
    function handleDownloadClick(e) {
      e.preventDefault();
      handleDownload(props.file);
    }
    function handleRemove(file) {
      const { xhrMap, doChange, onRemoveRef: { value: onRemove }, mergedFileListRef: { value: mergedFileList } } = NUpload2;
      void Promise.resolve(onRemove ? onRemove({
        file: Object.assign({}, file),
        fileList: mergedFileList
      }) : true).then((result) => {
        if (result === false)
          return;
        const fileAfterChange = Object.assign({}, file, {
          status: "removed"
        });
        xhrMap.delete(file.id);
        doChange(fileAfterChange, void 0, {
          remove: true
        });
      });
    }
    function handleDownload(file) {
      const { onDownloadRef: { value: onDownload } } = NUpload2;
      void Promise.resolve(onDownload ? onDownload(Object.assign({}, file)) : true).then((res) => {
        if (res !== false) {
          download(file.url, file.name);
        }
      });
    }
    function handleAbort(file) {
      const { xhrMap } = NUpload2;
      const xhr = xhrMap.get(file.id);
      xhr === null || xhr === void 0 ? void 0 : xhr.abort();
      handleRemove(Object.assign({}, file));
    }
    function handlePreviewClick() {
      const { onPreviewRef: { value: onPreview } } = NUpload2;
      if (onPreview) {
        onPreview(props.file);
      } else if (props.listType === "image-card") {
        const { value } = imageRef;
        if (!value)
          return;
        value.click();
      }
    }
    const deriveFileThumbnailUrl = () => __awaiter$1(this, void 0, void 0, function* () {
      const { listType } = props;
      if (listType !== "image" && listType !== "image-card") {
        return;
      }
      if (NUpload2.shouldUseThumbnailUrlRef.value(props.file)) {
        thumbnailUrlRef.value = yield NUpload2.getFileThumbnailUrlResolver(props.file);
      }
    });
    watchEffect(() => {
      void deriveFileThumbnailUrl();
    });
    return {
      mergedTheme: NUpload2.mergedThemeRef,
      progressStatus: progressStatusRef,
      buttonType: buttonTypeRef,
      showProgress: showProgressRef,
      disabled: NUpload2.mergedDisabledRef,
      showCancelButton: showCancelButtonRef,
      showRemoveButton: showRemoveButtonRef,
      showDownloadButton: showDownloadButtonRef,
      showRetryButton: showRetryButtonRef,
      showPreviewButton: showPreviewButtonRef,
      mergedThumbnailUrl: mergedThumbnailUrlRef,
      shouldUseThumbnailUrl: NUpload2.shouldUseThumbnailUrlRef,
      renderIcon: NUpload2.renderIconRef,
      imageRef,
      handleRemoveOrCancelClick,
      handleDownloadClick,
      handleRetryClick,
      handlePreviewClick
    };
  },
  render() {
    const { clsPrefix, mergedTheme, listType, file, renderIcon } = this;
    let icon;
    const isImageType = listType === "image";
    const isImageCardType = listType === "image-card";
    if (isImageType || isImageCardType) {
      icon = !this.shouldUseThumbnailUrl(file) || !this.mergedThumbnailUrl ? h("span", { class: `${clsPrefix}-upload-file-info__thumbnail` }, renderIcon ? renderIcon(file) : isImageFile(file) ? h(NBaseIcon, { clsPrefix }, { default: () => imageIcon }) : h(NBaseIcon, { clsPrefix }, { default: () => documentIcon })) : h("a", { rel: "noopener noreferer", target: "_blank", href: file.url || void 0, class: `${clsPrefix}-upload-file-info__thumbnail`, onClick: this.handlePreviewClick }, listType === "image-card" ? h(NImage, { src: this.mergedThumbnailUrl || void 0, previewSrc: file.url || void 0, alt: file.name, ref: "imageRef" }) : h("img", { src: this.mergedThumbnailUrl || void 0, alt: file.name }));
    } else {
      icon = h("span", { class: `${clsPrefix}-upload-file-info__thumbnail` }, renderIcon ? renderIcon(file) : h(NBaseIcon, { clsPrefix }, { default: () => h(AttachIcon, null) }));
    }
    const progress = h(NUploadProgress, { show: this.showProgress, percentage: file.percentage || 0, status: this.progressStatus });
    const showName = listType === "text" || listType === "image";
    return h(
      "div",
      { class: [
        `${clsPrefix}-upload-file`,
        `${clsPrefix}-upload-file--${this.progressStatus}-status`,
        file.url && file.status !== "error" && listType !== "image-card" && `${clsPrefix}-upload-file--with-url`,
        `${clsPrefix}-upload-file--${listType}-type`
      ] },
      h(
        "div",
        { class: `${clsPrefix}-upload-file-info` },
        icon,
        h(
          "div",
          { class: `${clsPrefix}-upload-file-info__name` },
          showName && (file.url && file.status !== "error" ? h("a", { rel: "noopener noreferer", target: "_blank", href: file.url || void 0, onClick: this.handlePreviewClick }, file.name) : h("span", { onClick: this.handlePreviewClick }, file.name)),
          isImageType && progress
        ),
        h(
          "div",
          { class: [
            `${clsPrefix}-upload-file-info__action`,
            `${clsPrefix}-upload-file-info__action--${listType}-type`
          ] },
          this.showPreviewButton ? h(NButton, { key: "preview", quaternary: true, type: this.buttonType, onClick: this.handlePreviewClick, theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, builtinThemeOverrides: buttonThemeOverrides }, {
            icon: () => h(NBaseIcon, { clsPrefix }, { default: () => h(EyeIcon, null) })
          }) : null,
          (this.showRemoveButton || this.showCancelButton) && !this.disabled && h(NButton, { key: "cancelOrTrash", theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, quaternary: true, builtinThemeOverrides: buttonThemeOverrides, type: this.buttonType, onClick: this.handleRemoveOrCancelClick }, {
            icon: () => h(NIconSwitchTransition, null, {
              default: () => this.showRemoveButton ? h(NBaseIcon, { clsPrefix, key: "trash" }, { default: () => h(TrashIcon, null) }) : h(NBaseIcon, { clsPrefix, key: "cancel" }, { default: () => h(CancelIcon, null) })
            })
          }),
          this.showRetryButton && !this.disabled && h(NButton, { key: "retry", quaternary: true, type: this.buttonType, onClick: this.handleRetryClick, theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, builtinThemeOverrides: buttonThemeOverrides }, {
            icon: () => h(NBaseIcon, { clsPrefix }, { default: () => h(RetryIcon, null) })
          }),
          this.showDownloadButton ? h(NButton, { key: "download", quaternary: true, type: this.buttonType, onClick: this.handleDownloadClick, theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, builtinThemeOverrides: buttonThemeOverrides }, {
            icon: () => h(NBaseIcon, { clsPrefix }, { default: () => h(DownloadIcon, null) })
          }) : null
        )
      ),
      !isImageType && progress
    );
  }
});
const NUploadFileList = defineComponent({
  name: "UploadFileList",
  setup(_, { slots }) {
    const NUpload2 = inject(uploadInjectionKey, null);
    if (!NUpload2) {
      throwError("upload-file-list", "`n-upload-file-list` must be placed inside `n-upload`.");
    }
    const { abstractRef, mergedClsPrefixRef, listTypeRef, mergedFileListRef, fileListStyleRef, cssVarsRef, themeClassRef, maxReachedRef, showTriggerRef, imageGroupPropsRef } = NUpload2;
    const isImageCardTypeRef = computed(() => listTypeRef.value === "image-card");
    const renderFileList = () => mergedFileListRef.value.map((file) => h(NUploadFile, { clsPrefix: mergedClsPrefixRef.value, key: file.id, file, listType: listTypeRef.value }));
    const renderUploadFileList = () => isImageCardTypeRef.value ? h(NImageGroup, Object.assign({}, imageGroupPropsRef.value), { default: renderFileList }) : h(NFadeInExpandTransition, { group: true }, {
      default: renderFileList
    });
    return () => {
      const { value: mergedClsPrefix } = mergedClsPrefixRef;
      const { value: abstract } = abstractRef;
      return h(
        "div",
        { class: [
          `${mergedClsPrefix}-upload-file-list`,
          isImageCardTypeRef.value && `${mergedClsPrefix}-upload-file-list--grid`,
          abstract ? themeClassRef === null || themeClassRef === void 0 ? void 0 : themeClassRef.value : void 0
        ], style: [
          abstract && cssVarsRef ? cssVarsRef.value : "",
          fileListStyleRef.value
        ] },
        renderUploadFileList(),
        showTriggerRef.value && !maxReachedRef.value && isImageCardTypeRef.value && h(NUploadTrigger, null, slots)
      );
    };
  }
});
const style = c([cB("upload", "width: 100%;", [cM("dragger-inside", [cB("upload-trigger", `
 display: block;
 `)]), cM("drag-over", [cB("upload-dragger", `
 border: var(--n-dragger-border-hover);
 `)])]), cB("upload-dragger", `
 cursor: pointer;
 box-sizing: border-box;
 width: 100%;
 text-align: center;
 border-radius: var(--n-border-radius);
 padding: 24px;
 opacity: 1;
 transition:
 opacity .3s var(--n-bezier),
 border-color .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 background-color: var(--n-dragger-color);
 border: var(--n-dragger-border);
 `, [c("&:hover", `
 border: var(--n-dragger-border-hover);
 `), cM("disabled", `
 cursor: not-allowed;
 `)]), cB("upload-trigger", `
 display: inline-block;
 box-sizing: border-box;
 opacity: 1;
 transition: opacity .3s var(--n-bezier);
 `, [c("+", [cB("upload-file-list", "margin-top: 8px;")]), cM("disabled", `
 opacity: var(--n-item-disabled-opacity);
 cursor: not-allowed;
 `), cM("image-card", `
 width: 96px;
 height: 96px;
 `, [cB("base-icon", `
 font-size: 24px;
 `), cB("upload-dragger", `
 padding: 0;
 height: 100%;
 width: 100%;
 display: flex;
 align-items: center;
 justify-content: center;
 `)])]), cB("upload-file-list", `
 line-height: var(--n-line-height);
 opacity: 1;
 transition: opacity .3s var(--n-bezier);
 `, [c("a, img", "outline: none;"), cM("disabled", `
 opacity: var(--n-item-disabled-opacity);
 cursor: not-allowed;
 `, [cB("upload-file", "cursor: not-allowed;")]), cM("grid", `
 display: grid;
 grid-template-columns: repeat(auto-fill, 96px);
 grid-gap: 8px;
 margin-top: 0;
 `), cB("upload-file", `
 display: block;
 box-sizing: border-box;
 cursor: default;
 padding: 0px 12px 0 6px;
 transition: background-color .3s var(--n-bezier);
 border-radius: var(--n-border-radius);
 `, [fadeInHeightExpandTransition(), cB("progress", [fadeInHeightExpandTransition({
  foldPadding: true
})]), c("&:hover", `
 background-color: var(--n-item-color-hover);
 `, [cB("upload-file-info", [cE("action", `
 opacity: 1;
 `)])]), cM("image-type", `
 border-radius: var(--n-border-radius);
 text-decoration: underline;
 text-decoration-color: #0000;
 `, [cB("upload-file-info", `
 padding-top: 0px;
 padding-bottom: 0px;
 width: 100%;
 height: 100%;
 display: flex;
 justify-content: space-between;
 align-items: center;
 padding: 6px 0;
 `, [cB("progress", `
 padding: 2px 0;
 margin-bottom: 0;
 `), cE("name", `
 padding: 0 8px;
 `), cE("thumbnail", `
 width: 32px;
 height: 32px;
 font-size: 28px;
 display: flex;
 justify-content: center;
 align-items: center;
 `, [c("img", `
 width: 100%;
 `)])])]), cM("text-type", [cB("progress", `
 box-sizing: border-box;
 padding-bottom: 6px;
 margin-bottom: 6px;
 `)]), cM("image-card-type", `
 position: relative;
 width: 96px;
 height: 96px;
 border: var(--n-item-border-image-card);
 border-radius: var(--n-border-radius);
 padding: 0;
 display: flex;
 align-items: center;
 justify-content: center;
 transition: border-color .3s var(--n-bezier), background-color .3s var(--n-bezier);
 border-radius: var(--n-border-radius);
 overflow: hidden;
 `, [cB("progress", `
 position: absolute;
 left: 8px;
 bottom: 8px;
 right: 8px;
 width: unset;
 `), cB("upload-file-info", `
 padding: 0;
 width: 100%;
 height: 100%;
 `, [cE("thumbnail", `
 width: 100%;
 height: 100%;
 display: flex;
 flex-direction: column;
 align-items: center;
 justify-content: center;
 font-size: 36px;
 `, [c("img", `
 width: 100%;
 `)])]), c("&::before", `
 position: absolute;
 z-index: 1;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 border-radius: inherit;
 opacity: 0;
 transition: opacity .2s var(--n-bezier);
 content: "";
 `), c("&:hover", [c("&::before", "opacity: 1;"), cB("upload-file-info", [cE("thumbnail", "opacity: .12;")])])]), cM("error-status", [c("&:hover", `
 background-color: var(--n-item-color-hover-error);
 `), cB("upload-file-info", [cE("name", "color: var(--n-item-text-color-error);"), cE("thumbnail", "color: var(--n-item-text-color-error);")]), cM("image-card-type", `
 border: var(--n-item-border-image-card-error);
 `)]), cM("with-url", `
 cursor: pointer;
 `, [cB("upload-file-info", [cE("name", `
 color: var(--n-item-text-color-success);
 text-decoration-color: var(--n-item-text-color-success);
 `, [c("a", `
 text-decoration: underline;
 `)])])]), cB("upload-file-info", `
 position: relative;
 padding-top: 6px;
 padding-bottom: 6px;
 display: flex;
 flex-wrap: nowrap;
 `, [cE("thumbnail", `
 font-size: 18px;
 opacity: 1;
 transition: opacity .2s var(--n-bezier);
 color: var(--n-item-icon-color);
 `, [cB("base-icon", `
 margin-right: 2px;
 vertical-align: middle;
 transition: color .3s var(--n-bezier);
 `)]), cE("action", `
 padding-top: inherit;
 padding-bottom: inherit;
 position: absolute;
 right: 0;
 top: 0;
 bottom: 0;
 width: 80px;
 display: flex;
 align-items: center;
 transition: opacity .2s var(--n-bezier);
 justify-content: flex-end;
 opacity: 0;
 `, [cB("button", [c("&:not(:last-child)", {
  marginRight: "4px"
}), cB("base-icon", [c("svg", [iconSwitchTransition()])])]), cM("image-type", `
 position: relative;
 max-width: 80px;
 width: auto;
 `), cM("image-card-type", `
 z-index: 2;
 position: absolute;
 width: 100%;
 height: 100%;
 left: 0;
 right: 0;
 bottom: 0;
 top: 0;
 display: flex;
 justify-content: center;
 align-items: center;
 `)]), cE("name", `
 color: var(--n-item-text-color);
 flex: 1;
 display: flex;
 justify-content: center;
 text-overflow: ellipsis;
 overflow: hidden;
 flex-direction: column;
 text-decoration-color: #0000;
 font-size: var(--n-font-size);
 transition:
 color .3s var(--n-bezier),
 text-decoration-color .3s var(--n-bezier); 
 `, [c("a", `
 color: inherit;
 text-decoration: underline;
 `)])])])]), cB("upload-file-input", `
 display: block;
 width: 0;
 height: 0;
 opacity: 0;
 `)]);
var __awaiter = globalThis && globalThis.__awaiter || function(thisArg, _arguments, P, generator) {
  function adopt(value) {
    return value instanceof P ? value : new P(function(resolve) {
      resolve(value);
    });
  }
  return new (P || (P = Promise))(function(resolve, reject) {
    function fulfilled(value) {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    }
    function rejected(value) {
      try {
        step(generator["throw"](value));
      } catch (e) {
        reject(e);
      }
    }
    function step(result) {
      result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
    }
    step((generator = generator.apply(thisArg, _arguments || [])).next());
  });
};
function createXhrHandlers(inst, file, xhr) {
  const { doChange, xhrMap } = inst;
  let percentage = 0;
  function handleXHRError(e) {
    var _a;
    let fileAfterChange = Object.assign({}, file, {
      status: "error",
      percentage
    });
    xhrMap.delete(file.id);
    fileAfterChange = createSettledFileInfo(((_a = inst.onError) === null || _a === void 0 ? void 0 : _a.call(inst, { file: fileAfterChange, event: e })) || fileAfterChange);
    doChange(fileAfterChange, e);
  }
  function handleXHRLoad(e) {
    var _a;
    if (inst.isErrorState) {
      if (inst.isErrorState(xhr)) {
        handleXHRError(e);
        return;
      }
    } else {
      if (xhr.status < 200 || xhr.status >= 300) {
        handleXHRError(e);
        return;
      }
    }
    let fileAfterChange = Object.assign({}, file, {
      status: "finished",
      percentage
    });
    xhrMap.delete(file.id);
    fileAfterChange = createSettledFileInfo(((_a = inst.onFinish) === null || _a === void 0 ? void 0 : _a.call(inst, { file: fileAfterChange, event: e })) || fileAfterChange);
    doChange(fileAfterChange, e);
  }
  return {
    handleXHRLoad,
    handleXHRError,
    handleXHRAbort(e) {
      const fileAfterChange = Object.assign({}, file, {
        status: "removed",
        file: null,
        percentage
      });
      xhrMap.delete(file.id);
      doChange(fileAfterChange, e);
    },
    handleXHRProgress(e) {
      const fileAfterChange = Object.assign({}, file, {
        status: "uploading"
      });
      if (e.lengthComputable) {
        const progress = Math.ceil(e.loaded / e.total * 100);
        fileAfterChange.percentage = progress;
        percentage = progress;
      }
      doChange(fileAfterChange, e);
    }
  };
}
function customSubmitImpl(options) {
  const { inst, file, data, headers, withCredentials, action, customRequest } = options;
  const { doChange } = options.inst;
  let percentage = 0;
  customRequest({
    file,
    data,
    headers,
    withCredentials,
    action,
    onProgress(event) {
      const fileAfterChange = Object.assign({}, file, {
        status: "uploading"
      });
      const progress = event.percent;
      fileAfterChange.percentage = progress;
      percentage = progress;
      doChange(fileAfterChange);
    },
    onFinish() {
      var _a;
      let fileAfterChange = Object.assign({}, file, {
        status: "finished",
        percentage
      });
      fileAfterChange = createSettledFileInfo(((_a = inst.onFinish) === null || _a === void 0 ? void 0 : _a.call(inst, { file: fileAfterChange })) || fileAfterChange);
      doChange(fileAfterChange);
    },
    onError() {
      var _a;
      let fileAfterChange = Object.assign({}, file, {
        status: "error",
        percentage
      });
      fileAfterChange = createSettledFileInfo(((_a = inst.onError) === null || _a === void 0 ? void 0 : _a.call(inst, { file: fileAfterChange })) || fileAfterChange);
      doChange(fileAfterChange);
    }
  });
}
function registerHandler(inst, file, request) {
  const handlers = createXhrHandlers(inst, file, request);
  request.onabort = handlers.handleXHRAbort;
  request.onerror = handlers.handleXHRError;
  request.onload = handlers.handleXHRLoad;
  if (request.upload) {
    request.upload.onprogress = handlers.handleXHRProgress;
  }
}
function unwrapFunctionValue(data, file) {
  if (typeof data === "function") {
    return data({ file });
  }
  if (data)
    return data;
  return {};
}
function setHeaders(request, headers, file) {
  const headersObject = unwrapFunctionValue(headers, file);
  if (!headersObject)
    return;
  Object.keys(headersObject).forEach((key) => {
    request.setRequestHeader(key, headersObject[key]);
  });
}
function appendData(formData, data, file) {
  const dataObject = unwrapFunctionValue(data, file);
  if (!dataObject)
    return;
  Object.keys(dataObject).forEach((key) => {
    formData.append(key, dataObject[key]);
  });
}
function submitImpl(inst, fieldName, file, { method, action, withCredentials, responseType, headers, data }) {
  const request = new XMLHttpRequest();
  request.responseType = responseType;
  inst.xhrMap.set(file.id, request);
  request.withCredentials = withCredentials;
  const formData = new FormData();
  appendData(formData, data, file);
  formData.append(fieldName, file.file);
  registerHandler(inst, file, request);
  if (action !== void 0) {
    request.open(method.toUpperCase(), action);
    setHeaders(request, headers, file);
    request.send(formData);
    const fileAfterChange = Object.assign({}, file, {
      status: "uploading"
    });
    inst.doChange(fileAfterChange);
  }
}
const uploadProps = Object.assign(Object.assign({}, useTheme.props), {
  name: {
    type: String,
    default: "file"
  },
  accept: String,
  action: String,
  customRequest: Function,
  directory: Boolean,
  directoryDnd: { type: Boolean, default: void 0 },
  method: {
    type: String,
    default: "POST"
  },
  multiple: Boolean,
  showFileList: {
    type: Boolean,
    default: true
  },
  data: [Object, Function],
  headers: [Object, Function],
  withCredentials: Boolean,
  responseType: {
    type: String,
    default: ""
  },
  disabled: {
    type: Boolean,
    default: void 0
  },
  onChange: Function,
  onRemove: Function,
  onFinish: Function,
  onError: Function,
  onBeforeUpload: Function,
  isErrorState: Function,
  /** currently not used */
  onDownload: Function,
  defaultUpload: {
    type: Boolean,
    default: true
  },
  fileList: Array,
  "onUpdate:fileList": [Function, Array],
  onUpdateFileList: [Function, Array],
  fileListStyle: [String, Object],
  defaultFileList: {
    type: Array,
    default: () => []
  },
  showCancelButton: {
    type: Boolean,
    default: true
  },
  showRemoveButton: {
    type: Boolean,
    default: true
  },
  showDownloadButton: Boolean,
  showRetryButton: {
    type: Boolean,
    default: true
  },
  showPreviewButton: {
    type: Boolean,
    default: true
  },
  listType: {
    type: String,
    default: "text"
  },
  onPreview: Function,
  shouldUseThumbnailUrl: {
    type: Function,
    default: (file) => {
      if (!environmentSupportFile)
        return false;
      return isImageFile(file);
    }
  },
  createThumbnailUrl: Function,
  abstract: Boolean,
  max: Number,
  showTrigger: {
    type: Boolean,
    default: true
  },
  imageGroupProps: Object,
  inputProps: Object,
  triggerStyle: [String, Object],
  renderIcon: Object
});
const NUpload = defineComponent({
  name: "Upload",
  props: uploadProps,
  setup(props) {
    if (props.abstract && props.listType === "image-card") {
      throwError("upload", "when the list-type is image-card, abstract is not supported.");
    }
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const themeRef = useTheme("Upload", "-upload", style, uploadLight, props, mergedClsPrefixRef);
    const formItem = useFormItem(props);
    const maxReachedRef = computed(() => {
      const { max } = props;
      if (max !== void 0) {
        return mergedFileListRef.value.length >= max;
      }
      return false;
    });
    const uncontrolledFileListRef = ref(props.defaultFileList);
    const controlledFileListRef = toRef(props, "fileList");
    const inputElRef = ref(null);
    const draggerInsideRef = {
      value: false
    };
    const dragOverRef = ref(false);
    const xhrMap = /* @__PURE__ */ new Map();
    const _mergedFileListRef = useMergedState(controlledFileListRef, uncontrolledFileListRef);
    const mergedFileListRef = computed(() => _mergedFileListRef.value.map(createSettledFileInfo));
    function openOpenFileDialog() {
      var _a;
      (_a = inputElRef.value) === null || _a === void 0 ? void 0 : _a.click();
    }
    function handleFileInputChange(e) {
      const target = e.target;
      handleFileAddition(target.files ? Array.from(target.files).map((file) => ({
        file,
        entry: null,
        source: "input"
      })) : null, e);
      target.value = "";
    }
    function doUpdateFileList(files) {
      const { "onUpdate:fileList": _onUpdateFileList, onUpdateFileList } = props;
      if (_onUpdateFileList)
        call(_onUpdateFileList, files);
      if (onUpdateFileList)
        call(onUpdateFileList, files);
      uncontrolledFileListRef.value = files;
    }
    const mergedMultipleRef = computed(() => props.multiple || props.directory);
    function handleFileAddition(fileAndEntries, e) {
      if (!fileAndEntries || fileAndEntries.length === 0)
        return;
      const { onBeforeUpload } = props;
      fileAndEntries = mergedMultipleRef.value ? fileAndEntries : [fileAndEntries[0]];
      const { max, accept } = props;
      fileAndEntries = fileAndEntries.filter(({ file, source }) => {
        if (source === "dnd" && (accept === null || accept === void 0 ? void 0 : accept.trim())) {
          return matchType(file.name, file.type, accept);
        } else {
          return true;
        }
      });
      if (max) {
        fileAndEntries = fileAndEntries.slice(0, max - mergedFileListRef.value.length);
      }
      const batchId = createId();
      void Promise.all(fileAndEntries.map(({ file, entry }) => __awaiter(this, void 0, void 0, function* () {
        var _a;
        const fileInfo = {
          id: createId(),
          batchId,
          name: file.name,
          status: "pending",
          percentage: 0,
          file,
          url: null,
          type: file.type,
          thumbnailUrl: null,
          fullPath: (_a = entry === null || entry === void 0 ? void 0 : entry.fullPath) !== null && _a !== void 0 ? _a : `/${file.webkitRelativePath || file.name}`
        };
        if (!onBeforeUpload || (yield onBeforeUpload({
          file: fileInfo,
          fileList: mergedFileListRef.value
        })) !== false) {
          return fileInfo;
        }
        return null;
      }))).then((fileInfos) => __awaiter(this, void 0, void 0, function* () {
        let nextTickChain = Promise.resolve();
        fileInfos.forEach((fileInfo) => {
          nextTickChain = nextTickChain.then(nextTick).then(() => {
            fileInfo && doChange(fileInfo, e, {
              append: true
            });
          });
        });
        yield nextTickChain;
      })).then(() => {
        if (props.defaultUpload) {
          submit();
        }
      });
    }
    function submit(fileId) {
      const { method, action, withCredentials, headers, data, name: fieldName } = props;
      const filesToUpload = fileId !== void 0 ? mergedFileListRef.value.filter((file) => file.id === fileId) : mergedFileListRef.value;
      const shouldReupload = fileId !== void 0;
      filesToUpload.forEach((file) => {
        const { status } = file;
        if (status === "pending" || status === "error" && shouldReupload) {
          if (props.customRequest) {
            customSubmitImpl({
              inst: {
                doChange,
                xhrMap,
                onFinish: props.onFinish,
                onError: props.onError
              },
              file,
              action,
              withCredentials,
              headers,
              data,
              customRequest: props.customRequest
            });
          } else {
            submitImpl({
              doChange,
              xhrMap,
              onFinish: props.onFinish,
              onError: props.onError,
              isErrorState: props.isErrorState
            }, fieldName, file, {
              method,
              action,
              withCredentials,
              responseType: props.responseType,
              headers,
              data
            });
          }
        }
      });
    }
    const doChange = (fileAfterChange, event, options = {
      append: false,
      remove: false
    }) => {
      const { append, remove } = options;
      const fileListAfterChange = Array.from(mergedFileListRef.value);
      const fileIndex = fileListAfterChange.findIndex((file) => file.id === fileAfterChange.id);
      if (append || remove || ~fileIndex) {
        if (append) {
          fileListAfterChange.push(fileAfterChange);
        } else if (remove) {
          fileListAfterChange.splice(fileIndex, 1);
        } else {
          fileListAfterChange.splice(fileIndex, 1, fileAfterChange);
        }
        const { onChange } = props;
        if (onChange) {
          onChange({
            file: fileAfterChange,
            fileList: fileListAfterChange,
            event
          });
        }
        doUpdateFileList(fileListAfterChange);
      }
    };
    function getFileThumbnailUrlResolver(file) {
      var _a;
      if (file.thumbnailUrl)
        return file.thumbnailUrl;
      const { createThumbnailUrl } = props;
      if (createThumbnailUrl) {
        return (_a = createThumbnailUrl(file.file, file)) !== null && _a !== void 0 ? _a : file.url || "";
      }
      if (file.url) {
        return file.url;
      } else if (file.file) {
        return createImageDataUrl(file.file);
      }
      return "";
    }
    const cssVarsRef = computed(() => {
      const { common: { cubicBezierEaseInOut }, self: { draggerColor, draggerBorder, draggerBorderHover, itemColorHover, itemColorHoverError, itemTextColorError, itemTextColorSuccess, itemTextColor, itemIconColor, itemDisabledOpacity, lineHeight, borderRadius, fontSize, itemBorderImageCardError, itemBorderImageCard } } = themeRef.value;
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-border-radius": borderRadius,
        "--n-dragger-border": draggerBorder,
        "--n-dragger-border-hover": draggerBorderHover,
        "--n-dragger-color": draggerColor,
        "--n-font-size": fontSize,
        "--n-item-color-hover": itemColorHover,
        "--n-item-color-hover-error": itemColorHoverError,
        "--n-item-disabled-opacity": itemDisabledOpacity,
        "--n-item-icon-color": itemIconColor,
        "--n-item-text-color": itemTextColor,
        "--n-item-text-color-error": itemTextColorError,
        "--n-item-text-color-success": itemTextColorSuccess,
        "--n-line-height": lineHeight,
        "--n-item-border-image-card-error": itemBorderImageCardError,
        "--n-item-border-image-card": itemBorderImageCard
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("upload", void 0, cssVarsRef, props) : void 0;
    provide(uploadInjectionKey, {
      mergedClsPrefixRef,
      mergedThemeRef: themeRef,
      showCancelButtonRef: toRef(props, "showCancelButton"),
      showDownloadButtonRef: toRef(props, "showDownloadButton"),
      showRemoveButtonRef: toRef(props, "showRemoveButton"),
      showRetryButtonRef: toRef(props, "showRetryButton"),
      onRemoveRef: toRef(props, "onRemove"),
      onDownloadRef: toRef(props, "onDownload"),
      mergedFileListRef,
      triggerStyleRef: toRef(props, "triggerStyle"),
      shouldUseThumbnailUrlRef: toRef(props, "shouldUseThumbnailUrl"),
      renderIconRef: toRef(props, "renderIcon"),
      xhrMap,
      submit,
      doChange,
      showPreviewButtonRef: toRef(props, "showPreviewButton"),
      onPreviewRef: toRef(props, "onPreview"),
      getFileThumbnailUrlResolver,
      listTypeRef: toRef(props, "listType"),
      dragOverRef,
      openOpenFileDialog,
      draggerInsideRef,
      handleFileAddition,
      mergedDisabledRef: formItem.mergedDisabledRef,
      maxReachedRef,
      fileListStyleRef: toRef(props, "fileListStyle"),
      abstractRef: toRef(props, "abstract"),
      acceptRef: toRef(props, "accept"),
      cssVarsRef: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClassRef: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender,
      showTriggerRef: toRef(props, "showTrigger"),
      imageGroupPropsRef: toRef(props, "imageGroupProps"),
      mergedDirectoryDndRef: computed(() => {
        var _a;
        return (_a = props.directoryDnd) !== null && _a !== void 0 ? _a : props.directory;
      })
    });
    const exposedMethods = {
      clear: () => {
        uncontrolledFileListRef.value = [];
      },
      submit,
      openOpenFileDialog
    };
    return Object.assign({
      mergedClsPrefix: mergedClsPrefixRef,
      draggerInsideRef,
      inputElRef,
      mergedTheme: themeRef,
      dragOver: dragOverRef,
      mergedMultiple: mergedMultipleRef,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender,
      handleFileInputChange
    }, exposedMethods);
  },
  render() {
    var _a, _b;
    const { draggerInsideRef, mergedClsPrefix, $slots, directory, onRender } = this;
    if ($slots.default && !this.abstract) {
      const firstChild = $slots.default()[0];
      if ((_a = firstChild === null || firstChild === void 0 ? void 0 : firstChild.type) === null || _a === void 0 ? void 0 : _a[uploadDraggerKey]) {
        draggerInsideRef.value = true;
      }
    }
    const inputNode = h("input", Object.assign({}, this.inputProps, {
      ref: "inputElRef",
      type: "file",
      class: `${mergedClsPrefix}-upload-file-input`,
      accept: this.accept,
      multiple: this.mergedMultiple,
      onChange: this.handleFileInputChange,
      // @ts-expect-error // seems vue-tsc will add the prop, so we can't use expect-error
      webkitdirectory: directory || void 0,
      directory: directory || void 0
    }));
    if (this.abstract) {
      return h(
        Fragment,
        null,
        (_b = $slots.default) === null || _b === void 0 ? void 0 : _b.call($slots),
        h(Teleport, { to: "body" }, inputNode)
      );
    }
    onRender === null || onRender === void 0 ? void 0 : onRender();
    return h(
      "div",
      { class: [
        `${mergedClsPrefix}-upload`,
        draggerInsideRef.value && `${mergedClsPrefix}-upload--dragger-inside`,
        this.dragOver && `${mergedClsPrefix}-upload--drag-over`,
        this.themeClass
      ], style: this.cssVars },
      inputNode,
      this.showTrigger && this.listType !== "image-card" && h(NUploadTrigger, null, $slots),
      this.showFileList && h(NUploadFileList, null, $slots)
    );
  }
});
const _hoisted_1$6 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$6 = /* @__PURE__ */ createBaseVNode(
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
const _hoisted_3$6 = /* @__PURE__ */ createBaseVNode(
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
const _hoisted_4$6 = [_hoisted_2$6, _hoisted_3$6];
const BuildOutline = defineComponent({
  name: "BuildOutline",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$6, _hoisted_4$6);
  }
});
const _hoisted_1$5 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$5 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M261.56 101.28a8 8 0 0 0-11.06 0L66.4 277.15a8 8 0 0 0-2.47 5.79L63.9 448a32 32 0 0 0 32 32H192a16 16 0 0 0 16-16V328a8 8 0 0 1 8-8h80a8 8 0 0 1 8 8v136a16 16 0 0 0 16 16h96.06a32 32 0 0 0 32-32V282.94a8 8 0 0 0-2.47-5.79z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$5 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M490.91 244.15l-74.8-71.56V64a16 16 0 0 0-16-16h-48a16 16 0 0 0-16 16v32l-57.92-55.38C272.77 35.14 264.71 32 256 32c-8.68 0-16.72 3.14-22.14 8.63l-212.7 203.5c-6.22 6-7 15.87-1.34 22.37A16 16 0 0 0 43 267.56L250.5 69.28a8 8 0 0 1 11.06 0l207.52 198.28a16 16 0 0 0 22.59-.44c6.14-6.36 5.63-16.86-.76-22.97z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$5 = [_hoisted_2$5, _hoisted_3$5];
const Home = defineComponent({
  name: "Home",
  render: function render2(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$5, _hoisted_4$5);
  }
});
const _hoisted_1$4 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$4 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-miterlimit": "10",
    "stroke-width": "48",
    d: "M88 152h336"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$4 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-miterlimit": "10",
    "stroke-width": "48",
    d: "M88 256h336"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$4 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-miterlimit": "10",
    "stroke-width": "48",
    d: "M88 360h336"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_5$2 = [_hoisted_2$4, _hoisted_3$4, _hoisted_4$4];
const Menu = defineComponent({
  name: "Menu",
  render: function render3(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$4, _hoisted_5$2);
  }
});
const _hoisted_1$3 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$3 = /* @__PURE__ */ createBaseVNode(
  "circle",
  {
    cx: "256",
    cy: "256",
    r: "48",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$3 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M470.39 300l-.47-.38l-31.56-24.75a16.11 16.11 0 0 1-6.1-13.33v-11.56a16 16 0 0 1 6.11-13.22L469.92 212l.47-.38a26.68 26.68 0 0 0 5.9-34.06l-42.71-73.9a1.59 1.59 0 0 1-.13-.22A26.86 26.86 0 0 0 401 92.14l-.35.13l-37.1 14.93a15.94 15.94 0 0 1-14.47-1.29q-4.92-3.1-10-5.86a15.94 15.94 0 0 1-8.19-11.82l-5.59-39.59l-.12-.72A27.22 27.22 0 0 0 298.76 26h-85.52a26.92 26.92 0 0 0-26.45 22.39l-.09.56l-5.57 39.67a16 16 0 0 1-8.13 11.82a175.21 175.21 0 0 0-10 5.82a15.92 15.92 0 0 1-14.43 1.27l-37.13-15l-.35-.14a26.87 26.87 0 0 0-32.48 11.34l-.13.22l-42.77 73.95a26.71 26.71 0 0 0 5.9 34.1l.47.38l31.56 24.75a16.11 16.11 0 0 1 6.1 13.33v11.56a16 16 0 0 1-6.11 13.22L42.08 300l-.47.38a26.68 26.68 0 0 0-5.9 34.06l42.71 73.9a1.59 1.59 0 0 1 .13.22a26.86 26.86 0 0 0 32.45 11.3l.35-.13l37.07-14.93a15.94 15.94 0 0 1 14.47 1.29q4.92 3.11 10 5.86a15.94 15.94 0 0 1 8.19 11.82l5.56 39.59l.12.72A27.22 27.22 0 0 0 213.24 486h85.52a26.92 26.92 0 0 0 26.45-22.39l.09-.56l5.57-39.67a16 16 0 0 1 8.18-11.82c3.42-1.84 6.76-3.79 10-5.82a15.92 15.92 0 0 1 14.43-1.27l37.13 14.95l.35.14a26.85 26.85 0 0 0 32.48-11.34a2.53 2.53 0 0 1 .13-.22l42.71-73.89a26.7 26.7 0 0 0-5.89-34.11zm-134.48-40.24a80 80 0 1 1-83.66-83.67a80.21 80.21 0 0 1 83.66 83.67z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$3 = [_hoisted_2$3, _hoisted_3$3];
const Settings = defineComponent({
  name: "Settings",
  render: function render4(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$3, _hoisted_4$3);
  }
});
const _sfc_main$5 = /* @__PURE__ */ defineComponent({
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
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "CivitAIDownload",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$5);
    };
  }
});
const _withScopeId = (n) => (pushScopeId("data-v-95c9f01a"), n = n(), popScopeId(), n);
const _hoisted_1$2 = { style: { "margin": "18px" } };
const _hoisted_2$2 = { style: { "width": "100%", "display": "inline-flex", "justify-content": "space-between", "align-items": "center" } };
const _hoisted_3$2 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("div", null, "Install custom models from Hugging Face", -1));
const _hoisted_4$2 = { style: { "display": "inline-flex", "align-items": "center" } };
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "HuggingfaceDownload",
  setup(__props) {
    const conf = useState();
    const message = useMessage();
    const customModel = ref("");
    function downloadModel(model) {
      const url = new URL(`${serverUrl}/api/models/download`);
      const modelName = typeof model === "string" ? model : model.value;
      url.searchParams.append("model", modelName);
      console.log(url);
      conf.state.downloading = true;
      customModel.value = "";
      message.info(`Downloading model: ${modelName}`);
      fetch(url, { method: "POST" }).then(() => {
        conf.state.downloading = false;
        message.success(`Downloaded model: ${modelName}`);
      }).catch(() => {
        conf.state.downloading = false;
        message.error(`Failed to download model: ${modelName}`);
      });
    }
    const renderIcon = (icon, size = "medium") => {
      return () => {
        return h(
          NIcon,
          {
            size
          },
          {
            default: () => h(icon)
          }
        );
      };
    };
    function getPluginOptions(row) {
      const options = [
        {
          label: "Hugging Face",
          key: "github",
          icon: renderIcon(Home),
          props: {
            onClick: () => window.open(row.huggingface_url, "_blank")
          }
        }
      ];
      return options;
    }
    const columns = [
      {
        title: "Name",
        key: "name",
        sorter: "default"
      },
      {
        title: "Repository",
        key: "huggingface_id",
        sorter: "default"
      },
      {
        title: "Download",
        key: "download",
        render(row) {
          return h(
            NButton,
            {
              type: "primary",
              secondary: true,
              round: true,
              block: true,
              bordered: false,
              disabled: conf.state.downloading,
              onClick: () => {
                downloadModel(row.huggingface_id);
              }
            },
            { default: () => "Download" }
          );
        }
      },
      {
        title: "",
        width: 60,
        key: "menu",
        render(row) {
          return h(
            NDropdown,
            {
              trigger: "hover",
              options: getPluginOptions(row),
              disabled: conf.state.downloading
            },
            { default: renderIcon(Menu) }
          );
        },
        filter: "default"
      }
    ];
    const modelData = reactive([]);
    const modelFilter = ref("");
    const dataRef = computed(() => {
      if (modelFilter.value !== "") {
        return modelData.filter(
          (model) => model.name.toLowerCase().includes(modelFilter.value.toLowerCase())
        );
      } else {
        return modelData;
      }
    });
    const pagination = reactive({ pageSize: 10 });
    fetch(huggingfaceModelsFile).then((res) => {
      res.json().then((data) => {
        modelData.push(...data["models"]);
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$2, [
        createVNode(unref(NCard), {
          title: "Custom model",
          segmented: ""
        }, {
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_2$2, [
              _hoisted_3$2,
              createBaseVNode("div", _hoisted_4$2, [
                createVNode(unref(NInput), {
                  value: customModel.value,
                  "onUpdate:value": _cache[0] || (_cache[0] = ($event) => customModel.value = $event),
                  placeholder: "Azher/Anything-v4.5-vae-fp16-diffuser",
                  style: { "width": "350px" }
                }, null, 8, ["value"]),
                createVNode(unref(NButton), {
                  type: "primary",
                  bordered: "",
                  onClick: _cache[1] || (_cache[1] = ($event) => downloadModel(customModel.value)),
                  loading: unref(conf).state.downloading,
                  disabled: unref(conf).state.downloading || customModel.value === "",
                  secondary: "",
                  style: { "margin-right": "16px", "margin-left": "4px" }
                }, {
                  default: withCtx(() => [
                    createTextVNode("Install")
                  ]),
                  _: 1
                }, 8, ["loading", "disabled"])
              ])
            ])
          ]),
          _: 1
        }),
        createVNode(unref(NCard), {
          title: "Currated models",
          style: { "margin-top": "12px" },
          segmented: ""
        }, {
          default: withCtx(() => [
            createVNode(unref(NInput), {
              value: modelFilter.value,
              "onUpdate:value": _cache[2] || (_cache[2] = ($event) => modelFilter.value = $event),
              style: { "margin-bottom": "12px" },
              placeholder: "Filter",
              clearable: ""
            }, null, 8, ["value"]),
            createVNode(unref(NDataTable), {
              columns,
              data: dataRef.value,
              pagination,
              bordered: true,
              style: { "padding-bottom": "24px" }
            }, null, 8, ["data", "pagination"])
          ]),
          _: 1
        })
      ]);
    };
  }
});
const HuggingfaceDownload_vue_vue_type_style_index_0_scoped_95c9f01a_lang = "";
const HuggingfaceDownload = /* @__PURE__ */ _export_sfc(_sfc_main$3, [["__scopeId", "data-v-95c9f01a"]]);
const _hoisted_1$1 = { style: { "margin": "16px" } };
const _hoisted_2$1 = { class: "flex-container" };
const _hoisted_3$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "FP32", -1);
const _hoisted_4$1 = { class: "flex-container" };
const _hoisted_5$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Output in safetensors format", -1);
const _hoisted_6$1 = { class: "flex-container" };
const _hoisted_7$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Model", -1);
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "ModelConvert",
  setup(__props) {
    const message = useMessage();
    const model = ref("");
    const modelOptions = reactive([]);
    const building = ref(false);
    const use_fp32 = ref(false);
    const safetensors = ref(false);
    const showUnloadModal = ref(false);
    fetch(`${serverUrl}/api/models/available`).then((res) => {
      res.json().then((data) => {
        modelOptions.splice(0, modelOptions.length);
        const pyTorch = data.filter((x) => x.backend === "PyTorch");
        if (pyTorch) {
          for (const model2 of pyTorch) {
            modelOptions.push({
              label: model2.name,
              value: model2.name,
              disabled: !model2.valid
            });
          }
        }
      });
    });
    const accelerateUnload = async () => {
      try {
        await fetch(`${serverUrl}/api/models/unload-all`, {
          method: "POST"
        });
        showUnloadModal.value = false;
        await convert();
      } catch {
        showUnloadModal.value = false;
        message.error("Failed to unload, check the console for more info.");
      }
    };
    const convert = async () => {
      showUnloadModal.value = false;
      building.value = true;
      await fetch(`${serverUrl}/api/generate/convert-model`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: model.value,
          use_fp32: use_fp32.value,
          safetensors: safetensors.value
        })
      }).then(() => {
        building.value = false;
      }).catch(() => {
        building.value = false;
        message.error("Failed to accelerate, check the console for more info.");
      });
    };
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
        createVNode(unref(NCard), { style: { "margin-top": "16px" } }, {
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_2$1, [
              _hoisted_3$1,
              createVNode(unref(NSwitch), {
                value: use_fp32.value,
                "onUpdate:value": _cache[0] || (_cache[0] = ($event) => use_fp32.value = $event)
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_4$1, [
              _hoisted_5$1,
              createVNode(unref(NSwitch), {
                value: safetensors.value,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => safetensors.value = $event)
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_6$1, [
              _hoisted_7$1,
              createVNode(unref(NSelect), {
                value: model.value,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => model.value = $event),
                options: modelOptions,
                style: { "margin-right": "12px" }
              }, null, 8, ["value", "options"])
            ])
          ]),
          _: 1
        }),
        createVNode(unref(NSpace), {
          vertical: "",
          justify: "center",
          style: { "width": "100%" },
          align: "center"
        }, {
          default: withCtx(() => [
            createVNode(unref(NButton), {
              style: { "margin-top": "16px", "padding": "0 92px" },
              type: "success",
              ghost: "",
              loading: building.value,
              disabled: building.value || modelOptions.length === 0,
              onClick: _cache[3] || (_cache[3] = ($event) => showUnloadModal.value = true)
            }, {
              default: withCtx(() => [
                createTextVNode("Convert")
              ]),
              _: 1
            }, 8, ["loading", "disabled"])
          ]),
          _: 1
        }),
        createVNode(unref(NModal), {
          show: showUnloadModal.value,
          "onUpdate:show": _cache[4] || (_cache[4] = ($event) => showUnloadModal.value = $event),
          preset: "dialog",
          title: "Unload other models",
          width: "400px",
          closable: false,
          "show-close": false,
          content: "Acceleration can be done with the other models loaded as well, but it will take a lot of resources. It is recommended to unload the other models before accelerating. Do you want to unload the other models?",
          "positive-text": "Unload models",
          "negative-text": "Keep models",
          onPositiveClick: accelerateUnload,
          onNegativeClick: convert
        }, null, 8, ["show"])
      ]);
    };
  }
});
const _hoisted_1 = { style: { "padding": "12px" } };
const _hoisted_2 = { style: { "margin-bottom": "12px", "display": "block" } };
const _hoisted_3 = { style: { "display": "inline-flex" } };
const _hoisted_4 = { style: { "margin-bottom": "12px", "display": "block" } };
const _hoisted_5 = { style: { "display": "inline-flex" } };
const _hoisted_6 = { style: { "margin-bottom": "12px", "display": "block" } };
const _hoisted_7 = { style: { "display": "inline-flex" } };
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "ModelManager",
  setup(__props) {
    const global = useState();
    const filter = ref("");
    const message = useMessage();
    const renderIcon = (icon) => {
      return () => {
        return h(NIcon, null, {
          default: () => h(icon)
        });
      };
    };
    const filteredModels = computed(() => {
      return global.state.models.filter((model) => {
        return model.path.toLowerCase().includes(filter.value.toLowerCase()) || filter.value === "";
      }).sort((a, b) => a.name.toLowerCase() < b.name.toLowerCase() ? -1 : 1);
    });
    const pyTorchModels = computed(() => {
      return filteredModels.value.filter((model) => {
        return model.backend === "PyTorch" && model.valid === true;
      });
    });
    const loraModels = computed(() => {
      return filteredModels.value.filter((model) => {
        return model.backend === "LoRA";
      });
    });
    const textualInversionModels = computed(() => {
      return filteredModels.value.filter((model) => {
        return model.backend === "Textual Inversion";
      });
    });
    function createPyTorchOptions(model_path) {
      return [
        {
          label: "Delete",
          key: `delete:${model_path}`,
          icon: renderIcon(TrashBin)
        }
        // {
        //   label: "Convert",
        //   key: `convert:${model_path}`,
        //   icon: renderIcon(GitCompare),
        // },
        // {
        //   label: "Accelerate",
        //   key: `accelerate:${model_path}`,
        //   icon: renderIcon(PlayForward),
        // },
      ];
    }
    function createLoraOptions(model_path) {
      return [
        {
          label: "Delete",
          key: `delete:${model_path}`,
          icon: renderIcon(TrashBin)
        }
      ];
    }
    function createTextualInversionOptions(model_path) {
      return [
        {
          label: "Delete",
          key: `delete:${model_path}`,
          icon: renderIcon(TrashBin)
        }
      ];
    }
    function deleteModel(model_path, model_type) {
      fetch(`${serverUrl}/api/models/delete-model`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model_path,
          model_type
        })
      }).then((response) => response.json()).then(() => {
        message.success("Model deleted");
      }).catch((error) => {
        message.error(error);
      });
    }
    function handlePyTorchModelAction(key) {
      const [action, model_path] = key.split(":");
      if (action === "delete") {
        deleteModel(model_path, "pytorch");
      } else if (action === "convert") {
        message.success(key);
      } else if (action === "accelerate") {
        message.success(key);
      }
    }
    function handleLoraModelAction(key) {
      const [action, model_path] = key.split(":");
      if (action === "delete") {
        deleteModel(model_path, "lora");
      }
    }
    function handleTextualInversionModelAction(key) {
      const [action, model_path] = key.split(":");
      if (action === "delete") {
        deleteModel(model_path, "textual-inversion");
      }
    }
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NInput), {
          value: filter.value,
          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => filter.value = $event),
          style: { "width": "100%", "margin-bottom": "12px" },
          placeholder: "Filter",
          clearable: ""
        }, null, 8, ["value"]),
        createVNode(unref(NGrid), {
          cols: "3",
          "x-gap": "12"
        }, {
          default: withCtx(() => [
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(unref(NCard), { title: "Model" }, {
                  default: withCtx(() => [
                    createVNode(unref(NUpload), {
                      multiple: "",
                      "directory-dnd": "",
                      action: `${unref(serverUrl)}/api/models/upload-model`,
                      max: 5,
                      accept: ".ckpt,.safetensors",
                      style: { "border-bottom": "1px solid rgb(66, 66, 71)", "padding-bottom": "12px" }
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NUploadDragger), { style: { "display": "flex", "flex-direction": "column", "align-items": "center", "justify-content": "center" } }, {
                          default: withCtx(() => [
                            createBaseVNode("div", _hoisted_2, [
                              createVNode(unref(NIcon), {
                                size: "48",
                                depth: 3
                              }, {
                                default: withCtx(() => [
                                  createVNode(unref(CloudUpload))
                                ]),
                                _: 1
                              })
                            ]),
                            createVNode(unref(NText), { style: { "font-size": "24px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" Model ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NText), { style: { "font-size": "16px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" Click or drag a model to this area to upload it to the server ")
                              ]),
                              _: 1
                            })
                          ]),
                          _: 1
                        })
                      ]),
                      _: 1
                    }, 8, ["action"]),
                    (openBlock(true), createElementBlock(Fragment, null, renderList(pyTorchModels.value, (model) => {
                      return openBlock(), createElementBlock("div", {
                        style: { "display": "inline-flex", "width": "100%", "align-items": "center", "justify-content": "space-between", "border-bottom": "1px solid rgb(66, 66, 71)" },
                        key: model.path
                      }, [
                        createBaseVNode("p", null, toDisplayString(model.name), 1),
                        createBaseVNode("div", _hoisted_3, [
                          createVNode(unref(NDropdown), {
                            options: createPyTorchOptions(model.path),
                            placement: "right",
                            onSelect: handlePyTorchModelAction
                          }, {
                            default: withCtx(() => [
                              createVNode(unref(NButton), {
                                "render-icon": renderIcon(unref(Settings))
                              }, null, 8, ["render-icon"])
                            ]),
                            _: 2
                          }, 1032, ["options"])
                        ])
                      ]);
                    }), 128))
                  ]),
                  _: 1
                })
              ]),
              _: 1
            }),
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(unref(NCard), { title: "LoRA" }, {
                  default: withCtx(() => [
                    createVNode(unref(NUpload), {
                      multiple: "",
                      "directory-dnd": "",
                      action: `${unref(serverUrl)}/api/models/upload-model?type=lora`,
                      max: 5,
                      accept: ".ckpt,.safetensors",
                      style: { "border-bottom": "1px solid rgb(66, 66, 71)", "padding-bottom": "12px" }
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NUploadDragger), { style: { "display": "flex", "flex-direction": "column", "align-items": "center", "justify-content": "center" } }, {
                          default: withCtx(() => [
                            createBaseVNode("div", _hoisted_4, [
                              createVNode(unref(NIcon), {
                                size: "48",
                                depth: 3
                              }, {
                                default: withCtx(() => [
                                  createVNode(unref(CloudUpload))
                                ]),
                                _: 1
                              })
                            ]),
                            createVNode(unref(NText), { style: { "font-size": "24px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" LoRA ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NText), { style: { "font-size": "16px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" Click or drag a model to this area to upload it to the server ")
                              ]),
                              _: 1
                            })
                          ]),
                          _: 1
                        })
                      ]),
                      _: 1
                    }, 8, ["action"]),
                    (openBlock(true), createElementBlock(Fragment, null, renderList(loraModels.value, (model) => {
                      return openBlock(), createElementBlock("div", {
                        style: { "display": "inline-flex", "width": "100%", "align-items": "center", "justify-content": "space-between", "border-bottom": "1px solid rgb(66, 66, 71)" },
                        key: model.path
                      }, [
                        createBaseVNode("p", null, toDisplayString(model.name), 1),
                        createBaseVNode("div", _hoisted_5, [
                          createVNode(unref(NDropdown), {
                            options: createLoraOptions(model.path),
                            placement: "right",
                            onSelect: handleLoraModelAction
                          }, {
                            default: withCtx(() => [
                              createVNode(unref(NButton), {
                                "render-icon": renderIcon(unref(Settings))
                              }, null, 8, ["render-icon"])
                            ]),
                            _: 2
                          }, 1032, ["options"])
                        ])
                      ]);
                    }), 128))
                  ]),
                  _: 1
                })
              ]),
              _: 1
            }),
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(unref(NCard), { title: "Textual Inversion" }, {
                  default: withCtx(() => [
                    createVNode(unref(NUpload), {
                      multiple: "",
                      "directory-dnd": "",
                      action: `${unref(serverUrl)}/api/models/upload-model?type=textual-inversion`,
                      max: 5,
                      accept: ".pt,.safetensors",
                      style: { "border-bottom": "1px solid rgb(66, 66, 71)", "padding-bottom": "12px" }
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NUploadDragger), { style: { "display": "flex", "flex-direction": "column", "align-items": "center", "justify-content": "center" } }, {
                          default: withCtx(() => [
                            createBaseVNode("div", _hoisted_6, [
                              createVNode(unref(NIcon), {
                                size: "48",
                                depth: 3
                              }, {
                                default: withCtx(() => [
                                  createVNode(unref(CloudUpload))
                                ]),
                                _: 1
                              })
                            ]),
                            createVNode(unref(NText), { style: { "font-size": "24px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" Textual Inversion ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NText), { style: { "font-size": "16px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" Click or drag a model to this area to upload it to the server ")
                              ]),
                              _: 1
                            })
                          ]),
                          _: 1
                        })
                      ]),
                      _: 1
                    }, 8, ["action"]),
                    (openBlock(true), createElementBlock(Fragment, null, renderList(textualInversionModels.value, (model) => {
                      return openBlock(), createElementBlock("div", {
                        style: { "display": "inline-flex", "width": "100%", "align-items": "center", "justify-content": "space-between", "border-bottom": "1px solid rgb(66, 66, 71)" },
                        key: model.path
                      }, [
                        createBaseVNode("p", null, toDisplayString(model.name), 1),
                        createBaseVNode("div", _hoisted_7, [
                          createVNode(unref(NDropdown), {
                            options: createTextualInversionOptions(model.path),
                            placement: "right",
                            onSelect: handleTextualInversionModelAction
                          }, {
                            default: withCtx(() => [
                              createVNode(unref(NButton), {
                                "render-icon": renderIcon(unref(Settings))
                              }, null, 8, ["render-icon"])
                            ]),
                            _: 2
                          }, 1032, ["options"])
                        ])
                      ]);
                    }), 128))
                  ]),
                  _: 1
                })
              ]),
              _: 1
            })
          ]),
          _: 1
        })
      ]);
    };
  }
});
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ModelsView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NTabs), { type: "segment" }, {
        default: withCtx(() => [
          createVNode(unref(NTabPane), { name: "Manager" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$1)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Huggingface" }, {
            default: withCtx(() => [
              createVNode(HuggingfaceDownload)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "CivitAI" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$4)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Convert" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$2)
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
  _sfc_main as default
};
