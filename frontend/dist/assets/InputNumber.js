import { d as defineComponent, z as h, bh as createTheme, bi as commonLight, by as buttonLight, bz as inputLight, Q as c, R as cB, I as useConfig, Z as useTheme, al as useLocale, J as useFormItem, A as ref, M as toRef, K as useMergedState, Y as useMemo, a7 as watch, a0 as useRtl, c as computed, j as NInput, av as resolveWrappedSlot, a5 as on, bA as rgba, an as resolveSlot, ao as NBaseIcon, bB as XButton, bC as AddIcon, P as call, ac as nextTick } from "./index.js";
const RemoveIcon = defineComponent({
  name: "Remove",
  render() {
    return h(
      "svg",
      { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 512 512" },
      h("line", { x1: "400", y1: "256", x2: "112", y2: "256", style: "\n        fill: none;\n        stroke: currentColor;\n        stroke-linecap: round;\n        stroke-linejoin: round;\n        stroke-width: 32px;\n      " })
    );
  }
});
const self = (vars) => {
  const { textColorDisabled } = vars;
  return {
    iconColorDisabled: textColorDisabled
  };
};
const inputNumberLight = createTheme({
  name: "InputNumber",
  common: commonLight,
  peers: {
    Button: buttonLight,
    Input: inputLight
  },
  self
});
const inputNumberLight$1 = inputNumberLight;
function parse(value) {
  if (value === void 0 || value === null || typeof value === "string" && value.trim() === "") {
    return null;
  }
  return Number(value);
}
function isWipValue(value) {
  return value.includes(".") && (/^(-)?\d+.*(\.|0)$/.test(value) || /^\.\d+$/.test(value));
}
function validator(value) {
  if (value === void 0 || value === null)
    return true;
  if (Number.isNaN(value))
    return false;
  return true;
}
function format(value, precision) {
  if (value === void 0 || value === null)
    return "";
  return precision === void 0 ? String(value) : value.toFixed(precision);
}
function parseNumber(number) {
  if (number === null)
    return null;
  if (typeof number === "number") {
    return number;
  } else {
    const parsedNumber = Number(number);
    if (Number.isNaN(parsedNumber))
      return null;
    else {
      return parsedNumber;
    }
  }
}
const style = c([cB("input-number-suffix", `
 display: inline-block;
 margin-right: 10px;
 `), cB("input-number-prefix", `
 display: inline-block;
 margin-left: 10px;
 `)]);
const HOLDING_CHANGE_THRESHOLD = 800;
const HOLDING_CHANGE_INTERVAL = 100;
const inputNumberProps = Object.assign(Object.assign({}, useTheme.props), {
  autofocus: Boolean,
  loading: {
    type: Boolean,
    default: void 0
  },
  placeholder: String,
  defaultValue: {
    type: Number,
    default: null
  },
  value: Number,
  step: {
    type: [Number, String],
    default: 1
  },
  min: [Number, String],
  max: [Number, String],
  size: String,
  disabled: {
    type: Boolean,
    default: void 0
  },
  validator: Function,
  bordered: {
    type: Boolean,
    default: void 0
  },
  showButton: {
    type: Boolean,
    default: true
  },
  buttonPlacement: {
    type: String,
    default: "right"
  },
  readonly: Boolean,
  clearable: Boolean,
  keyboard: {
    type: Object,
    default: {}
  },
  updateValueOnInput: {
    type: Boolean,
    default: true
  },
  parse: Function,
  format: Function,
  precision: Number,
  status: String,
  "onUpdate:value": [Function, Array],
  onUpdateValue: [Function, Array],
  onFocus: [Function, Array],
  onBlur: [Function, Array],
  onClear: [Function, Array],
  // deprecated
  onChange: [Function, Array]
});
const NInputNumber = defineComponent({
  name: "InputNumber",
  props: inputNumberProps,
  setup(props) {
    const { mergedBorderedRef, mergedClsPrefixRef, mergedRtlRef } = useConfig(props);
    const themeRef = useTheme("InputNumber", "-input-number", style, inputNumberLight$1, props, mergedClsPrefixRef);
    const { localeRef } = useLocale("InputNumber");
    const formItem = useFormItem(props);
    const { mergedSizeRef, mergedDisabledRef, mergedStatusRef } = formItem;
    const inputInstRef = ref(null);
    const minusButtonInstRef = ref(null);
    const addButtonInstRef = ref(null);
    const uncontrolledValueRef = ref(props.defaultValue);
    const controlledValueRef = toRef(props, "value");
    const mergedValueRef = useMergedState(controlledValueRef, uncontrolledValueRef);
    const displayedValueRef = ref("");
    const getPrecision = (value) => {
      const fraction = String(value).split(".")[1];
      return fraction ? fraction.length : 0;
    };
    const getMaxPrecision = (currentValue) => {
      const precisions = [props.min, props.max, props.step, currentValue].map((value) => {
        if (value === void 0)
          return 0;
        return getPrecision(value);
      });
      return Math.max(...precisions);
    };
    const mergedPlaceholderRef = useMemo(() => {
      const { placeholder } = props;
      if (placeholder !== void 0)
        return placeholder;
      return localeRef.value.placeholder;
    });
    const mergedStepRef = useMemo(() => {
      const parsedNumber = parseNumber(props.step);
      if (parsedNumber !== null) {
        return parsedNumber === 0 ? 1 : Math.abs(parsedNumber);
      }
      return 1;
    });
    const mergedMinRef = useMemo(() => {
      const parsedNumber = parseNumber(props.min);
      if (parsedNumber !== null)
        return parsedNumber;
      else
        return null;
    });
    const mergedMaxRef = useMemo(() => {
      const parsedNumber = parseNumber(props.max);
      if (parsedNumber !== null)
        return parsedNumber;
      else
        return null;
    });
    const doUpdateValue = (value) => {
      const { value: mergedValue } = mergedValueRef;
      if (value === mergedValue) {
        deriveDisplayedValueFromValue();
        return;
      }
      const { "onUpdate:value": _onUpdateValue, onUpdateValue, onChange } = props;
      const { nTriggerFormInput, nTriggerFormChange } = formItem;
      if (onChange)
        call(onChange, value);
      if (onUpdateValue)
        call(onUpdateValue, value);
      if (_onUpdateValue)
        call(_onUpdateValue, value);
      uncontrolledValueRef.value = value;
      nTriggerFormInput();
      nTriggerFormChange();
    };
    const deriveValueFromDisplayedValue = ({ offset, doUpdateIfValid, fixPrecision, isInputing }) => {
      const { value: displayedValue } = displayedValueRef;
      if (isInputing && isWipValue(displayedValue)) {
        return false;
      }
      const parsedValue = (props.parse || parse)(displayedValue);
      if (parsedValue === null) {
        if (doUpdateIfValid)
          doUpdateValue(null);
        return null;
      }
      if (validator(parsedValue)) {
        const currentPrecision = getPrecision(parsedValue);
        const { precision } = props;
        if (precision !== void 0 && precision < currentPrecision && !fixPrecision) {
          return false;
        }
        let nextValue = parseFloat((parsedValue + offset).toFixed(precision !== null && precision !== void 0 ? precision : getMaxPrecision(parsedValue)));
        if (validator(nextValue)) {
          const { value: mergedMax } = mergedMaxRef;
          const { value: mergedMin } = mergedMinRef;
          if (mergedMax !== null && nextValue > mergedMax) {
            if (!doUpdateIfValid || isInputing)
              return false;
            nextValue = mergedMax;
          }
          if (mergedMin !== null && nextValue < mergedMin) {
            if (!doUpdateIfValid || isInputing)
              return false;
            nextValue = mergedMin;
          }
          if (props.validator && !props.validator(nextValue))
            return false;
          if (doUpdateIfValid)
            doUpdateValue(nextValue);
          return nextValue;
        }
      }
      return false;
    };
    const deriveDisplayedValueFromValue = () => {
      const { value: mergedValue } = mergedValueRef;
      if (validator(mergedValue)) {
        const { format: formatProp, precision } = props;
        if (formatProp) {
          displayedValueRef.value = formatProp(mergedValue);
        } else {
          if (mergedValue === null || precision === void 0 || // precision overflow
          getPrecision(mergedValue) > precision) {
            displayedValueRef.value = format(mergedValue, void 0);
          } else {
            displayedValueRef.value = format(mergedValue, precision);
          }
        }
      } else {
        displayedValueRef.value = String(mergedValue);
      }
    };
    deriveDisplayedValueFromValue();
    const displayedValueInvalidRef = useMemo(() => {
      const derivedValue = deriveValueFromDisplayedValue({
        offset: 0,
        doUpdateIfValid: false,
        isInputing: false,
        fixPrecision: false
      });
      return derivedValue === false;
    });
    const minusableRef = useMemo(() => {
      const { value: mergedValue } = mergedValueRef;
      if (props.validator && mergedValue === null) {
        return false;
      }
      const { value: mergedStep } = mergedStepRef;
      const derivedNextValue = deriveValueFromDisplayedValue({
        offset: -mergedStep,
        doUpdateIfValid: false,
        isInputing: false,
        fixPrecision: false
      });
      return derivedNextValue !== false;
    });
    const addableRef = useMemo(() => {
      const { value: mergedValue } = mergedValueRef;
      if (props.validator && mergedValue === null) {
        return false;
      }
      const { value: mergedStep } = mergedStepRef;
      const derivedNextValue = deriveValueFromDisplayedValue({
        offset: +mergedStep,
        doUpdateIfValid: false,
        isInputing: false,
        fixPrecision: false
      });
      return derivedNextValue !== false;
    });
    function doFocus(e) {
      const { onFocus } = props;
      const { nTriggerFormFocus } = formItem;
      if (onFocus)
        call(onFocus, e);
      nTriggerFormFocus();
    }
    function doBlur(e) {
      var _a, _b;
      if (e.target === ((_a = inputInstRef.value) === null || _a === void 0 ? void 0 : _a.wrapperElRef)) {
        return;
      }
      const value = deriveValueFromDisplayedValue({
        offset: 0,
        doUpdateIfValid: true,
        isInputing: false,
        fixPrecision: true
      });
      if (value !== false) {
        const inputElRef = (_b = inputInstRef.value) === null || _b === void 0 ? void 0 : _b.inputElRef;
        if (inputElRef) {
          inputElRef.value = String(value || "");
        }
        if (mergedValueRef.value === value) {
          deriveDisplayedValueFromValue();
        }
      } else {
        deriveDisplayedValueFromValue();
      }
      const { onBlur } = props;
      const { nTriggerFormBlur } = formItem;
      if (onBlur)
        call(onBlur, e);
      nTriggerFormBlur();
      void nextTick(() => {
        deriveDisplayedValueFromValue();
      });
    }
    function doClear(e) {
      const { onClear } = props;
      if (onClear)
        call(onClear, e);
    }
    function doAdd() {
      const { value: addable } = addableRef;
      if (!addable) {
        clearAddHoldTimeout();
        return;
      }
      const { value: mergedValue } = mergedValueRef;
      if (mergedValue === null) {
        if (!props.validator) {
          doUpdateValue(createValidValue());
        }
      } else {
        const { value: mergedStep } = mergedStepRef;
        deriveValueFromDisplayedValue({
          offset: mergedStep,
          doUpdateIfValid: true,
          isInputing: false,
          fixPrecision: true
        });
      }
    }
    function doMinus() {
      const { value: minusable } = minusableRef;
      if (!minusable) {
        clearMinusHoldTimeout();
        return;
      }
      const { value: mergedValue } = mergedValueRef;
      if (mergedValue === null) {
        if (!props.validator) {
          doUpdateValue(createValidValue());
        }
      } else {
        const { value: mergedStep } = mergedStepRef;
        deriveValueFromDisplayedValue({
          offset: -mergedStep,
          doUpdateIfValid: true,
          isInputing: false,
          fixPrecision: true
        });
      }
    }
    const handleFocus = doFocus;
    const handleBlur = doBlur;
    function createValidValue() {
      if (props.validator)
        return null;
      const { value: mergedMin } = mergedMinRef;
      const { value: mergedMax } = mergedMaxRef;
      if (mergedMin !== null) {
        return Math.max(0, mergedMin);
      } else if (mergedMax !== null) {
        return Math.min(0, mergedMax);
      } else {
        return 0;
      }
    }
    function handleClear(e) {
      doClear(e);
      doUpdateValue(null);
    }
    function handleMouseDown(e) {
      var _a, _b, _c;
      if ((_a = addButtonInstRef.value) === null || _a === void 0 ? void 0 : _a.$el.contains(e.target)) {
        e.preventDefault();
      }
      if ((_b = minusButtonInstRef.value) === null || _b === void 0 ? void 0 : _b.$el.contains(e.target)) {
        e.preventDefault();
      }
      (_c = inputInstRef.value) === null || _c === void 0 ? void 0 : _c.activate();
    }
    let minusHoldStateIntervalId = null;
    let addHoldStateIntervalId = null;
    let firstMinusMousedownId = null;
    function clearMinusHoldTimeout() {
      if (firstMinusMousedownId) {
        window.clearTimeout(firstMinusMousedownId);
        firstMinusMousedownId = null;
      }
      if (minusHoldStateIntervalId) {
        window.clearInterval(minusHoldStateIntervalId);
        minusHoldStateIntervalId = null;
      }
    }
    function clearAddHoldTimeout() {
      if (firstAddMousedownId) {
        window.clearTimeout(firstAddMousedownId);
        firstAddMousedownId = null;
      }
      if (addHoldStateIntervalId) {
        window.clearInterval(addHoldStateIntervalId);
        addHoldStateIntervalId = null;
      }
    }
    function handleMinusMousedown() {
      clearMinusHoldTimeout();
      firstMinusMousedownId = window.setTimeout(() => {
        minusHoldStateIntervalId = window.setInterval(() => {
          doMinus();
        }, HOLDING_CHANGE_INTERVAL);
      }, HOLDING_CHANGE_THRESHOLD);
      on("mouseup", document, clearMinusHoldTimeout, {
        once: true
      });
    }
    let firstAddMousedownId = null;
    function handleAddMousedown() {
      clearAddHoldTimeout();
      firstAddMousedownId = window.setTimeout(() => {
        addHoldStateIntervalId = window.setInterval(() => {
          doAdd();
        }, HOLDING_CHANGE_INTERVAL);
      }, HOLDING_CHANGE_THRESHOLD);
      on("mouseup", document, clearAddHoldTimeout, {
        once: true
      });
    }
    const handleAddClick = () => {
      if (addHoldStateIntervalId)
        return;
      doAdd();
    };
    const handleMinusClick = () => {
      if (minusHoldStateIntervalId)
        return;
      doMinus();
    };
    function handleKeyDown(e) {
      var _a, _b;
      if (e.key === "Enter") {
        if (e.target === ((_a = inputInstRef.value) === null || _a === void 0 ? void 0 : _a.wrapperElRef)) {
          return;
        }
        const value = deriveValueFromDisplayedValue({
          offset: 0,
          doUpdateIfValid: true,
          isInputing: false,
          fixPrecision: true
        });
        if (value !== false) {
          (_b = inputInstRef.value) === null || _b === void 0 ? void 0 : _b.deactivate();
        }
      } else if (e.key === "ArrowUp") {
        if (!addableRef.value)
          return;
        if (props.keyboard.ArrowUp === false)
          return;
        e.preventDefault();
        const value = deriveValueFromDisplayedValue({
          offset: 0,
          doUpdateIfValid: true,
          isInputing: false,
          fixPrecision: true
        });
        if (value !== false) {
          doAdd();
        }
      } else if (e.key === "ArrowDown") {
        if (!minusableRef.value)
          return;
        if (props.keyboard.ArrowDown === false)
          return;
        e.preventDefault();
        const value = deriveValueFromDisplayedValue({
          offset: 0,
          doUpdateIfValid: true,
          isInputing: false,
          fixPrecision: true
        });
        if (value !== false) {
          doMinus();
        }
      }
    }
    function handleUpdateDisplayedValue(value) {
      displayedValueRef.value = value;
      if (props.updateValueOnInput && !props.format && !props.parse && props.precision === void 0) {
        deriveValueFromDisplayedValue({
          offset: 0,
          doUpdateIfValid: true,
          isInputing: true,
          fixPrecision: false
        });
      }
    }
    watch(mergedValueRef, () => {
      deriveDisplayedValueFromValue();
    });
    const exposedMethods = {
      focus: () => {
        var _a;
        return (_a = inputInstRef.value) === null || _a === void 0 ? void 0 : _a.focus();
      },
      blur: () => {
        var _a;
        return (_a = inputInstRef.value) === null || _a === void 0 ? void 0 : _a.blur();
      }
    };
    const rtlEnabledRef = useRtl("InputNumber", mergedRtlRef, mergedClsPrefixRef);
    return Object.assign(Object.assign({}, exposedMethods), {
      rtlEnabled: rtlEnabledRef,
      inputInstRef,
      minusButtonInstRef,
      addButtonInstRef,
      mergedClsPrefix: mergedClsPrefixRef,
      mergedBordered: mergedBorderedRef,
      uncontrolledValue: uncontrolledValueRef,
      mergedValue: mergedValueRef,
      mergedPlaceholder: mergedPlaceholderRef,
      displayedValueInvalid: displayedValueInvalidRef,
      mergedSize: mergedSizeRef,
      mergedDisabled: mergedDisabledRef,
      displayedValue: displayedValueRef,
      addable: addableRef,
      minusable: minusableRef,
      mergedStatus: mergedStatusRef,
      handleFocus,
      handleBlur,
      handleClear,
      handleMouseDown,
      handleAddClick,
      handleMinusClick,
      handleAddMousedown,
      handleMinusMousedown,
      handleKeyDown,
      handleUpdateDisplayedValue,
      // theme
      mergedTheme: themeRef,
      inputThemeOverrides: {
        paddingSmall: "0 8px 0 10px",
        paddingMedium: "0 8px 0 12px",
        paddingLarge: "0 8px 0 14px"
      },
      buttonThemeOverrides: computed(() => {
        const { self: { iconColorDisabled } } = themeRef.value;
        const [r, g, b, a] = rgba(iconColorDisabled);
        return {
          textColorTextDisabled: `rgb(${r}, ${g}, ${b})`,
          opacityDisabled: `${a}`
        };
      })
    });
  },
  render() {
    const { mergedClsPrefix, $slots } = this;
    const renderMinusButton = () => {
      return h(XButton, { text: true, disabled: !this.minusable || this.mergedDisabled || this.readonly, focusable: false, theme: this.mergedTheme.peers.Button, themeOverrides: this.mergedTheme.peerOverrides.Button, builtinThemeOverrides: this.buttonThemeOverrides, onClick: this.handleMinusClick, onMousedown: this.handleMinusMousedown, ref: "minusButtonInstRef" }, {
        icon: () => resolveSlot($slots["minus-icon"], () => [
          h(NBaseIcon, { clsPrefix: mergedClsPrefix }, {
            default: () => h(RemoveIcon, null)
          })
        ])
      });
    };
    const renderAddButton = () => {
      return h(XButton, { text: true, disabled: !this.addable || this.mergedDisabled || this.readonly, focusable: false, theme: this.mergedTheme.peers.Button, themeOverrides: this.mergedTheme.peerOverrides.Button, builtinThemeOverrides: this.buttonThemeOverrides, onClick: this.handleAddClick, onMousedown: this.handleAddMousedown, ref: "addButtonInstRef" }, {
        icon: () => resolveSlot($slots["add-icon"], () => [
          h(NBaseIcon, { clsPrefix: mergedClsPrefix }, {
            default: () => h(AddIcon, null)
          })
        ])
      });
    };
    return h(
      "div",
      { class: [
        `${mergedClsPrefix}-input-number`,
        this.rtlEnabled && `${mergedClsPrefix}-input-number--rtl`
      ] },
      h(NInput, { ref: "inputInstRef", autofocus: this.autofocus, status: this.mergedStatus, bordered: this.mergedBordered, loading: this.loading, value: this.displayedValue, onUpdateValue: this.handleUpdateDisplayedValue, theme: this.mergedTheme.peers.Input, themeOverrides: this.mergedTheme.peerOverrides.Input, builtinThemeOverrides: this.inputThemeOverrides, size: this.mergedSize, placeholder: this.mergedPlaceholder, disabled: this.mergedDisabled, readonly: this.readonly, textDecoration: this.displayedValueInvalid ? "line-through" : void 0, onFocus: this.handleFocus, onBlur: this.handleBlur, onKeydown: this.handleKeyDown, onMousedown: this.handleMouseDown, onClear: this.handleClear, clearable: this.clearable, internalLoadingBeforeSuffix: true }, {
        prefix: () => {
          var _a;
          return this.showButton && this.buttonPlacement === "both" ? [
            renderMinusButton(),
            resolveWrappedSlot($slots.prefix, (children) => {
              if (children) {
                return h("span", { class: `${mergedClsPrefix}-input-number-prefix` }, children);
              }
              return null;
            })
          ] : (_a = $slots.prefix) === null || _a === void 0 ? void 0 : _a.call($slots);
        },
        suffix: () => {
          var _a;
          return this.showButton ? [
            resolveWrappedSlot($slots.suffix, (children) => {
              if (children) {
                return h("span", { class: `${mergedClsPrefix}-input-number-suffix` }, children);
              }
              return null;
            }),
            this.buttonPlacement === "right" ? renderMinusButton() : null,
            renderAddButton()
          ] : (_a = $slots.suffix) === null || _a === void 0 ? void 0 : _a.call($slots);
        }
      })
    );
  }
});
export {
  NInputNumber as N
};
