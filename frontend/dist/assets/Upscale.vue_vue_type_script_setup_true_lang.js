import { R as inject, bF as getCurrentInstance, J as watch, aB as onBeforeUnmount, Q as cB, ab as cM, aa as c, P as createInjectionKey, d as defineComponent, S as useConfig, T as useTheme, y as ref, a3 as provide, x as h, bG as formLight, a2 as keysOf, i as computed, az as formatLength, aH as get, bH as commonVariables, at as cE, X as toRef, aW as createId, bI as formItemInjectionKey, ba as onMounted, ah as createKey, Y as useThemeClass, aX as Transition, av as resolveWrappedSlot, aN as warn, u as useSettings, o as openBlock, g as createElementBlock, b as createBaseVNode, e as createVNode, f as unref, c as createBlock, w as withCtx, k as createTextVNode, l as NTooltip, bJ as isDev, j as NSpace, h as createCommentVNode, F as Fragment, bK as NAlert, N as NCard, m as NSelect, a as useState, K as upscalerOptions, C as NTabPane, D as NTabs, L as renderList, z as NButton, B as toDisplayString, bB as convertToTextString, bL as resolveDynamicComponent, bg as NModal, A as NIcon } from "./index.js";
import { N as NSwitch } from "./Switch.js";
import { N as NInputNumber } from "./InputNumber.js";
import { N as NSlider } from "./Slider.js";
import { S as Settings, a as NCheckbox } from "./Settings.js";
function useInjectionInstanceCollection(injectionName, collectionKey, registerKeyRef) {
  var _a;
  const injection = inject(injectionName, null);
  if (injection === null)
    return;
  const vm = (_a = getCurrentInstance()) === null || _a === void 0 ? void 0 : _a.proxy;
  watch(registerKeyRef, registerInstance);
  registerInstance(registerKeyRef.value);
  onBeforeUnmount(() => {
    registerInstance(void 0, registerKeyRef.value);
  });
  function registerInstance(key, oldKey) {
    const collection = injection[collectionKey];
    if (oldKey !== void 0)
      removeInstance(collection, oldKey);
    if (key !== void 0)
      addInstance(collection, key);
  }
  function removeInstance(collection, key) {
    if (!collection[key])
      collection[key] = [];
    collection[key].splice(collection[key].findIndex((instance) => instance === vm), 1);
  }
  function addInstance(collection, key) {
    if (!collection[key])
      collection[key] = [];
    if (!~collection[key].findIndex((instance) => instance === vm)) {
      collection[key].push(vm);
    }
  }
}
const style$1 = cB("form", [cM("inline", `
 width: 100%;
 display: inline-flex;
 align-items: flex-start;
 align-content: space-around;
 `, [cB("form-item", {
  width: "auto",
  marginRight: "18px"
}, [c("&:last-child", {
  marginRight: 0
})])])]);
const formInjectionKey = createInjectionKey("n-form");
const formItemInstsInjectionKey = createInjectionKey("n-form-item-insts");
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
const formProps = Object.assign(Object.assign({}, useTheme.props), { inline: Boolean, labelWidth: [Number, String], labelAlign: String, labelPlacement: {
  type: String,
  default: "top"
}, model: {
  type: Object,
  default: () => {
  }
}, rules: Object, disabled: Boolean, size: String, showRequireMark: {
  type: Boolean,
  default: void 0
}, requireMarkPlacement: String, showFeedback: {
  type: Boolean,
  default: true
}, onSubmit: {
  type: Function,
  default: (e) => {
    e.preventDefault();
  }
}, showLabel: {
  type: Boolean,
  default: void 0
}, validateMessages: Object });
const NForm = defineComponent({
  name: "Form",
  props: formProps,
  setup(props) {
    const { mergedClsPrefixRef } = useConfig(props);
    useTheme("Form", "-form", style$1, formLight, props, mergedClsPrefixRef);
    const formItems = {};
    const maxChildLabelWidthRef = ref(void 0);
    const deriveMaxChildLabelWidth = (currentWidth) => {
      const currentMaxChildLabelWidth = maxChildLabelWidthRef.value;
      if (currentMaxChildLabelWidth === void 0 || currentWidth >= currentMaxChildLabelWidth) {
        maxChildLabelWidthRef.value = currentWidth;
      }
    };
    function validate(validateCallback, shouldRuleBeApplied = () => true) {
      return __awaiter$1(this, void 0, void 0, function* () {
        yield new Promise((resolve, reject) => {
          const formItemValidationPromises = [];
          for (const key of keysOf(formItems)) {
            const formItemInstances = formItems[key];
            for (const formItemInstance of formItemInstances) {
              if (formItemInstance.path) {
                formItemValidationPromises.push(formItemInstance.internalValidate(null, shouldRuleBeApplied));
              }
            }
          }
          void Promise.all(formItemValidationPromises).then((results) => {
            if (results.some((result) => !result.valid)) {
              const errors = results.filter((result) => result.errors).map((result) => result.errors);
              if (validateCallback) {
                validateCallback(errors);
              }
              reject(errors);
            } else {
              if (validateCallback)
                validateCallback();
              resolve();
            }
          });
        });
      });
    }
    function restoreValidation() {
      for (const key of keysOf(formItems)) {
        const formItemInstances = formItems[key];
        for (const formItemInstance of formItemInstances) {
          formItemInstance.restoreValidation();
        }
      }
    }
    provide(formInjectionKey, {
      props,
      maxChildLabelWidthRef,
      deriveMaxChildLabelWidth
    });
    provide(formItemInstsInjectionKey, { formItems });
    const formExposedMethod = {
      validate,
      restoreValidation
    };
    return Object.assign(formExposedMethod, {
      mergedClsPrefix: mergedClsPrefixRef
    });
  },
  render() {
    const { mergedClsPrefix } = this;
    return h("form", { class: [
      `${mergedClsPrefix}-form`,
      this.inline && `${mergedClsPrefix}-form--inline`
    ], onSubmit: this.onSubmit }, this.$slots);
  }
});
function _extends() {
  _extends = Object.assign ? Object.assign.bind() : function(target) {
    for (var i = 1; i < arguments.length; i++) {
      var source = arguments[i];
      for (var key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
          target[key] = source[key];
        }
      }
    }
    return target;
  };
  return _extends.apply(this, arguments);
}
function _inheritsLoose(subClass, superClass) {
  subClass.prototype = Object.create(superClass.prototype);
  subClass.prototype.constructor = subClass;
  _setPrototypeOf(subClass, superClass);
}
function _getPrototypeOf(o) {
  _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf2(o2) {
    return o2.__proto__ || Object.getPrototypeOf(o2);
  };
  return _getPrototypeOf(o);
}
function _setPrototypeOf(o, p) {
  _setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf2(o2, p2) {
    o2.__proto__ = p2;
    return o2;
  };
  return _setPrototypeOf(o, p);
}
function _isNativeReflectConstruct() {
  if (typeof Reflect === "undefined" || !Reflect.construct)
    return false;
  if (Reflect.construct.sham)
    return false;
  if (typeof Proxy === "function")
    return true;
  try {
    Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
    return true;
  } catch (e) {
    return false;
  }
}
function _construct(Parent, args, Class) {
  if (_isNativeReflectConstruct()) {
    _construct = Reflect.construct.bind();
  } else {
    _construct = function _construct2(Parent2, args2, Class2) {
      var a = [null];
      a.push.apply(a, args2);
      var Constructor = Function.bind.apply(Parent2, a);
      var instance = new Constructor();
      if (Class2)
        _setPrototypeOf(instance, Class2.prototype);
      return instance;
    };
  }
  return _construct.apply(null, arguments);
}
function _isNativeFunction(fn) {
  return Function.toString.call(fn).indexOf("[native code]") !== -1;
}
function _wrapNativeSuper(Class) {
  var _cache = typeof Map === "function" ? /* @__PURE__ */ new Map() : void 0;
  _wrapNativeSuper = function _wrapNativeSuper2(Class2) {
    if (Class2 === null || !_isNativeFunction(Class2))
      return Class2;
    if (typeof Class2 !== "function") {
      throw new TypeError("Super expression must either be null or a function");
    }
    if (typeof _cache !== "undefined") {
      if (_cache.has(Class2))
        return _cache.get(Class2);
      _cache.set(Class2, Wrapper);
    }
    function Wrapper() {
      return _construct(Class2, arguments, _getPrototypeOf(this).constructor);
    }
    Wrapper.prototype = Object.create(Class2.prototype, {
      constructor: {
        value: Wrapper,
        enumerable: false,
        writable: true,
        configurable: true
      }
    });
    return _setPrototypeOf(Wrapper, Class2);
  };
  return _wrapNativeSuper(Class);
}
var formatRegExp = /%[sdj%]/g;
var warning = function warning2() {
};
if (typeof process !== "undefined" && process.env && false) {
  warning = function warning3(type4, errors) {
    if (typeof console !== "undefined" && console.warn && typeof ASYNC_VALIDATOR_NO_WARNING === "undefined") {
      if (errors.every(function(e) {
        return typeof e === "string";
      })) {
        console.warn(type4, errors);
      }
    }
  };
}
function convertFieldsError(errors) {
  if (!errors || !errors.length)
    return null;
  var fields = {};
  errors.forEach(function(error) {
    var field = error.field;
    fields[field] = fields[field] || [];
    fields[field].push(error);
  });
  return fields;
}
function format(template) {
  for (var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    args[_key - 1] = arguments[_key];
  }
  var i = 0;
  var len = args.length;
  if (typeof template === "function") {
    return template.apply(null, args);
  }
  if (typeof template === "string") {
    var str = template.replace(formatRegExp, function(x) {
      if (x === "%%") {
        return "%";
      }
      if (i >= len) {
        return x;
      }
      switch (x) {
        case "%s":
          return String(args[i++]);
        case "%d":
          return Number(args[i++]);
        case "%j":
          try {
            return JSON.stringify(args[i++]);
          } catch (_) {
            return "[Circular]";
          }
          break;
        default:
          return x;
      }
    });
    return str;
  }
  return template;
}
function isNativeStringType(type4) {
  return type4 === "string" || type4 === "url" || type4 === "hex" || type4 === "email" || type4 === "date" || type4 === "pattern";
}
function isEmptyValue(value, type4) {
  if (value === void 0 || value === null) {
    return true;
  }
  if (type4 === "array" && Array.isArray(value) && !value.length) {
    return true;
  }
  if (isNativeStringType(type4) && typeof value === "string" && !value) {
    return true;
  }
  return false;
}
function asyncParallelArray(arr, func, callback) {
  var results = [];
  var total = 0;
  var arrLength = arr.length;
  function count(errors) {
    results.push.apply(results, errors || []);
    total++;
    if (total === arrLength) {
      callback(results);
    }
  }
  arr.forEach(function(a) {
    func(a, count);
  });
}
function asyncSerialArray(arr, func, callback) {
  var index = 0;
  var arrLength = arr.length;
  function next(errors) {
    if (errors && errors.length) {
      callback(errors);
      return;
    }
    var original = index;
    index = index + 1;
    if (original < arrLength) {
      func(arr[original], next);
    } else {
      callback([]);
    }
  }
  next([]);
}
function flattenObjArr(objArr) {
  var ret = [];
  Object.keys(objArr).forEach(function(k) {
    ret.push.apply(ret, objArr[k] || []);
  });
  return ret;
}
var AsyncValidationError = /* @__PURE__ */ function(_Error) {
  _inheritsLoose(AsyncValidationError2, _Error);
  function AsyncValidationError2(errors, fields) {
    var _this;
    _this = _Error.call(this, "Async Validation Error") || this;
    _this.errors = errors;
    _this.fields = fields;
    return _this;
  }
  return AsyncValidationError2;
}(/* @__PURE__ */ _wrapNativeSuper(Error));
function asyncMap(objArr, option, func, callback, source) {
  if (option.first) {
    var _pending = new Promise(function(resolve, reject) {
      var next = function next2(errors) {
        callback(errors);
        return errors.length ? reject(new AsyncValidationError(errors, convertFieldsError(errors))) : resolve(source);
      };
      var flattenArr = flattenObjArr(objArr);
      asyncSerialArray(flattenArr, func, next);
    });
    _pending["catch"](function(e) {
      return e;
    });
    return _pending;
  }
  var firstFields = option.firstFields === true ? Object.keys(objArr) : option.firstFields || [];
  var objArrKeys = Object.keys(objArr);
  var objArrLength = objArrKeys.length;
  var total = 0;
  var results = [];
  var pending = new Promise(function(resolve, reject) {
    var next = function next2(errors) {
      results.push.apply(results, errors);
      total++;
      if (total === objArrLength) {
        callback(results);
        return results.length ? reject(new AsyncValidationError(results, convertFieldsError(results))) : resolve(source);
      }
    };
    if (!objArrKeys.length) {
      callback(results);
      resolve(source);
    }
    objArrKeys.forEach(function(key) {
      var arr = objArr[key];
      if (firstFields.indexOf(key) !== -1) {
        asyncSerialArray(arr, func, next);
      } else {
        asyncParallelArray(arr, func, next);
      }
    });
  });
  pending["catch"](function(e) {
    return e;
  });
  return pending;
}
function isErrorObj(obj) {
  return !!(obj && obj.message !== void 0);
}
function getValue(value, path) {
  var v = value;
  for (var i = 0; i < path.length; i++) {
    if (v == void 0) {
      return v;
    }
    v = v[path[i]];
  }
  return v;
}
function complementError(rule, source) {
  return function(oe) {
    var fieldValue;
    if (rule.fullFields) {
      fieldValue = getValue(source, rule.fullFields);
    } else {
      fieldValue = source[oe.field || rule.fullField];
    }
    if (isErrorObj(oe)) {
      oe.field = oe.field || rule.fullField;
      oe.fieldValue = fieldValue;
      return oe;
    }
    return {
      message: typeof oe === "function" ? oe() : oe,
      fieldValue,
      field: oe.field || rule.fullField
    };
  };
}
function deepMerge(target, source) {
  if (source) {
    for (var s in source) {
      if (source.hasOwnProperty(s)) {
        var value = source[s];
        if (typeof value === "object" && typeof target[s] === "object") {
          target[s] = _extends({}, target[s], value);
        } else {
          target[s] = value;
        }
      }
    }
  }
  return target;
}
var required$1 = function required(rule, value, source, errors, options, type4) {
  if (rule.required && (!source.hasOwnProperty(rule.field) || isEmptyValue(value, type4 || rule.type))) {
    errors.push(format(options.messages.required, rule.fullField));
  }
};
var whitespace = function whitespace2(rule, value, source, errors, options) {
  if (/^\s+$/.test(value) || value === "") {
    errors.push(format(options.messages.whitespace, rule.fullField));
  }
};
var urlReg;
var getUrlRegex = function() {
  if (urlReg) {
    return urlReg;
  }
  var word = "[a-fA-F\\d:]";
  var b = function b2(options) {
    return options && options.includeBoundaries ? "(?:(?<=\\s|^)(?=" + word + ")|(?<=" + word + ")(?=\\s|$))" : "";
  };
  var v4 = "(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]\\d|\\d)(?:\\.(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]\\d|\\d)){3}";
  var v6seg = "[a-fA-F\\d]{1,4}";
  var v6 = ("\n(?:\n(?:" + v6seg + ":){7}(?:" + v6seg + "|:)|                                    // 1:2:3:4:5:6:7::  1:2:3:4:5:6:7:8\n(?:" + v6seg + ":){6}(?:" + v4 + "|:" + v6seg + "|:)|                             // 1:2:3:4:5:6::    1:2:3:4:5:6::8   1:2:3:4:5:6::8  1:2:3:4:5:6::1.2.3.4\n(?:" + v6seg + ":){5}(?::" + v4 + "|(?::" + v6seg + "){1,2}|:)|                   // 1:2:3:4:5::      1:2:3:4:5::7:8   1:2:3:4:5::8    1:2:3:4:5::7:1.2.3.4\n(?:" + v6seg + ":){4}(?:(?::" + v6seg + "){0,1}:" + v4 + "|(?::" + v6seg + "){1,3}|:)| // 1:2:3:4::        1:2:3:4::6:7:8   1:2:3:4::8      1:2:3:4::6:7:1.2.3.4\n(?:" + v6seg + ":){3}(?:(?::" + v6seg + "){0,2}:" + v4 + "|(?::" + v6seg + "){1,4}|:)| // 1:2:3::          1:2:3::5:6:7:8   1:2:3::8        1:2:3::5:6:7:1.2.3.4\n(?:" + v6seg + ":){2}(?:(?::" + v6seg + "){0,3}:" + v4 + "|(?::" + v6seg + "){1,5}|:)| // 1:2::            1:2::4:5:6:7:8   1:2::8          1:2::4:5:6:7:1.2.3.4\n(?:" + v6seg + ":){1}(?:(?::" + v6seg + "){0,4}:" + v4 + "|(?::" + v6seg + "){1,6}|:)| // 1::              1::3:4:5:6:7:8   1::8            1::3:4:5:6:7:1.2.3.4\n(?::(?:(?::" + v6seg + "){0,5}:" + v4 + "|(?::" + v6seg + "){1,7}|:))             // ::2:3:4:5:6:7:8  ::2:3:4:5:6:7:8  ::8             ::1.2.3.4\n)(?:%[0-9a-zA-Z]{1,})?                                             // %eth0            %1\n").replace(/\s*\/\/.*$/gm, "").replace(/\n/g, "").trim();
  var v46Exact = new RegExp("(?:^" + v4 + "$)|(?:^" + v6 + "$)");
  var v4exact = new RegExp("^" + v4 + "$");
  var v6exact = new RegExp("^" + v6 + "$");
  var ip = function ip2(options) {
    return options && options.exact ? v46Exact : new RegExp("(?:" + b(options) + v4 + b(options) + ")|(?:" + b(options) + v6 + b(options) + ")", "g");
  };
  ip.v4 = function(options) {
    return options && options.exact ? v4exact : new RegExp("" + b(options) + v4 + b(options), "g");
  };
  ip.v6 = function(options) {
    return options && options.exact ? v6exact : new RegExp("" + b(options) + v6 + b(options), "g");
  };
  var protocol = "(?:(?:[a-z]+:)?//)";
  var auth = "(?:\\S+(?::\\S*)?@)?";
  var ipv4 = ip.v4().source;
  var ipv6 = ip.v6().source;
  var host = "(?:(?:[a-z\\u00a1-\\uffff0-9][-_]*)*[a-z\\u00a1-\\uffff0-9]+)";
  var domain = "(?:\\.(?:[a-z\\u00a1-\\uffff0-9]-*)*[a-z\\u00a1-\\uffff0-9]+)*";
  var tld = "(?:\\.(?:[a-z\\u00a1-\\uffff]{2,}))";
  var port = "(?::\\d{2,5})?";
  var path = '(?:[/?#][^\\s"]*)?';
  var regex = "(?:" + protocol + "|www\\.)" + auth + "(?:localhost|" + ipv4 + "|" + ipv6 + "|" + host + domain + tld + ")" + port + path;
  urlReg = new RegExp("(?:^" + regex + "$)", "i");
  return urlReg;
};
var pattern$2 = {
  // http://emailregex.com/
  email: /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]+\.)+[a-zA-Z\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]{2,}))$/,
  // url: new RegExp(
  //   '^(?!mailto:)(?:(?:http|https|ftp)://|//)(?:\\S+(?::\\S*)?@)?(?:(?:(?:[1-9]\\d?|1\\d\\d|2[01]\\d|22[0-3])(?:\\.(?:1?\\d{1,2}|2[0-4]\\d|25[0-5])){2}(?:\\.(?:[0-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-4]))|(?:(?:[a-z\\u00a1-\\uffff0-9]+-*)*[a-z\\u00a1-\\uffff0-9]+)(?:\\.(?:[a-z\\u00a1-\\uffff0-9]+-*)*[a-z\\u00a1-\\uffff0-9]+)*(?:\\.(?:[a-z\\u00a1-\\uffff]{2,})))|localhost)(?::\\d{2,5})?(?:(/|\\?|#)[^\\s]*)?$',
  //   'i',
  // ),
  hex: /^#?([a-f0-9]{6}|[a-f0-9]{3})$/i
};
var types = {
  integer: function integer(value) {
    return types.number(value) && parseInt(value, 10) === value;
  },
  "float": function float(value) {
    return types.number(value) && !types.integer(value);
  },
  array: function array(value) {
    return Array.isArray(value);
  },
  regexp: function regexp(value) {
    if (value instanceof RegExp) {
      return true;
    }
    try {
      return !!new RegExp(value);
    } catch (e) {
      return false;
    }
  },
  date: function date(value) {
    return typeof value.getTime === "function" && typeof value.getMonth === "function" && typeof value.getYear === "function" && !isNaN(value.getTime());
  },
  number: function number(value) {
    if (isNaN(value)) {
      return false;
    }
    return typeof value === "number";
  },
  object: function object(value) {
    return typeof value === "object" && !types.array(value);
  },
  method: function method(value) {
    return typeof value === "function";
  },
  email: function email(value) {
    return typeof value === "string" && value.length <= 320 && !!value.match(pattern$2.email);
  },
  url: function url(value) {
    return typeof value === "string" && value.length <= 2048 && !!value.match(getUrlRegex());
  },
  hex: function hex(value) {
    return typeof value === "string" && !!value.match(pattern$2.hex);
  }
};
var type$1 = function type(rule, value, source, errors, options) {
  if (rule.required && value === void 0) {
    required$1(rule, value, source, errors, options);
    return;
  }
  var custom = ["integer", "float", "array", "regexp", "object", "method", "email", "number", "date", "url", "hex"];
  var ruleType = rule.type;
  if (custom.indexOf(ruleType) > -1) {
    if (!types[ruleType](value)) {
      errors.push(format(options.messages.types[ruleType], rule.fullField, rule.type));
    }
  } else if (ruleType && typeof value !== rule.type) {
    errors.push(format(options.messages.types[ruleType], rule.fullField, rule.type));
  }
};
var range = function range2(rule, value, source, errors, options) {
  var len = typeof rule.len === "number";
  var min = typeof rule.min === "number";
  var max = typeof rule.max === "number";
  var spRegexp = /[\uD800-\uDBFF][\uDC00-\uDFFF]/g;
  var val = value;
  var key = null;
  var num = typeof value === "number";
  var str = typeof value === "string";
  var arr = Array.isArray(value);
  if (num) {
    key = "number";
  } else if (str) {
    key = "string";
  } else if (arr) {
    key = "array";
  }
  if (!key) {
    return false;
  }
  if (arr) {
    val = value.length;
  }
  if (str) {
    val = value.replace(spRegexp, "_").length;
  }
  if (len) {
    if (val !== rule.len) {
      errors.push(format(options.messages[key].len, rule.fullField, rule.len));
    }
  } else if (min && !max && val < rule.min) {
    errors.push(format(options.messages[key].min, rule.fullField, rule.min));
  } else if (max && !min && val > rule.max) {
    errors.push(format(options.messages[key].max, rule.fullField, rule.max));
  } else if (min && max && (val < rule.min || val > rule.max)) {
    errors.push(format(options.messages[key].range, rule.fullField, rule.min, rule.max));
  }
};
var ENUM$1 = "enum";
var enumerable$1 = function enumerable(rule, value, source, errors, options) {
  rule[ENUM$1] = Array.isArray(rule[ENUM$1]) ? rule[ENUM$1] : [];
  if (rule[ENUM$1].indexOf(value) === -1) {
    errors.push(format(options.messages[ENUM$1], rule.fullField, rule[ENUM$1].join(", ")));
  }
};
var pattern$1 = function pattern(rule, value, source, errors, options) {
  if (rule.pattern) {
    if (rule.pattern instanceof RegExp) {
      rule.pattern.lastIndex = 0;
      if (!rule.pattern.test(value)) {
        errors.push(format(options.messages.pattern.mismatch, rule.fullField, value, rule.pattern));
      }
    } else if (typeof rule.pattern === "string") {
      var _pattern = new RegExp(rule.pattern);
      if (!_pattern.test(value)) {
        errors.push(format(options.messages.pattern.mismatch, rule.fullField, value, rule.pattern));
      }
    }
  }
};
var rules = {
  required: required$1,
  whitespace,
  type: type$1,
  range,
  "enum": enumerable$1,
  pattern: pattern$1
};
var string = function string2(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value, "string") && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options, "string");
    if (!isEmptyValue(value, "string")) {
      rules.type(rule, value, source, errors, options);
      rules.range(rule, value, source, errors, options);
      rules.pattern(rule, value, source, errors, options);
      if (rule.whitespace === true) {
        rules.whitespace(rule, value, source, errors, options);
      }
    }
  }
  callback(errors);
};
var method2 = function method3(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
    if (value !== void 0) {
      rules.type(rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var number2 = function number3(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (value === "") {
      value = void 0;
    }
    if (isEmptyValue(value) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
    if (value !== void 0) {
      rules.type(rule, value, source, errors, options);
      rules.range(rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var _boolean = function _boolean2(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
    if (value !== void 0) {
      rules.type(rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var regexp2 = function regexp3(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
    if (!isEmptyValue(value)) {
      rules.type(rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var integer2 = function integer3(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
    if (value !== void 0) {
      rules.type(rule, value, source, errors, options);
      rules.range(rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var floatFn = function floatFn2(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
    if (value !== void 0) {
      rules.type(rule, value, source, errors, options);
      rules.range(rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var array2 = function array3(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if ((value === void 0 || value === null) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options, "array");
    if (value !== void 0 && value !== null) {
      rules.type(rule, value, source, errors, options);
      rules.range(rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var object2 = function object3(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
    if (value !== void 0) {
      rules.type(rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var ENUM = "enum";
var enumerable2 = function enumerable3(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
    if (value !== void 0) {
      rules[ENUM](rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var pattern2 = function pattern3(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value, "string") && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
    if (!isEmptyValue(value, "string")) {
      rules.pattern(rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var date2 = function date3(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value, "date") && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
    if (!isEmptyValue(value, "date")) {
      var dateObject;
      if (value instanceof Date) {
        dateObject = value;
      } else {
        dateObject = new Date(value);
      }
      rules.type(rule, dateObject, source, errors, options);
      if (dateObject) {
        rules.range(rule, dateObject.getTime(), source, errors, options);
      }
    }
  }
  callback(errors);
};
var required2 = function required3(rule, value, callback, source, options) {
  var errors = [];
  var type4 = Array.isArray(value) ? "array" : typeof value;
  rules.required(rule, value, source, errors, options, type4);
  callback(errors);
};
var type2 = function type3(rule, value, callback, source, options) {
  var ruleType = rule.type;
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value, ruleType) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options, ruleType);
    if (!isEmptyValue(value, ruleType)) {
      rules.type(rule, value, source, errors, options);
    }
  }
  callback(errors);
};
var any = function any2(rule, value, callback, source, options) {
  var errors = [];
  var validate = rule.required || !rule.required && source.hasOwnProperty(rule.field);
  if (validate) {
    if (isEmptyValue(value) && !rule.required) {
      return callback();
    }
    rules.required(rule, value, source, errors, options);
  }
  callback(errors);
};
var validators = {
  string,
  method: method2,
  number: number2,
  "boolean": _boolean,
  regexp: regexp2,
  integer: integer2,
  "float": floatFn,
  array: array2,
  object: object2,
  "enum": enumerable2,
  pattern: pattern2,
  date: date2,
  url: type2,
  hex: type2,
  email: type2,
  required: required2,
  any
};
function newMessages() {
  return {
    "default": "Validation error on field %s",
    required: "%s is required",
    "enum": "%s must be one of %s",
    whitespace: "%s cannot be empty",
    date: {
      format: "%s date %s is invalid for format %s",
      parse: "%s date could not be parsed, %s is invalid ",
      invalid: "%s date %s is invalid"
    },
    types: {
      string: "%s is not a %s",
      method: "%s is not a %s (function)",
      array: "%s is not an %s",
      object: "%s is not an %s",
      number: "%s is not a %s",
      date: "%s is not a %s",
      "boolean": "%s is not a %s",
      integer: "%s is not an %s",
      "float": "%s is not a %s",
      regexp: "%s is not a valid %s",
      email: "%s is not a valid %s",
      url: "%s is not a valid %s",
      hex: "%s is not a valid %s"
    },
    string: {
      len: "%s must be exactly %s characters",
      min: "%s must be at least %s characters",
      max: "%s cannot be longer than %s characters",
      range: "%s must be between %s and %s characters"
    },
    number: {
      len: "%s must equal %s",
      min: "%s cannot be less than %s",
      max: "%s cannot be greater than %s",
      range: "%s must be between %s and %s"
    },
    array: {
      len: "%s must be exactly %s in length",
      min: "%s cannot be less than %s in length",
      max: "%s cannot be greater than %s in length",
      range: "%s must be between %s and %s in length"
    },
    pattern: {
      mismatch: "%s value %s does not match pattern %s"
    },
    clone: function clone() {
      var cloned = JSON.parse(JSON.stringify(this));
      cloned.clone = this.clone;
      return cloned;
    }
  };
}
var messages = newMessages();
var Schema = /* @__PURE__ */ function() {
  function Schema2(descriptor) {
    this.rules = null;
    this._messages = messages;
    this.define(descriptor);
  }
  var _proto = Schema2.prototype;
  _proto.define = function define(rules2) {
    var _this = this;
    if (!rules2) {
      throw new Error("Cannot configure a schema with no rules");
    }
    if (typeof rules2 !== "object" || Array.isArray(rules2)) {
      throw new Error("Rules must be an object");
    }
    this.rules = {};
    Object.keys(rules2).forEach(function(name) {
      var item = rules2[name];
      _this.rules[name] = Array.isArray(item) ? item : [item];
    });
  };
  _proto.messages = function messages2(_messages) {
    if (_messages) {
      this._messages = deepMerge(newMessages(), _messages);
    }
    return this._messages;
  };
  _proto.validate = function validate(source_, o, oc) {
    var _this2 = this;
    if (o === void 0) {
      o = {};
    }
    if (oc === void 0) {
      oc = function oc2() {
      };
    }
    var source = source_;
    var options = o;
    var callback = oc;
    if (typeof options === "function") {
      callback = options;
      options = {};
    }
    if (!this.rules || Object.keys(this.rules).length === 0) {
      if (callback) {
        callback(null, source);
      }
      return Promise.resolve(source);
    }
    function complete(results) {
      var errors = [];
      var fields = {};
      function add(e) {
        if (Array.isArray(e)) {
          var _errors;
          errors = (_errors = errors).concat.apply(_errors, e);
        } else {
          errors.push(e);
        }
      }
      for (var i = 0; i < results.length; i++) {
        add(results[i]);
      }
      if (!errors.length) {
        callback(null, source);
      } else {
        fields = convertFieldsError(errors);
        callback(errors, fields);
      }
    }
    if (options.messages) {
      var messages$1 = this.messages();
      if (messages$1 === messages) {
        messages$1 = newMessages();
      }
      deepMerge(messages$1, options.messages);
      options.messages = messages$1;
    } else {
      options.messages = this.messages();
    }
    var series = {};
    var keys = options.keys || Object.keys(this.rules);
    keys.forEach(function(z) {
      var arr = _this2.rules[z];
      var value = source[z];
      arr.forEach(function(r) {
        var rule = r;
        if (typeof rule.transform === "function") {
          if (source === source_) {
            source = _extends({}, source);
          }
          value = source[z] = rule.transform(value);
        }
        if (typeof rule === "function") {
          rule = {
            validator: rule
          };
        } else {
          rule = _extends({}, rule);
        }
        rule.validator = _this2.getValidationMethod(rule);
        if (!rule.validator) {
          return;
        }
        rule.field = z;
        rule.fullField = rule.fullField || z;
        rule.type = _this2.getType(rule);
        series[z] = series[z] || [];
        series[z].push({
          rule,
          value,
          source,
          field: z
        });
      });
    });
    var errorFields = {};
    return asyncMap(series, options, function(data, doIt) {
      var rule = data.rule;
      var deep = (rule.type === "object" || rule.type === "array") && (typeof rule.fields === "object" || typeof rule.defaultField === "object");
      deep = deep && (rule.required || !rule.required && data.value);
      rule.field = data.field;
      function addFullField(key, schema) {
        return _extends({}, schema, {
          fullField: rule.fullField + "." + key,
          fullFields: rule.fullFields ? [].concat(rule.fullFields, [key]) : [key]
        });
      }
      function cb(e) {
        if (e === void 0) {
          e = [];
        }
        var errorList = Array.isArray(e) ? e : [e];
        if (!options.suppressWarning && errorList.length) {
          Schema2.warning("async-validator:", errorList);
        }
        if (errorList.length && rule.message !== void 0) {
          errorList = [].concat(rule.message);
        }
        var filledErrors = errorList.map(complementError(rule, source));
        if (options.first && filledErrors.length) {
          errorFields[rule.field] = 1;
          return doIt(filledErrors);
        }
        if (!deep) {
          doIt(filledErrors);
        } else {
          if (rule.required && !data.value) {
            if (rule.message !== void 0) {
              filledErrors = [].concat(rule.message).map(complementError(rule, source));
            } else if (options.error) {
              filledErrors = [options.error(rule, format(options.messages.required, rule.field))];
            }
            return doIt(filledErrors);
          }
          var fieldsSchema = {};
          if (rule.defaultField) {
            Object.keys(data.value).map(function(key) {
              fieldsSchema[key] = rule.defaultField;
            });
          }
          fieldsSchema = _extends({}, fieldsSchema, data.rule.fields);
          var paredFieldsSchema = {};
          Object.keys(fieldsSchema).forEach(function(field) {
            var fieldSchema = fieldsSchema[field];
            var fieldSchemaList = Array.isArray(fieldSchema) ? fieldSchema : [fieldSchema];
            paredFieldsSchema[field] = fieldSchemaList.map(addFullField.bind(null, field));
          });
          var schema = new Schema2(paredFieldsSchema);
          schema.messages(options.messages);
          if (data.rule.options) {
            data.rule.options.messages = options.messages;
            data.rule.options.error = options.error;
          }
          schema.validate(data.value, data.rule.options || options, function(errs) {
            var finalErrors = [];
            if (filledErrors && filledErrors.length) {
              finalErrors.push.apply(finalErrors, filledErrors);
            }
            if (errs && errs.length) {
              finalErrors.push.apply(finalErrors, errs);
            }
            doIt(finalErrors.length ? finalErrors : null);
          });
        }
      }
      var res;
      if (rule.asyncValidator) {
        res = rule.asyncValidator(rule, data.value, cb, data.source, options);
      } else if (rule.validator) {
        try {
          res = rule.validator(rule, data.value, cb, data.source, options);
        } catch (error) {
          console.error == null ? void 0 : console.error(error);
          if (!options.suppressValidatorError) {
            setTimeout(function() {
              throw error;
            }, 0);
          }
          cb(error.message);
        }
        if (res === true) {
          cb();
        } else if (res === false) {
          cb(typeof rule.message === "function" ? rule.message(rule.fullField || rule.field) : rule.message || (rule.fullField || rule.field) + " fails");
        } else if (res instanceof Array) {
          cb(res);
        } else if (res instanceof Error) {
          cb(res.message);
        }
      }
      if (res && res.then) {
        res.then(function() {
          return cb();
        }, function(e) {
          return cb(e);
        });
      }
    }, function(results) {
      complete(results);
    }, source);
  };
  _proto.getType = function getType(rule) {
    if (rule.type === void 0 && rule.pattern instanceof RegExp) {
      rule.type = "pattern";
    }
    if (typeof rule.validator !== "function" && rule.type && !validators.hasOwnProperty(rule.type)) {
      throw new Error(format("Unknown rule type %s", rule.type));
    }
    return rule.type || "string";
  };
  _proto.getValidationMethod = function getValidationMethod(rule) {
    if (typeof rule.validator === "function") {
      return rule.validator;
    }
    var keys = Object.keys(rule);
    var messageIndex = keys.indexOf("message");
    if (messageIndex !== -1) {
      keys.splice(messageIndex, 1);
    }
    if (keys.length === 1 && keys[0] === "required") {
      return validators.required;
    }
    return validators[this.getType(rule)] || void 0;
  };
  return Schema2;
}();
Schema.register = function register(type4, validator) {
  if (typeof validator !== "function") {
    throw new Error("Cannot register a validator by type, validator is not a function");
  }
  validators[type4] = validator;
};
Schema.warning = warning;
Schema.messages = messages;
Schema.validators = validators;
function formItemSize(props) {
  const NForm2 = inject(formInjectionKey, null);
  return {
    mergedSize: computed(() => {
      if (props.size !== void 0)
        return props.size;
      if ((NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.props.size) !== void 0)
        return NForm2.props.size;
      return "medium";
    })
  };
}
function formItemMisc(props) {
  const NForm2 = inject(formInjectionKey, null);
  const mergedLabelPlacementRef = computed(() => {
    const { labelPlacement } = props;
    if (labelPlacement !== void 0)
      return labelPlacement;
    if (NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.props.labelPlacement)
      return NForm2.props.labelPlacement;
    return "top";
  });
  const isAutoLabelWidthRef = computed(() => {
    return mergedLabelPlacementRef.value === "left" && (props.labelWidth === "auto" || (NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.props.labelWidth) === "auto");
  });
  const mergedLabelWidthRef = computed(() => {
    if (mergedLabelPlacementRef.value === "top")
      return;
    const { labelWidth } = props;
    if (labelWidth !== void 0 && labelWidth !== "auto") {
      return formatLength(labelWidth);
    }
    if (isAutoLabelWidthRef.value) {
      const autoComputedWidth = NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.maxChildLabelWidthRef.value;
      if (autoComputedWidth !== void 0) {
        return formatLength(autoComputedWidth);
      } else {
        return void 0;
      }
    }
    if ((NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.props.labelWidth) !== void 0) {
      return formatLength(NForm2.props.labelWidth);
    }
    return void 0;
  });
  const mergedLabelAlignRef = computed(() => {
    const { labelAlign } = props;
    if (labelAlign)
      return labelAlign;
    if (NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.props.labelAlign)
      return NForm2.props.labelAlign;
    return void 0;
  });
  const mergedLabelStyleRef = computed(() => {
    var _a;
    return [
      (_a = props.labelProps) === null || _a === void 0 ? void 0 : _a.style,
      props.labelStyle,
      {
        width: mergedLabelWidthRef.value
      }
    ];
  });
  const mergedShowRequireMarkRef = computed(() => {
    const { showRequireMark } = props;
    if (showRequireMark !== void 0)
      return showRequireMark;
    return NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.props.showRequireMark;
  });
  const mergedRequireMarkPlacementRef = computed(() => {
    const { requireMarkPlacement } = props;
    if (requireMarkPlacement !== void 0)
      return requireMarkPlacement;
    return (NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.props.requireMarkPlacement) || "right";
  });
  const validationErroredRef = ref(false);
  const mergedValidationStatusRef = computed(() => {
    const { validationStatus } = props;
    if (validationStatus !== void 0)
      return validationStatus;
    if (validationErroredRef.value)
      return "error";
    return void 0;
  });
  const mergedShowFeedbackRef = computed(() => {
    const { showFeedback } = props;
    if (showFeedback !== void 0)
      return showFeedback;
    if ((NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.props.showFeedback) !== void 0)
      return NForm2.props.showFeedback;
    return true;
  });
  const mergedShowLabelRef = computed(() => {
    const { showLabel } = props;
    if (showLabel !== void 0)
      return showLabel;
    if ((NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.props.showLabel) !== void 0)
      return NForm2.props.showLabel;
    return true;
  });
  return {
    validationErrored: validationErroredRef,
    mergedLabelStyle: mergedLabelStyleRef,
    mergedLabelPlacement: mergedLabelPlacementRef,
    mergedLabelAlign: mergedLabelAlignRef,
    mergedShowRequireMark: mergedShowRequireMarkRef,
    mergedRequireMarkPlacement: mergedRequireMarkPlacementRef,
    mergedValidationStatus: mergedValidationStatusRef,
    mergedShowFeedback: mergedShowFeedbackRef,
    mergedShowLabel: mergedShowLabelRef,
    isAutoLabelWidth: isAutoLabelWidthRef
  };
}
function formItemRule(props) {
  const NForm2 = inject(formInjectionKey, null);
  const compatibleRulePathRef = computed(() => {
    const { rulePath } = props;
    if (rulePath !== void 0)
      return rulePath;
    const { path } = props;
    if (path !== void 0)
      return path;
    return void 0;
  });
  const mergedRulesRef = computed(() => {
    const rules2 = [];
    const { rule } = props;
    if (rule !== void 0) {
      if (Array.isArray(rule))
        rules2.push(...rule);
      else
        rules2.push(rule);
    }
    if (NForm2) {
      const { rules: formRules } = NForm2.props;
      const { value: rulePath } = compatibleRulePathRef;
      if (formRules !== void 0 && rulePath !== void 0) {
        const formRule = get(formRules, rulePath);
        if (formRule !== void 0) {
          if (Array.isArray(formRule)) {
            rules2.push(...formRule);
          } else {
            rules2.push(formRule);
          }
        }
      }
    }
    return rules2;
  });
  const hasRequiredRuleRef = computed(() => {
    return mergedRulesRef.value.some((rule) => rule.required);
  });
  const mergedRequiredRef = computed(() => {
    return hasRequiredRuleRef.value || props.required;
  });
  return {
    mergedRules: mergedRulesRef,
    mergedRequired: mergedRequiredRef
  };
}
const {
  cubicBezierEaseInOut
} = commonVariables;
function fadeDownTransition({
  name = "fade-down",
  fromOffset = "-4px",
  enterDuration = ".3s",
  leaveDuration = ".3s",
  enterCubicBezier = cubicBezierEaseInOut,
  leaveCubicBezier = cubicBezierEaseInOut
} = {}) {
  return [c(`&.${name}-transition-enter-from, &.${name}-transition-leave-to`, {
    opacity: 0,
    transform: `translateY(${fromOffset})`
  }), c(`&.${name}-transition-enter-to, &.${name}-transition-leave-from`, {
    opacity: 1,
    transform: "translateY(0)"
  }), c(`&.${name}-transition-leave-active`, {
    transition: `opacity ${leaveDuration} ${leaveCubicBezier}, transform ${leaveDuration} ${leaveCubicBezier}`
  }), c(`&.${name}-transition-enter-active`, {
    transition: `opacity ${enterDuration} ${enterCubicBezier}, transform ${enterDuration} ${enterCubicBezier}`
  })];
}
const style = cB("form-item", `
 display: grid;
 line-height: var(--n-line-height);
`, [cB("form-item-label", `
 grid-area: label;
 align-items: center;
 line-height: 1.25;
 text-align: var(--n-label-text-align);
 font-size: var(--n-label-font-size);
 min-height: var(--n-label-height);
 padding: var(--n-label-padding);
 color: var(--n-label-text-color);
 transition: color .3s var(--n-bezier);
 box-sizing: border-box;
 font-weight: var(--n-label-font-weight);
 `, [cE("asterisk", `
 white-space: nowrap;
 user-select: none;
 -webkit-user-select: none;
 color: var(--n-asterisk-color);
 transition: color .3s var(--n-bezier);
 `), cE("asterisk-placeholder", `
 grid-area: mark;
 user-select: none;
 -webkit-user-select: none;
 visibility: hidden; 
 `)]), cB("form-item-blank", `
 grid-area: blank;
 min-height: var(--n-blank-height);
 `), cM("auto-label-width", [cB("form-item-label", "white-space: nowrap;")]), cM("left-labelled", `
 grid-template-areas:
 "label blank"
 "label feedback";
 grid-template-columns: auto minmax(0, 1fr);
 grid-template-rows: auto 1fr;
 align-items: start;
 `, [cB("form-item-label", `
 display: grid;
 grid-template-columns: 1fr auto;
 min-height: var(--n-blank-height);
 height: auto;
 box-sizing: border-box;
 flex-shrink: 0;
 flex-grow: 0;
 `, [cM("reverse-columns-space", `
 grid-template-columns: auto 1fr;
 `), cM("left-mark", `
 grid-template-areas:
 "mark text"
 ". text";
 `), cM("right-mark", `
 grid-template-areas: 
 "text mark"
 "text .";
 `), cM("right-hanging-mark", `
 grid-template-areas: 
 "text mark"
 "text .";
 `), cE("text", `
 grid-area: text; 
 `), cE("asterisk", `
 grid-area: mark; 
 align-self: end;
 `)])]), cM("top-labelled", `
 grid-template-areas:
 "label"
 "blank"
 "feedback";
 grid-template-rows: minmax(var(--n-label-height), auto) 1fr;
 grid-template-columns: minmax(0, 100%);
 `, [cM("no-label", `
 grid-template-areas:
 "blank"
 "feedback";
 grid-template-rows: 1fr;
 `), cB("form-item-label", `
 display: flex;
 align-items: flex-start;
 justify-content: var(--n-label-text-align);
 `)]), cB("form-item-blank", `
 box-sizing: border-box;
 display: flex;
 align-items: center;
 position: relative;
 `), cB("form-item-feedback-wrapper", `
 grid-area: feedback;
 box-sizing: border-box;
 min-height: var(--n-feedback-height);
 font-size: var(--n-feedback-font-size);
 line-height: 1.25;
 transform-origin: top left;
 `, [c("&:not(:empty)", `
 padding: var(--n-feedback-padding);
 `), cB("form-item-feedback", {
  transition: "color .3s var(--n-bezier)",
  color: "var(--n-feedback-text-color)"
}, [cM("warning", {
  color: "var(--n-feedback-text-color-warning)"
}), cM("error", {
  color: "var(--n-feedback-text-color-error)"
}), fadeDownTransition({
  fromOffset: "-3px",
  enterDuration: ".3s",
  leaveDuration: ".2s"
})])])]);
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
const formItemProps = Object.assign(Object.assign({}, useTheme.props), { label: String, labelWidth: [Number, String], labelStyle: [String, Object], labelAlign: String, labelPlacement: String, path: String, first: Boolean, rulePath: String, required: Boolean, showRequireMark: {
  type: Boolean,
  default: void 0
}, requireMarkPlacement: String, showFeedback: {
  type: Boolean,
  default: void 0
}, rule: [Object, Array], size: String, ignorePathChange: Boolean, validationStatus: String, feedback: String, showLabel: {
  type: Boolean,
  default: void 0
}, labelProps: Object });
function wrapValidator(validator, async) {
  return (...args) => {
    try {
      const validateResult = validator(...args);
      if (!async && (typeof validateResult === "boolean" || validateResult instanceof Error || Array.isArray(validateResult)) || // Error[]
      (validateResult === null || validateResult === void 0 ? void 0 : validateResult.then)) {
        return validateResult;
      } else if (validateResult === void 0) {
        return true;
      } else {
        warn("form-item/validate", `You return a ${typeof validateResult} typed value in the validator method, which is not recommended. Please use ` + (async ? "`Promise`" : "`boolean`, `Error` or `Promise`") + " typed value instead.");
        return true;
      }
    } catch (err) {
      warn("form-item/validate", "An error is catched in the validation, so the validation won't be done. Your callback in `validate` method of `n-form` or `n-form-item` won't be called in this validation.");
      console.error(err);
      return void 0;
    }
  };
}
const NFormItem = defineComponent({
  name: "FormItem",
  props: formItemProps,
  setup(props) {
    useInjectionInstanceCollection(formItemInstsInjectionKey, "formItems", toRef(props, "path"));
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const NForm2 = inject(formInjectionKey, null);
    const formItemSizeRefs = formItemSize(props);
    const formItemMiscRefs = formItemMisc(props);
    const { validationErrored: validationErroredRef } = formItemMiscRefs;
    const { mergedRequired: mergedRequiredRef, mergedRules: mergedRulesRef } = formItemRule(props);
    const { mergedSize: mergedSizeRef } = formItemSizeRefs;
    const { mergedLabelPlacement: labelPlacementRef, mergedLabelAlign: labelTextAlignRef, mergedRequireMarkPlacement: mergedRequireMarkPlacementRef } = formItemMiscRefs;
    const renderExplainsRef = ref([]);
    const feedbackIdRef = ref(createId());
    const mergedDisabledRef = NForm2 ? toRef(NForm2.props, "disabled") : ref(false);
    const themeRef = useTheme("Form", "-form-item", style, formLight, props, mergedClsPrefixRef);
    watch(toRef(props, "path"), () => {
      if (props.ignorePathChange)
        return;
      restoreValidation();
    });
    function restoreValidation() {
      renderExplainsRef.value = [];
      validationErroredRef.value = false;
      if (props.feedback) {
        feedbackIdRef.value = createId();
      }
    }
    function handleContentBlur() {
      void internalValidate("blur");
    }
    function handleContentChange() {
      void internalValidate("change");
    }
    function handleContentFocus() {
      void internalValidate("focus");
    }
    function handleContentInput() {
      void internalValidate("input");
    }
    function validate(options, callback) {
      return __awaiter(this, void 0, void 0, function* () {
        let trigger;
        let validateCallback;
        let shouldRuleBeApplied;
        let asyncValidatorOptions;
        if (typeof options === "string") {
          trigger = options;
          validateCallback = callback;
        } else if (options !== null && typeof options === "object") {
          trigger = options.trigger;
          validateCallback = options.callback;
          shouldRuleBeApplied = options.shouldRuleBeApplied;
          asyncValidatorOptions = options.options;
        }
        yield new Promise((resolve, reject) => {
          void internalValidate(trigger, shouldRuleBeApplied, asyncValidatorOptions).then(({ valid, errors }) => {
            if (valid) {
              if (validateCallback) {
                validateCallback();
              }
              resolve();
            } else {
              if (validateCallback) {
                validateCallback(errors);
              }
              reject(errors);
            }
          });
        });
      });
    }
    const internalValidate = (trigger = null, shouldRuleBeApplied = () => true, options = {
      suppressWarning: true
    }) => __awaiter(this, void 0, void 0, function* () {
      const { path } = props;
      if (!options) {
        options = {};
      } else {
        if (!options.first)
          options.first = props.first;
      }
      const { value: rules2 } = mergedRulesRef;
      const value = NForm2 ? get(NForm2.props.model, path || "") : void 0;
      const messageRenderers = {};
      const originalMessageRendersMessage = {};
      const activeRules = (!trigger ? rules2 : rules2.filter((rule) => {
        if (Array.isArray(rule.trigger)) {
          return rule.trigger.includes(trigger);
        } else {
          return rule.trigger === trigger;
        }
      })).filter(shouldRuleBeApplied).map((rule, i) => {
        const shallowClonedRule = Object.assign({}, rule);
        if (shallowClonedRule.validator) {
          shallowClonedRule.validator = wrapValidator(shallowClonedRule.validator, false);
        }
        if (shallowClonedRule.asyncValidator) {
          shallowClonedRule.asyncValidator = wrapValidator(shallowClonedRule.asyncValidator, true);
        }
        if (shallowClonedRule.renderMessage) {
          const rendererKey = `__renderMessage__${i}`;
          originalMessageRendersMessage[rendererKey] = shallowClonedRule.message;
          shallowClonedRule.message = rendererKey;
          messageRenderers[rendererKey] = shallowClonedRule.renderMessage;
        }
        return shallowClonedRule;
      });
      if (!activeRules.length) {
        return {
          valid: true
        };
      }
      const mergedPath = path !== null && path !== void 0 ? path : "__n_no_path__";
      const validator = new Schema({ [mergedPath]: activeRules });
      const { validateMessages } = (NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.props) || {};
      if (validateMessages) {
        validator.messages(validateMessages);
      }
      return yield new Promise((resolve) => {
        void validator.validate({ [mergedPath]: value }, options, (errors) => {
          if (errors === null || errors === void 0 ? void 0 : errors.length) {
            renderExplainsRef.value = errors.map((error) => {
              const transformedMessage = (error === null || error === void 0 ? void 0 : error.message) || "";
              return {
                key: transformedMessage,
                render: () => {
                  if (transformedMessage.startsWith("__renderMessage__")) {
                    return messageRenderers[transformedMessage]();
                  }
                  return transformedMessage;
                }
              };
            });
            errors.forEach((error) => {
              var _a;
              if ((_a = error.message) === null || _a === void 0 ? void 0 : _a.startsWith("__renderMessage__")) {
                error.message = originalMessageRendersMessage[error.message];
              }
            });
            validationErroredRef.value = true;
            resolve({
              valid: false,
              errors
            });
          } else {
            restoreValidation();
            resolve({
              valid: true
            });
          }
        });
      });
    });
    provide(formItemInjectionKey, {
      path: toRef(props, "path"),
      disabled: mergedDisabledRef,
      mergedSize: formItemSizeRefs.mergedSize,
      mergedValidationStatus: formItemMiscRefs.mergedValidationStatus,
      restoreValidation,
      handleContentBlur,
      handleContentChange,
      handleContentFocus,
      handleContentInput
    });
    const exposedRef = {
      validate,
      restoreValidation,
      internalValidate
    };
    const labelElementRef = ref(null);
    onMounted(() => {
      if (!formItemMiscRefs.isAutoLabelWidth.value)
        return;
      const labelElement = labelElementRef.value;
      if (labelElement !== null) {
        const memoizedWhitespace = labelElement.style.whiteSpace;
        labelElement.style.whiteSpace = "nowrap";
        labelElement.style.width = "";
        NForm2 === null || NForm2 === void 0 ? void 0 : NForm2.deriveMaxChildLabelWidth(Number(getComputedStyle(labelElement).width.slice(0, -2)));
        labelElement.style.whiteSpace = memoizedWhitespace;
      }
    });
    const cssVarsRef = computed(() => {
      var _a;
      const { value: size } = mergedSizeRef;
      const { value: labelPlacement } = labelPlacementRef;
      const direction = labelPlacement === "top" ? "vertical" : "horizontal";
      const { common: { cubicBezierEaseInOut: cubicBezierEaseInOut2 }, self: { labelTextColor, asteriskColor, lineHeight, feedbackTextColor, feedbackTextColorWarning, feedbackTextColorError, feedbackPadding, labelFontWeight, [createKey("labelHeight", size)]: labelHeight, [createKey("blankHeight", size)]: blankHeight, [createKey("feedbackFontSize", size)]: feedbackFontSize, [createKey("feedbackHeight", size)]: feedbackHeight, [createKey("labelPadding", direction)]: labelPadding, [createKey("labelTextAlign", direction)]: labelTextAlign, [createKey(createKey("labelFontSize", labelPlacement), size)]: labelFontSize } } = themeRef.value;
      let mergedLabelTextAlign = (_a = labelTextAlignRef.value) !== null && _a !== void 0 ? _a : labelTextAlign;
      if (labelPlacement === "top") {
        mergedLabelTextAlign = mergedLabelTextAlign === "right" ? "flex-end" : "flex-start";
      }
      const cssVars = {
        "--n-bezier": cubicBezierEaseInOut2,
        "--n-line-height": lineHeight,
        "--n-blank-height": blankHeight,
        "--n-label-font-size": labelFontSize,
        "--n-label-text-align": mergedLabelTextAlign,
        "--n-label-height": labelHeight,
        "--n-label-padding": labelPadding,
        "--n-label-font-weight": labelFontWeight,
        "--n-asterisk-color": asteriskColor,
        "--n-label-text-color": labelTextColor,
        "--n-feedback-padding": feedbackPadding,
        "--n-feedback-font-size": feedbackFontSize,
        "--n-feedback-height": feedbackHeight,
        "--n-feedback-text-color": feedbackTextColor,
        "--n-feedback-text-color-warning": feedbackTextColorWarning,
        "--n-feedback-text-color-error": feedbackTextColorError
      };
      return cssVars;
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("form-item", computed(() => {
      var _a;
      return `${mergedSizeRef.value[0]}${labelPlacementRef.value[0]}${((_a = labelTextAlignRef.value) === null || _a === void 0 ? void 0 : _a[0]) || ""}`;
    }), cssVarsRef, props) : void 0;
    const reverseColSpaceRef = computed(() => {
      return labelPlacementRef.value === "left" && mergedRequireMarkPlacementRef.value === "left" && labelTextAlignRef.value === "left";
    });
    return Object.assign(Object.assign(Object.assign(Object.assign({ labelElementRef, mergedClsPrefix: mergedClsPrefixRef, mergedRequired: mergedRequiredRef, feedbackId: feedbackIdRef, renderExplains: renderExplainsRef, reverseColSpace: reverseColSpaceRef }, formItemMiscRefs), formItemSizeRefs), exposedRef), { cssVars: inlineThemeDisabled ? void 0 : cssVarsRef, themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass, onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender });
  },
  render() {
    const { $slots, mergedClsPrefix, mergedShowLabel, mergedShowRequireMark, mergedRequireMarkPlacement, onRender } = this;
    const renderedShowRequireMark = mergedShowRequireMark !== void 0 ? mergedShowRequireMark : this.mergedRequired;
    onRender === null || onRender === void 0 ? void 0 : onRender();
    const renderLabel = () => {
      const labelText = this.$slots.label ? this.$slots.label() : this.label;
      if (!labelText)
        return null;
      const textNode = h("span", { class: `${mergedClsPrefix}-form-item-label__text` }, labelText);
      const markNode = renderedShowRequireMark ? h("span", { class: `${mergedClsPrefix}-form-item-label__asterisk` }, mergedRequireMarkPlacement !== "left" ? " *" : "* ") : mergedRequireMarkPlacement === "right-hanging" && h("span", { class: `${mergedClsPrefix}-form-item-label__asterisk-placeholder` }, " *");
      const { labelProps } = this;
      return h("label", Object.assign({}, labelProps, { class: [
        labelProps === null || labelProps === void 0 ? void 0 : labelProps.class,
        `${mergedClsPrefix}-form-item-label`,
        `${mergedClsPrefix}-form-item-label--${mergedRequireMarkPlacement}-mark`,
        this.reverseColSpace && `${mergedClsPrefix}-form-item-label--reverse-columns-space`
      ], style: this.mergedLabelStyle, ref: "labelElementRef" }), mergedRequireMarkPlacement === "left" ? [markNode, textNode] : [textNode, markNode]);
    };
    return h(
      "div",
      { class: [
        `${mergedClsPrefix}-form-item`,
        this.themeClass,
        `${mergedClsPrefix}-form-item--${this.mergedSize}-size`,
        `${mergedClsPrefix}-form-item--${this.mergedLabelPlacement}-labelled`,
        this.isAutoLabelWidth && `${mergedClsPrefix}-form-item--auto-label-width`,
        !mergedShowLabel && `${mergedClsPrefix}-form-item--no-label`
      ], style: this.cssVars },
      mergedShowLabel && renderLabel(),
      h("div", { class: [
        `${mergedClsPrefix}-form-item-blank`,
        this.mergedValidationStatus && `${mergedClsPrefix}-form-item-blank--${this.mergedValidationStatus}`
      ] }, $slots),
      this.mergedShowFeedback ? h(
        "div",
        { key: this.feedbackId, class: `${mergedClsPrefix}-form-item-feedback-wrapper` },
        h(Transition, { name: "fade-down-transition", mode: "out-in" }, {
          default: () => {
            const { mergedValidationStatus } = this;
            return resolveWrappedSlot($slots.feedback, (children) => {
              var _a;
              const { feedback } = this;
              const feedbackNodes = children || feedback ? h("div", { key: "__feedback__", class: `${mergedClsPrefix}-form-item-feedback__line` }, children || feedback) : this.renderExplains.length ? (_a = this.renderExplains) === null || _a === void 0 ? void 0 : _a.map(({ key, render }) => h("div", { key, class: `${mergedClsPrefix}-form-item-feedback__line` }, render())) : null;
              return feedbackNodes ? mergedValidationStatus === "warning" ? h("div", { key: "controlled-warning", class: `${mergedClsPrefix}-form-item-feedback ${mergedClsPrefix}-form-item-feedback--warning` }, feedbackNodes) : mergedValidationStatus === "error" ? h("div", { key: "controlled-error", class: `${mergedClsPrefix}-form-item-feedback ${mergedClsPrefix}-form-item-feedback--error` }, feedbackNodes) : mergedValidationStatus === "success" ? h("div", { key: "controlled-success", class: `${mergedClsPrefix}-form-item-feedback ${mergedClsPrefix}-form-item-feedback--success` }, feedbackNodes) : h("div", { key: "controlled-default", class: `${mergedClsPrefix}-form-item-feedback` }, feedbackNodes) : null;
            });
          }
        })
      ) : null
    );
  }
});
const _hoisted_1$7 = { class: "flex-container" };
const _hoisted_2$7 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_3$6 = { class: "flex-container" };
const _hoisted_4$5 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_5$5 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_6$5 = { class: "flex-container" };
const _hoisted_7$5 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Seed", -1);
const _hoisted_8$4 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "For random seed use -1.", -1);
const _hoisted_9$4 = { class: "flex-container" };
const _hoisted_10$3 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Strength", -1);
const _hoisted_11$2 = { class: "flex-container" };
const _hoisted_12$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Mask Dilation", -1);
const _hoisted_13$2 = { class: "flex-container" };
const _hoisted_14$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Mask Blur", -1);
const _hoisted_15$2 = { class: "flex-container" };
const _hoisted_16$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Mask Padding", -1);
const _hoisted_17$2 = { class: "flex-container" };
const _hoisted_18$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Iterations", -1);
const _hoisted_19$1 = { class: "flex-container" };
const _hoisted_20 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Upscale", -1);
const _sfc_main$9 = /* @__PURE__ */ defineComponent({
  __name: "ADetailer",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const target = computed(() => {
      if (props.target === "settings") {
        return settings.data.settings;
      }
      return settings.defaultSettings;
    });
    const adetailerTarget = computed(
      () => props.target === "settings" ? "adetailer" : "defaultSettingsAdetailer"
    );
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createBaseVNode("div", _hoisted_1$7, [
          _hoisted_2$7,
          createVNode(unref(NSwitch), {
            value: target.value[props.tab].adetailer.enabled,
            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => target.value[props.tab].adetailer.enabled = $event)
          }, null, 8, ["value"])
        ]),
        target.value[props.tab].adetailer.enabled ? (openBlock(), createBlock(unref(NSpace), {
          key: 0,
          vertical: "",
          class: "left-container",
          "builtin-theme-overrides": {
            gapMedium: "0 12px"
          }
        }, {
          default: withCtx(() => [
            createVNode(unref(_sfc_main$2), {
              tab: "inpainting",
              target: adetailerTarget.value
            }, null, 8, ["target"]),
            createBaseVNode("div", _hoisted_3$6, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_4$5
                ]),
                default: withCtx(() => [
                  createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                  _hoisted_5$5
                ]),
                _: 1
              }),
              createVNode(unref(NSlider), {
                value: target.value[props.tab].adetailer.steps,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => target.value[props.tab].adetailer.steps = $event),
                min: 5,
                max: 300,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].adetailer.steps,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => target.value[props.tab].adetailer.steps = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" }
              }, null, 8, ["value"])
            ]),
            createVNode(unref(_sfc_main$8), {
              tab: "inpainting",
              target: adetailerTarget.value
            }, null, 8, ["target"]),
            createVNode(unref(_sfc_main$3), {
              tab: "inpainting",
              target: adetailerTarget.value
            }, null, 8, ["target"]),
            createBaseVNode("div", _hoisted_6$5, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_7$5
                ]),
                default: withCtx(() => [
                  createTextVNode(" Seed is a number that represents the starting canvas of your image. If you want to create the same image as your friend, you can use the same settings and seed to do so. "),
                  _hoisted_8$4
                ]),
                _: 1
              }),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].adetailer.seed,
                "onUpdate:value": _cache[3] || (_cache[3] = ($event) => target.value[props.tab].adetailer.seed = $event),
                size: "small",
                min: -1,
                max: 999999999999,
                style: { "flex-grow": "1" }
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_9$4, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_10$3
                ]),
                default: withCtx(() => [
                  createTextVNode(" How much should the masked are be changed from the original ")
                ]),
                _: 1
              }),
              createVNode(unref(NSlider), {
                value: target.value[props.tab].adetailer.strength,
                "onUpdate:value": _cache[4] || (_cache[4] = ($event) => target.value[props.tab].adetailer.strength = $event),
                min: 0,
                max: 1,
                step: 0.01,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].adetailer.strength,
                "onUpdate:value": _cache[5] || (_cache[5] = ($event) => target.value[props.tab].adetailer.strength = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                min: 0,
                max: 1,
                step: 0.01
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_11$2, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_12$2
                ]),
                default: withCtx(() => [
                  createTextVNode(" Expands bright pixels in the mask to cover more of the image. ")
                ]),
                _: 1
              }),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].adetailer.mask_dilation,
                "onUpdate:value": _cache[6] || (_cache[6] = ($event) => target.value[props.tab].adetailer.mask_dilation = $event),
                size: "small",
                min: 0,
                style: { "flex-grow": "1" }
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_13$2, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_14$2
                ]),
                default: withCtx(() => [
                  createTextVNode(" Makes for a smooth transition between masked and unmasked areas. ")
                ]),
                _: 1
              }),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].adetailer.mask_blur,
                "onUpdate:value": _cache[7] || (_cache[7] = ($event) => target.value[props.tab].adetailer.mask_blur = $event),
                size: "small",
                min: 0,
                style: { "flex-grow": "1" }
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_15$2, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_16$2
                ]),
                default: withCtx(() => [
                  createTextVNode(" Image will be cropped to the mask size plus padding. More padding might mean smoother transitions but slower generation. ")
                ]),
                _: 1
              }),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].adetailer.mask_padding,
                "onUpdate:value": _cache[8] || (_cache[8] = ($event) => target.value[props.tab].adetailer.mask_padding = $event),
                size: "small",
                min: 0,
                style: { "flex-grow": "1" }
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_17$2, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_18$2
                ]),
                default: withCtx(() => [
                  createTextVNode(" Iterations should increase the quality of the image at the cost of time. ")
                ]),
                _: 1
              }),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].adetailer.iterations,
                "onUpdate:value": _cache[9] || (_cache[9] = ($event) => target.value[props.tab].adetailer.iterations = $event),
                disabled: !unref(isDev),
                size: "small",
                min: 1,
                style: { "flex-grow": "1" }
              }, null, 8, ["value", "disabled"])
            ]),
            createBaseVNode("div", _hoisted_19$1, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_20
                ]),
                default: withCtx(() => [
                  createTextVNode(" Hom much should the image be upscaled before processing. This increases the quality of the image at the cost of time as bigger canvas can usually hold more detail. ")
                ]),
                _: 1
              }),
              createVNode(unref(NSlider), {
                value: target.value[props.tab].adetailer.upscale,
                "onUpdate:value": _cache[10] || (_cache[10] = ($event) => target.value[props.tab].adetailer.upscale = $event),
                min: 1,
                max: 4,
                step: 0.1,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].adetailer.upscale,
                "onUpdate:value": _cache[11] || (_cache[11] = ($event) => target.value[props.tab].adetailer.upscale = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                min: 1,
                max: 4,
                step: 0.1
              }, null, 8, ["value"])
            ])
          ]),
          _: 1
        })) : createCommentVNode("", true)
      ], 64);
    };
  }
});
const _hoisted_1$6 = { class: "flex-container" };
const _hoisted_2$6 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "CFG Scale", -1);
const _hoisted_3$5 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 3-15 for most images.", -1);
const _sfc_main$8 = /* @__PURE__ */ defineComponent({
  __name: "CFGScaleInput",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const cfgMax = computed(() => {
      var scale = 30;
      return scale + Math.max(
        settings.defaultSettings.api.apply_unsharp_mask ? 15 : 0,
        settings.defaultSettings.api.cfg_rescale_threshold == "off" ? 0 : 30
      );
    });
    const settingsTarget = computed(() => {
      let t;
      if (props.target === "settings") {
        t = settings.data.settings[props.tab];
      } else if (props.target === "adetailer") {
        t = settings.data.settings[props.tab].adetailer;
      } else if (props.target === "defaultSettingsAdetailer") {
        t = settings.defaultSettings[props.tab].adetailer;
      } else {
        t = settings.defaultSettings[props.tab];
      }
      return t;
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$6, [
        createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
          trigger: withCtx(() => [
            _hoisted_2$6
          ]),
          default: withCtx(() => [
            createTextVNode(" Guidance scale indicates how close should the model stay to the prompt. Higher values might be exactly what you want, but generated images might have some artifacts. Lower values give the model more freedom, and therefore might produce more coherent/less-artifacty images, but wouldn't follow the prompt as closely. "),
            _hoisted_3$5
          ]),
          _: 1
        }),
        createVNode(unref(NSlider), {
          value: settingsTarget.value.cfg_scale,
          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => settingsTarget.value.cfg_scale = $event),
          min: 1,
          max: cfgMax.value,
          step: 0.5,
          style: { "margin-right": "12px" }
        }, null, 8, ["value", "max"]),
        createVNode(unref(NInputNumber), {
          value: settingsTarget.value.cfg_scale,
          "onUpdate:value": _cache[1] || (_cache[1] = ($event) => settingsTarget.value.cfg_scale = $event),
          size: "small",
          style: { "min-width": "96px", "width": "96px" },
          min: 1,
          max: cfgMax.value,
          step: 0.5
        }, null, 8, ["value", "max"])
      ]);
    };
  }
});
const _hoisted_1$5 = { class: "flex-container" };
const _hoisted_2$5 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_3$4 = { key: 0 };
const _hoisted_4$4 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "diffusers", -1);
const _hoisted_5$4 = { class: "flex-container space-between" };
const _hoisted_6$4 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Depth", -1);
const _hoisted_7$4 = { class: "flex-container" };
const _hoisted_8$3 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Stop at", -1);
const _hoisted_9$3 = { class: "flex-container space-between" };
const _hoisted_10$2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Depth", -1);
const _hoisted_11$1 = { class: "flex-container" };
const _hoisted_12$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Stop at", -1);
const _hoisted_13$1 = { class: "flex-container" };
const _hoisted_14$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Scale", -1);
const _hoisted_15$1 = { class: "flex-container" };
const _hoisted_16$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Latent scaler", -1);
const _hoisted_17$1 = { class: "flex-container" };
const _hoisted_18$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Early out", -1);
const _sfc_main$7 = /* @__PURE__ */ defineComponent({
  __name: "DeepShrink",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const latentUpscalerOptions = [
      { label: "Nearest", value: "nearest" },
      { label: "Nearest exact", value: "nearest-exact" },
      { label: "Area", value: "area" },
      { label: "Bilinear", value: "bilinear" },
      { label: "Bicubic", value: "bicubic" },
      { label: "Bislerp", value: "bislerp" }
    ];
    const target = computed(() => {
      if (props.target === "settings") {
        return settings.data.settings;
      }
      return settings.defaultSettings;
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createBaseVNode("div", _hoisted_1$5, [
          _hoisted_2$5,
          createVNode(unref(NSwitch), {
            value: target.value[props.tab].deepshrink.enabled,
            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => target.value[props.tab].deepshrink.enabled = $event)
          }, null, 8, ["value"])
        ]),
        target.value[props.tab].deepshrink.enabled ? (openBlock(), createElementBlock("div", _hoisted_3$4, [
          createVNode(unref(NAlert), { type: "warning" }, {
            default: withCtx(() => [
              createTextVNode(" Only works on "),
              _hoisted_4$4,
              createTextVNode(" samplers ")
            ]),
            _: 1
          }),
          createVNode(unref(NCard), {
            bordered: false,
            title: "First layer"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_5$4, [
                _hoisted_6$4,
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].deepshrink.depth_1,
                  "onUpdate:value": _cache[1] || (_cache[1] = ($event) => target.value[props.tab].deepshrink.depth_1 = $event),
                  max: 4,
                  min: 1,
                  step: 1
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_7$4, [
                _hoisted_8$3,
                createVNode(unref(NSlider), {
                  value: target.value[props.tab].deepshrink.stop_at_1,
                  "onUpdate:value": _cache[2] || (_cache[2] = ($event) => target.value[props.tab].deepshrink.stop_at_1 = $event),
                  min: 0.05,
                  max: 1,
                  step: 0.05,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].deepshrink.stop_at_1,
                  "onUpdate:value": _cache[3] || (_cache[3] = ($event) => target.value[props.tab].deepshrink.stop_at_1 = $event),
                  max: 1,
                  min: 0.05,
                  step: 0.05
                }, null, 8, ["value"])
              ])
            ]),
            _: 1
          }),
          createVNode(unref(NCard), {
            bordered: false,
            title: "Second layer"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_9$3, [
                _hoisted_10$2,
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].deepshrink.depth_2,
                  "onUpdate:value": _cache[4] || (_cache[4] = ($event) => target.value[props.tab].deepshrink.depth_2 = $event),
                  max: 4,
                  min: 1,
                  step: 1
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_11$1, [
                _hoisted_12$1,
                createVNode(unref(NSlider), {
                  value: target.value[props.tab].deepshrink.stop_at_2,
                  "onUpdate:value": _cache[5] || (_cache[5] = ($event) => target.value[props.tab].deepshrink.stop_at_2 = $event),
                  min: 0.05,
                  max: 1,
                  step: 0.05
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].deepshrink.stop_at_2,
                  "onUpdate:value": _cache[6] || (_cache[6] = ($event) => target.value[props.tab].deepshrink.stop_at_2 = $event),
                  max: 1,
                  min: 0.05,
                  step: 0.05
                }, null, 8, ["value"])
              ])
            ]),
            _: 1
          }),
          createVNode(unref(NCard), {
            bordered: false,
            title: "Scale"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_13$1, [
                _hoisted_14$1,
                createVNode(unref(NSlider), {
                  value: target.value[props.tab].deepshrink.base_scale,
                  "onUpdate:value": _cache[7] || (_cache[7] = ($event) => target.value[props.tab].deepshrink.base_scale = $event),
                  min: 0.05,
                  max: 1,
                  step: 0.05
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].deepshrink.base_scale,
                  "onUpdate:value": _cache[8] || (_cache[8] = ($event) => target.value[props.tab].deepshrink.base_scale = $event),
                  max: 1,
                  min: 0.05,
                  step: 0.05
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_15$1, [
                _hoisted_16$1,
                createVNode(unref(NSelect), {
                  value: target.value[props.tab].deepshrink.scaler,
                  "onUpdate:value": _cache[9] || (_cache[9] = ($event) => target.value[props.tab].deepshrink.scaler = $event),
                  filterable: "",
                  options: latentUpscalerOptions
                }, null, 8, ["value"])
              ])
            ]),
            _: 1
          }),
          createVNode(unref(NCard), {
            bordered: false,
            title: "Other"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_17$1, [
                _hoisted_18$1,
                createVNode(unref(NSwitch), {
                  value: target.value[props.tab].deepshrink.early_out,
                  "onUpdate:value": _cache[10] || (_cache[10] = ($event) => target.value[props.tab].deepshrink.early_out = $event)
                }, null, 8, ["value"])
              ])
            ]),
            _: 1
          })
        ])) : createCommentVNode("", true)
      ], 64);
    };
  }
});
const _hoisted_1$4 = { class: "flex-container" };
const _hoisted_2$4 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_3$3 = { class: "flex-container" };
const _hoisted_4$3 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Mode")
], -1);
const _hoisted_5$3 = { key: 0 };
const _hoisted_6$3 = { class: "flex-container" };
const _hoisted_7$3 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Upscaler", -1);
const _hoisted_8$2 = { key: 1 };
const _hoisted_9$2 = { class: "flex-container" };
const _hoisted_10$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Antialiased", -1);
const _hoisted_11 = { class: "flex-container" };
const _hoisted_12 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Latent Mode", -1);
const _hoisted_13 = { class: "flex-container" };
const _hoisted_14 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Steps", -1);
const _hoisted_15 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "We recommend using 20-50 steps for most images.", -1);
const _hoisted_16 = { class: "flex-container" };
const _hoisted_17 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Scale", -1);
const _hoisted_18 = { class: "flex-container" };
const _hoisted_19 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Strength", -1);
const _sfc_main$6 = /* @__PURE__ */ defineComponent({
  __name: "HighResFix",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const global = useState();
    const target = computed(() => {
      if (props.target === "settings") {
        return settings.data.settings;
      }
      return settings.defaultSettings;
    });
    const imageUpscalerOptions = computed(() => {
      const localModels = global.state.models.filter(
        (model) => model.backend === "Upscaler" && !(upscalerOptions.map((option) => option.label).indexOf(model.name) !== -1)
      ).map((model) => ({
        label: model.name,
        value: model.path
      }));
      return [...upscalerOptions, ...localModels];
    });
    const latentUpscalerOptions = [
      { label: "Nearest", value: "nearest" },
      { label: "Nearest exact", value: "nearest-exact" },
      { label: "Area", value: "area" },
      { label: "Bilinear", value: "bilinear" },
      { label: "Bicubic", value: "bicubic" },
      { label: "Bislerp", value: "bislerp" }
    ];
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createBaseVNode("div", _hoisted_1$4, [
          _hoisted_2$4,
          createVNode(unref(NSwitch), {
            value: target.value[props.tab].highres.enabled,
            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => target.value[props.tab].highres.enabled = $event)
          }, null, 8, ["value"])
        ]),
        target.value[props.tab].highres.enabled ? (openBlock(), createBlock(unref(NSpace), {
          key: 0,
          vertical: "",
          class: "left-container"
        }, {
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_3$3, [
              _hoisted_4$3,
              createVNode(unref(NSelect), {
                value: target.value[props.tab].highres.mode,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => target.value[props.tab].highres.mode = $event),
                options: [
                  { label: "Latent", value: "latent" },
                  { label: "Image", value: "image" }
                ]
              }, null, 8, ["value"])
            ]),
            target.value[props.tab].highres.mode === "image" ? (openBlock(), createElementBlock("div", _hoisted_5$3, [
              createBaseVNode("div", _hoisted_6$3, [
                _hoisted_7$3,
                createVNode(unref(NSelect), {
                  value: target.value[props.tab].highres.image_upscaler,
                  "onUpdate:value": _cache[2] || (_cache[2] = ($event) => target.value[props.tab].highres.image_upscaler = $event),
                  size: "small",
                  style: { "flex-grow": "1" },
                  filterable: "",
                  options: imageUpscalerOptions.value
                }, null, 8, ["value", "options"])
              ])
            ])) : (openBlock(), createElementBlock("div", _hoisted_8$2, [
              createBaseVNode("div", _hoisted_9$2, [
                _hoisted_10$1,
                createVNode(unref(NSwitch), {
                  value: target.value[props.tab].highres.antialiased,
                  "onUpdate:value": _cache[3] || (_cache[3] = ($event) => target.value[props.tab].highres.antialiased = $event)
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_11, [
                _hoisted_12,
                createVNode(unref(NSelect), {
                  value: target.value[props.tab].highres.latent_scale_mode,
                  "onUpdate:value": _cache[4] || (_cache[4] = ($event) => target.value[props.tab].highres.latent_scale_mode = $event),
                  size: "small",
                  style: { "flex-grow": "1" },
                  filterable: "",
                  options: latentUpscalerOptions
                }, null, 8, ["value"])
              ])
            ])),
            createBaseVNode("div", _hoisted_13, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_14
                ]),
                default: withCtx(() => [
                  createTextVNode(" Number of steps to take in the diffusion process. Higher values will result in more detailed images but will take longer to generate. There is also a point of diminishing returns around 100 steps. "),
                  _hoisted_15
                ]),
                _: 1
              }),
              createVNode(unref(NSlider), {
                value: target.value[props.tab].highres.steps,
                "onUpdate:value": _cache[5] || (_cache[5] = ($event) => target.value[props.tab].highres.steps = $event),
                min: 5,
                max: 300,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].highres.steps,
                "onUpdate:value": _cache[6] || (_cache[6] = ($event) => target.value[props.tab].highres.steps = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" }
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_16, [
              _hoisted_17,
              createVNode(unref(NSlider), {
                value: target.value[props.tab].highres.scale,
                "onUpdate:value": _cache[7] || (_cache[7] = ($event) => target.value[props.tab].highres.scale = $event),
                min: 1,
                max: 8,
                step: 0.1,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].highres.scale,
                "onUpdate:value": _cache[8] || (_cache[8] = ($event) => target.value[props.tab].highres.scale = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                step: 0.1
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_18, [
              _hoisted_19,
              createVNode(unref(NSlider), {
                value: target.value[props.tab].highres.strength,
                "onUpdate:value": _cache[9] || (_cache[9] = ($event) => target.value[props.tab].highres.strength = $event),
                min: 0.1,
                max: 0.9,
                step: 0.05,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: target.value[props.tab].highres.strength,
                "onUpdate:value": _cache[10] || (_cache[10] = ($event) => target.value[props.tab].highres.strength = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                min: 0.1,
                max: 0.9,
                step: 0.05
              }, null, 8, ["value"])
            ])
          ]),
          _: 1
        })) : createCommentVNode("", true)
      ], 64);
    };
  }
});
const _sfc_main$5 = /* @__PURE__ */ defineComponent({
  __name: "HighResFixTabs",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), {
        title: "High Resolution Fix",
        class: "generate-extra-card"
      }, {
        default: withCtx(() => [
          createVNode(unref(NTabs), {
            animated: "",
            type: "segment"
          }, {
            default: withCtx(() => [
              createVNode(unref(NTabPane), {
                tab: "Image to Image",
                name: "highresfix"
              }, {
                default: withCtx(() => [
                  createVNode(unref(_sfc_main$6), {
                    tab: props.tab,
                    target: props.target
                  }, null, 8, ["tab", "target"])
                ]),
                _: 1
              }),
              createVNode(unref(NTabPane), {
                tab: "Scalecrafter",
                name: "scalecrafter"
              }, {
                default: withCtx(() => [
                  createVNode(unref(_sfc_main$1), {
                    tab: props.tab,
                    target: props.target
                  }, null, 8, ["tab", "target"])
                ]),
                _: 1
              }),
              createVNode(unref(NTabPane), {
                tab: "DeepShrink",
                name: "deepshrink"
              }, {
                default: withCtx(() => [
                  createVNode(unref(_sfc_main$7), {
                    tab: props.tab,
                    target: props.target
                  }, null, 8, ["tab", "target"])
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
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "Restoration",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), {
        title: "Restoration",
        class: "generate-extra-card"
      }, {
        default: withCtx(() => [
          createVNode(unref(NTabs), {
            animated: "",
            type: "segment"
          }, {
            default: withCtx(() => [
              createVNode(unref(NTabPane), {
                tab: "ADetailer",
                name: "adetailer"
              }, {
                default: withCtx(() => [
                  createVNode(unref(_sfc_main$9), {
                    tab: props.tab,
                    target: props.target
                  }, null, 8, ["tab", "target"])
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
const _hoisted_1$3 = {
  key: 0,
  class: "flex-container"
};
const _hoisted_2$3 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Self Attention Scale", -1);
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "SAGInput",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const settingsTarget = computed(() => {
      let t;
      if (props.target === "settings") {
        t = settings.data.settings[props.tab];
      } else if (props.target === "adetailer") {
        t = settings.data.settings[props.tab].adetailer;
      } else if (props.target === "defaultSettingsAdetailer") {
        t = settings.defaultSettings[props.tab].adetailer;
      } else {
        t = settings.defaultSettings[props.tab];
      }
      return t;
    });
    return (_ctx, _cache) => {
      var _a;
      return ((_a = unref(settings).data.settings.model) == null ? void 0 : _a.backend) === "PyTorch" ? (openBlock(), createElementBlock("div", _hoisted_1$3, [
        createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
          trigger: withCtx(() => [
            _hoisted_2$3
          ]),
          default: withCtx(() => [
            createTextVNode(" If self attention is >0, SAG will guide the model and improve the quality of the image at the cost of speed. Higher values will follow the guidance more closely, which can lead to better, more sharp and detailed outputs. ")
          ]),
          _: 1
        }),
        createVNode(unref(NSlider), {
          value: settingsTarget.value.self_attention_scale,
          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => settingsTarget.value.self_attention_scale = $event),
          min: 0,
          max: 1,
          step: 0.05,
          style: { "margin-right": "12px" }
        }, null, 8, ["value"]),
        createVNode(unref(NInputNumber), {
          value: settingsTarget.value.self_attention_scale,
          "onUpdate:value": _cache[1] || (_cache[1] = ($event) => settingsTarget.value.self_attention_scale = $event),
          size: "small",
          style: { "min-width": "96px", "width": "96px" },
          step: 0.05
        }, null, 8, ["value"])
      ])) : createCommentVNode("", true);
    };
  }
});
const _hoisted_1$2 = { class: "flex-container" };
const _hoisted_2$2 = { style: { "margin-left": "12px", "margin-right": "12px", "white-space": "nowrap" } };
const _hoisted_3$2 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "100px" } }, "Sampler", -1);
const _hoisted_4$2 = /* @__PURE__ */ createBaseVNode("a", {
  target: "_blank",
  href: "https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
}, "Learn more", -1);
const _hoisted_5$2 = { class: "flex-container" };
const _hoisted_6$2 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "94px" } }, "Sigmas", -1);
const _hoisted_7$2 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, 'Only "Default" and "Karras" sigmas work on diffusers samplers (and "Karras" are only applied to KDPM samplers)', -1);
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "SamplerPicker",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const showModal = ref(false);
    function getValue2(param) {
      const val = sigmasTarget.value.sampler_config[settingsTarget.value.sampler][param];
      return val;
    }
    function setValue(param, value) {
      sigmasTarget.value.sampler_config[settingsTarget.value.sampler][param] = value;
    }
    function resolveComponent(settings2, param) {
      switch (settings2.componentType) {
        case "slider":
          return h(NSlider, {
            min: settings2.min,
            max: settings2.max,
            step: settings2.step,
            value: getValue2(param),
            onUpdateValue: (value) => setValue(param, value)
          });
        case "select":
          return h(NSelect, {
            options: settings2.options,
            value: getValue2(param),
            onUpdateValue: (value) => setValue(param, value)
          });
        case "boolean":
          return h(NCheckbox, {
            checked: getValue2(param),
            onUpdateChecked: (value) => setValue(param, value)
          });
        case "number":
          return h(NInputNumber, {
            min: settings2.min,
            max: settings2.max,
            step: settings2.step,
            value: getValue2(param),
            onUpdateValue: (value) => setValue(param, value)
          });
      }
    }
    const settingsTarget = computed(() => {
      let t;
      if (props.target === "settings") {
        t = settings.data.settings[props.tab];
      } else if (props.target === "adetailer") {
        t = settings.data.settings[props.tab].adetailer;
      } else if (props.target === "defaultSettingsAdetailer") {
        t = settings.defaultSettings[props.tab].adetailer;
      } else {
        t = settings.defaultSettings[props.tab];
      }
      return t;
    });
    const sigmasTarget = computed(() => {
      if (props.target === "settings") {
        return settings.data.settings;
      }
      return settings.defaultSettings;
    });
    const computedSettings = computed(() => {
      return sigmasTarget.value.sampler_config[settingsTarget.value.sampler] ?? {};
    });
    const sigmaOptions = computed(() => {
      const karras = typeof settingsTarget.value.sampler === "string";
      return [
        {
          label: "Automatic",
          value: "automatic"
        },
        {
          label: "Karras",
          value: "karras"
        },
        {
          label: "Exponential",
          value: "exponential",
          disabled: !karras
        },
        {
          label: "Polyexponential",
          value: "polyexponential",
          disabled: !karras
        },
        {
          label: "VP",
          value: "vp",
          disabled: !karras
        }
      ];
    });
    const sigmaValidationStatus = computed(() => {
      if (typeof settingsTarget.value.sampler !== "string") {
        if (!["automatic", "karras"].includes(settingsTarget.value.sigmas)) {
          return "error";
        } else {
          return void 0;
        }
      }
      return void 0;
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createBaseVNode("div", _hoisted_1$2, [
          createVNode(unref(NModal), {
            show: showModal.value,
            "onUpdate:show": _cache[1] || (_cache[1] = ($event) => showModal.value = $event),
            "close-on-esc": "",
            "mask-closable": ""
          }, {
            default: withCtx(() => [
              createVNode(unref(NCard), {
                title: "Sampler settings",
                style: { "max-width": "90vw", "max-height": "90vh" },
                closable: "",
                onClose: _cache[0] || (_cache[0] = ($event) => showModal.value = false)
              }, {
                default: withCtx(() => [
                  (openBlock(true), createElementBlock(Fragment, null, renderList(Object.keys(computedSettings.value), (param) => {
                    return openBlock(), createElementBlock("div", {
                      class: "flex-container",
                      key: param
                    }, [
                      createVNode(unref(NButton), {
                        type: computedSettings.value[param] !== null ? "error" : "default",
                        ghost: "",
                        disabled: computedSettings.value[param] === null,
                        onClick: ($event) => setValue(param, null),
                        style: { "min-width": "100px" }
                      }, {
                        default: withCtx(() => [
                          createTextVNode(toDisplayString(computedSettings.value[param] !== null ? "Reset" : "Disabled"), 1)
                        ]),
                        _: 2
                      }, 1032, ["type", "disabled", "onClick"]),
                      createBaseVNode("p", _hoisted_2$2, toDisplayString(unref(convertToTextString)(param)), 1),
                      (openBlock(), createBlock(resolveDynamicComponent(
                        resolveComponent(
                          sigmasTarget.value.sampler_config["ui_settings"][param],
                          param
                        )
                      )))
                    ]);
                  }), 128))
                ]),
                _: 1
              })
            ]),
            _: 1
          }, 8, ["show"]),
          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
            trigger: withCtx(() => [
              _hoisted_3$2
            ]),
            default: withCtx(() => [
              createTextVNode(" The sampler is the method used to generate the image. Your result may vary drastically depending on the sampler you choose. "),
              _hoisted_4$2
            ]),
            _: 1
          }),
          createVNode(unref(NSelect), {
            options: unref(settings).scheduler_options,
            filterable: "",
            value: settingsTarget.value.sampler,
            "onUpdate:value": _cache[2] || (_cache[2] = ($event) => settingsTarget.value.sampler = $event),
            style: { "flex-grow": "1" }
          }, null, 8, ["options", "value"]),
          createVNode(unref(NButton), {
            style: { "margin-left": "4px" },
            onClick: _cache[3] || (_cache[3] = ($event) => showModal.value = true)
          }, {
            default: withCtx(() => [
              createVNode(unref(NIcon), null, {
                default: withCtx(() => [
                  createVNode(unref(Settings))
                ]),
                _: 1
              })
            ]),
            _: 1
          })
        ]),
        createBaseVNode("div", _hoisted_5$2, [
          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
            trigger: withCtx(() => [
              _hoisted_6$2
            ]),
            default: withCtx(() => [
              createTextVNode(" Changes the sigmas used in the diffusion process. Can change the quality of the output. "),
              _hoisted_7$2
            ]),
            _: 1
          }),
          createVNode(unref(NSelect), {
            options: sigmaOptions.value,
            value: settingsTarget.value.sigmas,
            "onUpdate:value": _cache[4] || (_cache[4] = ($event) => settingsTarget.value.sigmas = $event),
            status: sigmaValidationStatus.value
          }, null, 8, ["options", "value", "status"])
        ])
      ], 64);
    };
  }
});
const _hoisted_1$1 = { class: "flex-container" };
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode("div", { class: "slider-label" }, [
  /* @__PURE__ */ createBaseVNode("p", null, "Enabled")
], -1);
const _hoisted_3$1 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "Automatic", -1);
const _hoisted_4$1 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, "Karras", -1);
const _hoisted_5$1 = { class: "flex-container" };
const _hoisted_6$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Disperse", -1);
const _hoisted_7$1 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, " However, this comes at the cost of increased vram usage, generally in the range of 3-4x. ", -1);
const _hoisted_8$1 = { class: "flex-container" };
const _hoisted_9$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Unsafe resolutions", -1);
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "Scalecrafter",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    const target = computed(() => {
      if (props.target === "settings") {
        return settings.data.settings;
      }
      return settings.defaultSettings;
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createBaseVNode("div", _hoisted_1$1, [
          _hoisted_2$1,
          createVNode(unref(NSwitch), {
            value: target.value[props.tab].scalecrafter.enabled,
            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => target.value[props.tab].scalecrafter.enabled = $event)
          }, null, 8, ["value"])
        ]),
        target.value[props.tab].scalecrafter.enabled ? (openBlock(), createBlock(unref(NSpace), {
          key: 0,
          vertical: "",
          class: "left-container"
        }, {
          default: withCtx(() => [
            createVNode(unref(NAlert), { type: "warning" }, {
              default: withCtx(() => [
                createTextVNode(" Only works with "),
                _hoisted_3$1,
                createTextVNode(" and "),
                _hoisted_4$1,
                createTextVNode(" sigmas ")
              ]),
              _: 1
            }),
            createBaseVNode("div", _hoisted_5$1, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_6$1
                ]),
                default: withCtx(() => [
                  createTextVNode(" May generate more unique images. "),
                  _hoisted_7$1
                ]),
                _: 1
              }),
              createVNode(unref(NSwitch), {
                value: target.value[props.tab].scalecrafter.disperse,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => target.value[props.tab].scalecrafter.disperse = $event)
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_8$1, [
              createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                trigger: withCtx(() => [
                  _hoisted_9$1
                ]),
                default: withCtx(() => [
                  createTextVNode(" Allow generating with unique resolutions that don't have configs ready for them, or clamp them (really, force them) to the closest resolution. ")
                ]),
                _: 1
              }),
              createVNode(unref(NSwitch), {
                value: target.value[props.tab].scalecrafter.unsafe_resolutions,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => target.value[props.tab].scalecrafter.unsafe_resolutions = $event)
              }, null, 8, ["value"])
            ])
          ]),
          _: 1
        })) : createCommentVNode("", true)
      ], 64);
    };
  }
});
const _hoisted_1 = { class: "flex-container" };
const _hoisted_2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Enabled", -1);
const _hoisted_3 = { class: "flex-container" };
const _hoisted_4 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Model", -1);
const _hoisted_5 = { class: "flex-container" };
const _hoisted_6 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Scale Factor", -1);
const _hoisted_7 = { class: "flex-container" };
const _hoisted_8 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Tile Size", -1);
const _hoisted_9 = { class: "flex-container" };
const _hoisted_10 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Tile Padding", -1);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "Upscale",
  props: {
    tab: {
      type: String,
      required: true
    },
    target: {
      type: String,
      required: false,
      default: "settings"
    }
  },
  setup(__props) {
    const props = __props;
    const global = useState();
    const settings = useSettings();
    const target = computed(() => {
      if (props.target === "settings") {
        return settings.data.settings;
      }
      return settings.defaultSettings;
    });
    const upscalerOptionsFull = computed(() => {
      const localModels = global.state.models.filter(
        (model) => model.backend === "Upscaler" && !(upscalerOptions.map((option) => option.label).indexOf(model.name) !== -1)
      ).map((model) => ({
        label: model.name,
        value: model.path
      }));
      return [...upscalerOptions, ...localModels];
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), {
        title: "Upscale",
        class: "generate-extra-card"
      }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1, [
            _hoisted_2,
            createVNode(unref(NSwitch), {
              value: target.value[props.tab].upscale.enabled,
              "onUpdate:value": _cache[0] || (_cache[0] = ($event) => target.value[props.tab].upscale.enabled = $event)
            }, null, 8, ["value"])
          ]),
          target.value[props.tab].upscale.enabled ? (openBlock(), createBlock(unref(NSpace), {
            key: 0,
            vertical: "",
            class: "left-container"
          }, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_3, [
                _hoisted_4,
                createVNode(unref(NSelect), {
                  value: target.value[props.tab].upscale.model,
                  "onUpdate:value": _cache[1] || (_cache[1] = ($event) => target.value[props.tab].upscale.model = $event),
                  style: { "margin-right": "12px" },
                  filterable: "",
                  tag: "",
                  options: upscalerOptionsFull.value
                }, null, 8, ["value", "options"])
              ]),
              createBaseVNode("div", _hoisted_5, [
                createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                  trigger: withCtx(() => [
                    _hoisted_6
                  ]),
                  default: withCtx(() => [
                    createTextVNode(" TODO ")
                  ]),
                  _: 1
                }),
                createVNode(unref(NSlider), {
                  value: target.value[props.tab].upscale.upscale_factor,
                  "onUpdate:value": _cache[2] || (_cache[2] = ($event) => target.value[props.tab].upscale.upscale_factor = $event),
                  min: 1,
                  max: 4,
                  step: 0.1,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].upscale.upscale_factor,
                  "onUpdate:value": _cache[3] || (_cache[3] = ($event) => target.value[props.tab].upscale.upscale_factor = $event),
                  size: "small",
                  style: { "min-width": "96px", "width": "96px" },
                  min: 1,
                  max: 4,
                  step: 0.1
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_7, [
                createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                  trigger: withCtx(() => [
                    _hoisted_8
                  ]),
                  default: withCtx(() => [
                    createTextVNode(" How large each tile should be. Larger tiles will use more memory. 0 will disable tiling. ")
                  ]),
                  _: 1
                }),
                createVNode(unref(NSlider), {
                  value: target.value[props.tab].upscale.tile_size,
                  "onUpdate:value": _cache[4] || (_cache[4] = ($event) => target.value[props.tab].upscale.tile_size = $event),
                  min: 32,
                  max: 2048,
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].upscale.tile_size,
                  "onUpdate:value": _cache[5] || (_cache[5] = ($event) => target.value[props.tab].upscale.tile_size = $event),
                  size: "small",
                  min: 32,
                  max: 2048,
                  style: { "min-width": "96px", "width": "96px" }
                }, null, 8, ["value"])
              ]),
              createBaseVNode("div", _hoisted_9, [
                createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                  trigger: withCtx(() => [
                    _hoisted_10
                  ]),
                  default: withCtx(() => [
                    createTextVNode(" How much should tiles overlap. Larger padding will use more memory, but image should not have visible seams. ")
                  ]),
                  _: 1
                }),
                createVNode(unref(NSlider), {
                  value: target.value[props.tab].upscale.tile_padding,
                  "onUpdate:value": _cache[6] || (_cache[6] = ($event) => target.value[props.tab].upscale.tile_padding = $event),
                  style: { "margin-right": "12px" }
                }, null, 8, ["value"]),
                createVNode(unref(NInputNumber), {
                  value: target.value[props.tab].upscale.tile_padding,
                  "onUpdate:value": _cache[7] || (_cache[7] = ($event) => target.value[props.tab].upscale.tile_padding = $event),
                  size: "small",
                  style: { "min-width": "96px", "width": "96px" }
                }, null, 8, ["value"])
              ])
            ]),
            _: 1
          })) : createCommentVNode("", true)
        ]),
        _: 1
      });
    };
  }
});
export {
  NForm as N,
  _sfc_main$2 as _,
  _sfc_main$8 as a,
  _sfc_main$3 as b,
  _sfc_main$5 as c,
  _sfc_main as d,
  _sfc_main$4 as e,
  NFormItem as f
};
