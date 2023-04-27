import { be as toString, bf as replaceable, z as h, d as defineComponent, bg as isBrowser, Z as useTheme, O as createInjectionKey, bh as createTheme, bi as commonLight, bj as tooltipLight, Q as c, R as cB, bk as fadeInTransition, aR as fadeInScaleUpTransition, ak as cNotM, M as toRef, A as ref, al as useLocale, a7 as watch, aB as onBeforeUnmount, aC as off, c as computed, I as useConfig, a2 as useThemeClass, bl as isMounted, bm as LazyTeleport, bn as withDirectives, bo as zindexable, aS as Transition, F as Fragment, ao as NBaseIcon, bp as vShow, a5 as on, X as inject, bq as normalizeStyle, m as NTooltip, aQ as beforeNextFrameOnce, a3 as createId, L as provide, b4 as getCurrentInstance, b8 as onMounted, am as watchEffect } from "./index.js";
function arrayReduce(array, iteratee, accumulator, initAccum) {
  var index = -1, length = array == null ? 0 : array.length;
  if (initAccum && length) {
    accumulator = array[++index];
  }
  while (++index < length) {
    accumulator = iteratee(accumulator, array[index], index, array);
  }
  return accumulator;
}
function basePropertyOf(object) {
  return function(key) {
    return object == null ? void 0 : object[key];
  };
}
var deburredLetters = {
  // Latin-1 Supplement block.
  "À": "A",
  "Á": "A",
  "Â": "A",
  "Ã": "A",
  "Ä": "A",
  "Å": "A",
  "à": "a",
  "á": "a",
  "â": "a",
  "ã": "a",
  "ä": "a",
  "å": "a",
  "Ç": "C",
  "ç": "c",
  "Ð": "D",
  "ð": "d",
  "È": "E",
  "É": "E",
  "Ê": "E",
  "Ë": "E",
  "è": "e",
  "é": "e",
  "ê": "e",
  "ë": "e",
  "Ì": "I",
  "Í": "I",
  "Î": "I",
  "Ï": "I",
  "ì": "i",
  "í": "i",
  "î": "i",
  "ï": "i",
  "Ñ": "N",
  "ñ": "n",
  "Ò": "O",
  "Ó": "O",
  "Ô": "O",
  "Õ": "O",
  "Ö": "O",
  "Ø": "O",
  "ò": "o",
  "ó": "o",
  "ô": "o",
  "õ": "o",
  "ö": "o",
  "ø": "o",
  "Ù": "U",
  "Ú": "U",
  "Û": "U",
  "Ü": "U",
  "ù": "u",
  "ú": "u",
  "û": "u",
  "ü": "u",
  "Ý": "Y",
  "ý": "y",
  "ÿ": "y",
  "Æ": "Ae",
  "æ": "ae",
  "Þ": "Th",
  "þ": "th",
  "ß": "ss",
  // Latin Extended-A block.
  "Ā": "A",
  "Ă": "A",
  "Ą": "A",
  "ā": "a",
  "ă": "a",
  "ą": "a",
  "Ć": "C",
  "Ĉ": "C",
  "Ċ": "C",
  "Č": "C",
  "ć": "c",
  "ĉ": "c",
  "ċ": "c",
  "č": "c",
  "Ď": "D",
  "Đ": "D",
  "ď": "d",
  "đ": "d",
  "Ē": "E",
  "Ĕ": "E",
  "Ė": "E",
  "Ę": "E",
  "Ě": "E",
  "ē": "e",
  "ĕ": "e",
  "ė": "e",
  "ę": "e",
  "ě": "e",
  "Ĝ": "G",
  "Ğ": "G",
  "Ġ": "G",
  "Ģ": "G",
  "ĝ": "g",
  "ğ": "g",
  "ġ": "g",
  "ģ": "g",
  "Ĥ": "H",
  "Ħ": "H",
  "ĥ": "h",
  "ħ": "h",
  "Ĩ": "I",
  "Ī": "I",
  "Ĭ": "I",
  "Į": "I",
  "İ": "I",
  "ĩ": "i",
  "ī": "i",
  "ĭ": "i",
  "į": "i",
  "ı": "i",
  "Ĵ": "J",
  "ĵ": "j",
  "Ķ": "K",
  "ķ": "k",
  "ĸ": "k",
  "Ĺ": "L",
  "Ļ": "L",
  "Ľ": "L",
  "Ŀ": "L",
  "Ł": "L",
  "ĺ": "l",
  "ļ": "l",
  "ľ": "l",
  "ŀ": "l",
  "ł": "l",
  "Ń": "N",
  "Ņ": "N",
  "Ň": "N",
  "Ŋ": "N",
  "ń": "n",
  "ņ": "n",
  "ň": "n",
  "ŋ": "n",
  "Ō": "O",
  "Ŏ": "O",
  "Ő": "O",
  "ō": "o",
  "ŏ": "o",
  "ő": "o",
  "Ŕ": "R",
  "Ŗ": "R",
  "Ř": "R",
  "ŕ": "r",
  "ŗ": "r",
  "ř": "r",
  "Ś": "S",
  "Ŝ": "S",
  "Ş": "S",
  "Š": "S",
  "ś": "s",
  "ŝ": "s",
  "ş": "s",
  "š": "s",
  "Ţ": "T",
  "Ť": "T",
  "Ŧ": "T",
  "ţ": "t",
  "ť": "t",
  "ŧ": "t",
  "Ũ": "U",
  "Ū": "U",
  "Ŭ": "U",
  "Ů": "U",
  "Ű": "U",
  "Ų": "U",
  "ũ": "u",
  "ū": "u",
  "ŭ": "u",
  "ů": "u",
  "ű": "u",
  "ų": "u",
  "Ŵ": "W",
  "ŵ": "w",
  "Ŷ": "Y",
  "ŷ": "y",
  "Ÿ": "Y",
  "Ź": "Z",
  "Ż": "Z",
  "Ž": "Z",
  "ź": "z",
  "ż": "z",
  "ž": "z",
  "Ĳ": "IJ",
  "ĳ": "ij",
  "Œ": "Oe",
  "œ": "oe",
  "ŉ": "'n",
  "ſ": "s"
};
var deburrLetter = basePropertyOf(deburredLetters);
const deburrLetter$1 = deburrLetter;
var reLatin = /[\xc0-\xd6\xd8-\xf6\xf8-\xff\u0100-\u017f]/g;
var rsComboMarksRange$1 = "\\u0300-\\u036f", reComboHalfMarksRange$1 = "\\ufe20-\\ufe2f", rsComboSymbolsRange$1 = "\\u20d0-\\u20ff", rsComboRange$1 = rsComboMarksRange$1 + reComboHalfMarksRange$1 + rsComboSymbolsRange$1;
var rsCombo$1 = "[" + rsComboRange$1 + "]";
var reComboMark = RegExp(rsCombo$1, "g");
function deburr(string) {
  string = toString(string);
  return string && string.replace(reLatin, deburrLetter$1).replace(reComboMark, "");
}
var reAsciiWord = /[^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+/g;
function asciiWords(string) {
  return string.match(reAsciiWord) || [];
}
var reHasUnicodeWord = /[a-z][A-Z]|[A-Z]{2}[a-z]|[0-9][a-zA-Z]|[a-zA-Z][0-9]|[^a-zA-Z0-9 ]/;
function hasUnicodeWord(string) {
  return reHasUnicodeWord.test(string);
}
var rsAstralRange = "\\ud800-\\udfff", rsComboMarksRange = "\\u0300-\\u036f", reComboHalfMarksRange = "\\ufe20-\\ufe2f", rsComboSymbolsRange = "\\u20d0-\\u20ff", rsComboRange = rsComboMarksRange + reComboHalfMarksRange + rsComboSymbolsRange, rsDingbatRange = "\\u2700-\\u27bf", rsLowerRange = "a-z\\xdf-\\xf6\\xf8-\\xff", rsMathOpRange = "\\xac\\xb1\\xd7\\xf7", rsNonCharRange = "\\x00-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\xbf", rsPunctuationRange = "\\u2000-\\u206f", rsSpaceRange = " \\t\\x0b\\f\\xa0\\ufeff\\n\\r\\u2028\\u2029\\u1680\\u180e\\u2000\\u2001\\u2002\\u2003\\u2004\\u2005\\u2006\\u2007\\u2008\\u2009\\u200a\\u202f\\u205f\\u3000", rsUpperRange = "A-Z\\xc0-\\xd6\\xd8-\\xde", rsVarRange = "\\ufe0e\\ufe0f", rsBreakRange = rsMathOpRange + rsNonCharRange + rsPunctuationRange + rsSpaceRange;
var rsApos$1 = "['’]", rsBreak = "[" + rsBreakRange + "]", rsCombo = "[" + rsComboRange + "]", rsDigits = "\\d+", rsDingbat = "[" + rsDingbatRange + "]", rsLower = "[" + rsLowerRange + "]", rsMisc = "[^" + rsAstralRange + rsBreakRange + rsDigits + rsDingbatRange + rsLowerRange + rsUpperRange + "]", rsFitz = "\\ud83c[\\udffb-\\udfff]", rsModifier = "(?:" + rsCombo + "|" + rsFitz + ")", rsNonAstral = "[^" + rsAstralRange + "]", rsRegional = "(?:\\ud83c[\\udde6-\\uddff]){2}", rsSurrPair = "[\\ud800-\\udbff][\\udc00-\\udfff]", rsUpper = "[" + rsUpperRange + "]", rsZWJ = "\\u200d";
var rsMiscLower = "(?:" + rsLower + "|" + rsMisc + ")", rsMiscUpper = "(?:" + rsUpper + "|" + rsMisc + ")", rsOptContrLower = "(?:" + rsApos$1 + "(?:d|ll|m|re|s|t|ve))?", rsOptContrUpper = "(?:" + rsApos$1 + "(?:D|LL|M|RE|S|T|VE))?", reOptMod = rsModifier + "?", rsOptVar = "[" + rsVarRange + "]?", rsOptJoin = "(?:" + rsZWJ + "(?:" + [rsNonAstral, rsRegional, rsSurrPair].join("|") + ")" + rsOptVar + reOptMod + ")*", rsOrdLower = "\\d*(?:1st|2nd|3rd|(?![123])\\dth)(?=\\b|[A-Z_])", rsOrdUpper = "\\d*(?:1ST|2ND|3RD|(?![123])\\dTH)(?=\\b|[a-z_])", rsSeq = rsOptVar + reOptMod + rsOptJoin, rsEmoji = "(?:" + [rsDingbat, rsRegional, rsSurrPair].join("|") + ")" + rsSeq;
var reUnicodeWord = RegExp([
  rsUpper + "?" + rsLower + "+" + rsOptContrLower + "(?=" + [rsBreak, rsUpper, "$"].join("|") + ")",
  rsMiscUpper + "+" + rsOptContrUpper + "(?=" + [rsBreak, rsUpper + rsMiscLower, "$"].join("|") + ")",
  rsUpper + "?" + rsMiscLower + "+" + rsOptContrLower,
  rsUpper + "+" + rsOptContrUpper,
  rsOrdUpper,
  rsOrdLower,
  rsDigits,
  rsEmoji
].join("|"), "g");
function unicodeWords(string) {
  return string.match(reUnicodeWord) || [];
}
function words(string, pattern, guard) {
  string = toString(string);
  pattern = guard ? void 0 : pattern;
  if (pattern === void 0) {
    return hasUnicodeWord(string) ? unicodeWords(string) : asciiWords(string);
  }
  return string.match(pattern) || [];
}
var rsApos = "['’]";
var reApos = RegExp(rsApos, "g");
function createCompounder(callback) {
  return function(string) {
    return arrayReduce(words(deburr(string).replace(reApos, "")), callback, "");
  };
}
var kebabCase = createCompounder(function(result, word, index) {
  return result + (index ? "-" : "") + word.toLowerCase();
});
const kebabCase$1 = kebabCase;
const RotateClockwiseIcon = replaceable("rotateClockwise", h(
  "svg",
  { viewBox: "0 0 20 20", fill: "none", xmlns: "http://www.w3.org/2000/svg" },
  h("path", { d: "M3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10C17 12.7916 15.3658 15.2026 13 16.3265V14.5C13 14.2239 12.7761 14 12.5 14C12.2239 14 12 14.2239 12 14.5V17.5C12 17.7761 12.2239 18 12.5 18H15.5C15.7761 18 16 17.7761 16 17.5C16 17.2239 15.7761 17 15.5 17H13.8758C16.3346 15.6357 18 13.0128 18 10C18 5.58172 14.4183 2 10 2C5.58172 2 2 5.58172 2 10C2 10.2761 2.22386 10.5 2.5 10.5C2.77614 10.5 3 10.2761 3 10Z", fill: "currentColor" }),
  h("path", { d: "M10 12C11.1046 12 12 11.1046 12 10C12 8.89543 11.1046 8 10 8C8.89543 8 8 8.89543 8 10C8 11.1046 8.89543 12 10 12ZM10 11C9.44772 11 9 10.5523 9 10C9 9.44772 9.44772 9 10 9C10.5523 9 11 9.44772 11 10C11 10.5523 10.5523 11 10 11Z", fill: "currentColor" })
));
const RotateCounterclockwiseIcon = replaceable("rotateClockwise", h(
  "svg",
  { viewBox: "0 0 20 20", fill: "none", xmlns: "http://www.w3.org/2000/svg" },
  h("path", { d: "M17 10C17 6.13401 13.866 3 10 3C6.13401 3 3 6.13401 3 10C3 12.7916 4.63419 15.2026 7 16.3265V14.5C7 14.2239 7.22386 14 7.5 14C7.77614 14 8 14.2239 8 14.5V17.5C8 17.7761 7.77614 18 7.5 18H4.5C4.22386 18 4 17.7761 4 17.5C4 17.2239 4.22386 17 4.5 17H6.12422C3.66539 15.6357 2 13.0128 2 10C2 5.58172 5.58172 2 10 2C14.4183 2 18 5.58172 18 10C18 10.2761 17.7761 10.5 17.5 10.5C17.2239 10.5 17 10.2761 17 10Z", fill: "currentColor" }),
  h("path", { d: "M10 12C8.89543 12 8 11.1046 8 10C8 8.89543 8.89543 8 10 8C11.1046 8 12 8.89543 12 10C12 11.1046 11.1046 12 10 12ZM10 11C10.5523 11 11 10.5523 11 10C11 9.44772 10.5523 9 10 9C9.44772 9 9 9.44772 9 10C9 10.5523 9.44772 11 10 11Z", fill: "currentColor" })
));
const ZoomInIcon = replaceable("zoomIn", h(
  "svg",
  { viewBox: "0 0 20 20", fill: "none", xmlns: "http://www.w3.org/2000/svg" },
  h("path", { d: "M11.5 8.5C11.5 8.22386 11.2761 8 11 8H9V6C9 5.72386 8.77614 5.5 8.5 5.5C8.22386 5.5 8 5.72386 8 6V8H6C5.72386 8 5.5 8.22386 5.5 8.5C5.5 8.77614 5.72386 9 6 9H8V11C8 11.2761 8.22386 11.5 8.5 11.5C8.77614 11.5 9 11.2761 9 11V9H11C11.2761 9 11.5 8.77614 11.5 8.5Z", fill: "currentColor" }),
  h("path", { d: "M8.5 3C11.5376 3 14 5.46243 14 8.5C14 9.83879 13.5217 11.0659 12.7266 12.0196L16.8536 16.1464C17.0488 16.3417 17.0488 16.6583 16.8536 16.8536C16.68 17.0271 16.4106 17.0464 16.2157 16.9114L16.1464 16.8536L12.0196 12.7266C11.0659 13.5217 9.83879 14 8.5 14C5.46243 14 3 11.5376 3 8.5C3 5.46243 5.46243 3 8.5 3ZM8.5 4C6.01472 4 4 6.01472 4 8.5C4 10.9853 6.01472 13 8.5 13C10.9853 13 13 10.9853 13 8.5C13 6.01472 10.9853 4 8.5 4Z", fill: "currentColor" })
));
const ZoomOutIcon = replaceable("zoomOut", h(
  "svg",
  { viewBox: "0 0 20 20", fill: "none", xmlns: "http://www.w3.org/2000/svg" },
  h("path", { d: "M11 8C11.2761 8 11.5 8.22386 11.5 8.5C11.5 8.77614 11.2761 9 11 9H6C5.72386 9 5.5 8.77614 5.5 8.5C5.5 8.22386 5.72386 8 6 8H11Z", fill: "currentColor" }),
  h("path", { d: "M14 8.5C14 5.46243 11.5376 3 8.5 3C5.46243 3 3 5.46243 3 8.5C3 11.5376 5.46243 14 8.5 14C9.83879 14 11.0659 13.5217 12.0196 12.7266L16.1464 16.8536L16.2157 16.9114C16.4106 17.0464 16.68 17.0271 16.8536 16.8536C17.0488 16.6583 17.0488 16.3417 16.8536 16.1464L12.7266 12.0196C13.5217 11.0659 14 9.83879 14 8.5ZM4 8.5C4 6.01472 6.01472 4 8.5 4C10.9853 4 13 6.01472 13 8.5C13 10.9853 10.9853 13 8.5 13C6.01472 13 4 10.9853 4 8.5Z", fill: "currentColor" })
));
const ResizeSmallIcon = defineComponent({
  name: "ResizeSmall",
  render() {
    return h(
      "svg",
      { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 20 20" },
      h(
        "g",
        { fill: "none" },
        h("path", { d: "M5.5 4A1.5 1.5 0 0 0 4 5.5v1a.5.5 0 0 1-1 0v-1A2.5 2.5 0 0 1 5.5 3h1a.5.5 0 0 1 0 1h-1zM16 5.5A1.5 1.5 0 0 0 14.5 4h-1a.5.5 0 0 1 0-1h1A2.5 2.5 0 0 1 17 5.5v1a.5.5 0 0 1-1 0v-1zm0 9a1.5 1.5 0 0 1-1.5 1.5h-1a.5.5 0 0 0 0 1h1a2.5 2.5 0 0 0 2.5-2.5v-1a.5.5 0 0 0-1 0v1zm-12 0A1.5 1.5 0 0 0 5.5 16h1.25a.5.5 0 0 1 0 1H5.5A2.5 2.5 0 0 1 3 14.5v-1.25a.5.5 0 0 1 1 0v1.25zM8.5 7A1.5 1.5 0 0 0 7 8.5v3A1.5 1.5 0 0 0 8.5 13h3a1.5 1.5 0 0 0 1.5-1.5v-3A1.5 1.5 0 0 0 11.5 7h-3zM8 8.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 .5.5v3a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-3z", fill: "currentColor" })
      )
    );
  }
});
const isImageSupportNativeLazy = isBrowser && "loading" in document.createElement("img");
const resolveOptionsAndHash = (options = {}) => {
  var _a;
  const { root = null } = options;
  return {
    hash: `${options.rootMargin || "0px 0px 0px 0px"}-${Array.isArray(options.threshold) ? options.threshold.join(",") : (_a = options.threshold) !== null && _a !== void 0 ? _a : "0"}`,
    options: Object.assign(Object.assign({}, options), { root: (typeof root === "string" ? document.querySelector(root) : root) || document.documentElement })
  };
};
const observers = /* @__PURE__ */ new WeakMap();
const unobserveHandleMap = /* @__PURE__ */ new WeakMap();
const shouldStartLoadingRefMap = /* @__PURE__ */ new WeakMap();
const observeIntersection = (el, options, shouldStartLoadingRef) => {
  if (!el)
    return () => {
    };
  const resolvedOptionsAndHash = resolveOptionsAndHash(options);
  const { root } = resolvedOptionsAndHash.options;
  let rootObservers;
  const _rootObservers = observers.get(root);
  if (_rootObservers) {
    rootObservers = _rootObservers;
  } else {
    rootObservers = /* @__PURE__ */ new Map();
    observers.set(root, rootObservers);
  }
  let observer;
  let observerAndObservedElements;
  if (rootObservers.has(resolvedOptionsAndHash.hash)) {
    observerAndObservedElements = // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    rootObservers.get(resolvedOptionsAndHash.hash);
    if (!observerAndObservedElements[1].has(el)) {
      observer = observerAndObservedElements[0];
      observerAndObservedElements[1].add(el);
      observer.observe(el);
    }
  } else {
    observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const _unobserve = unobserveHandleMap.get(entry.target);
          const _shouldStartLoadingRef = shouldStartLoadingRefMap.get(entry.target);
          if (_unobserve)
            _unobserve();
          if (_shouldStartLoadingRef) {
            _shouldStartLoadingRef.value = true;
          }
        }
      });
    }, resolvedOptionsAndHash.options);
    observer.observe(el);
    observerAndObservedElements = [observer, /* @__PURE__ */ new Set([el])];
    rootObservers.set(resolvedOptionsAndHash.hash, observerAndObservedElements);
  }
  let unobservered = false;
  const unobserve = () => {
    if (unobservered)
      return;
    unobserveHandleMap.delete(el);
    shouldStartLoadingRefMap.delete(el);
    unobservered = true;
    if (observerAndObservedElements[1].has(el)) {
      observerAndObservedElements[0].unobserve(el);
      observerAndObservedElements[1].delete(el);
    }
    if (observerAndObservedElements[1].size <= 0) {
      rootObservers.delete(resolvedOptionsAndHash.hash);
    }
    if (!rootObservers.size) {
      observers.delete(root);
    }
  };
  unobserveHandleMap.set(el, unobserve);
  shouldStartLoadingRefMap.set(el, shouldStartLoadingRef);
  return unobserve;
};
const imagePreviewSharedProps = Object.assign(Object.assign({}, useTheme.props), { showToolbar: { type: Boolean, default: true }, showToolbarTooltip: Boolean });
const imageContextKey = createInjectionKey("n-image");
function self() {
  return {
    toolbarIconColor: "rgba(255, 255, 255, .9)",
    toolbarColor: "rgba(0, 0, 0, .35)",
    toolbarBoxShadow: "none",
    toolbarBorderRadius: "24px"
  };
}
const imageLight = createTheme({
  name: "Image",
  common: commonLight,
  peers: {
    Tooltip: tooltipLight
  },
  self
});
const prevIcon = h(
  "svg",
  { viewBox: "0 0 20 20", fill: "none", xmlns: "http://www.w3.org/2000/svg" },
  h("path", { d: "M6 5C5.75454 5 5.55039 5.17688 5.50806 5.41012L5.5 5.5V14.5C5.5 14.7761 5.72386 15 6 15C6.24546 15 6.44961 14.8231 6.49194 14.5899L6.5 14.5V5.5C6.5 5.22386 6.27614 5 6 5ZM13.8536 5.14645C13.68 4.97288 13.4106 4.9536 13.2157 5.08859L13.1464 5.14645L8.64645 9.64645C8.47288 9.82001 8.4536 10.0894 8.58859 10.2843L8.64645 10.3536L13.1464 14.8536C13.3417 15.0488 13.6583 15.0488 13.8536 14.8536C14.0271 14.68 14.0464 14.4106 13.9114 14.2157L13.8536 14.1464L9.70711 10L13.8536 5.85355C14.0488 5.65829 14.0488 5.34171 13.8536 5.14645Z", fill: "currentColor" })
);
const nextIcon = h(
  "svg",
  { viewBox: "0 0 20 20", fill: "none", xmlns: "http://www.w3.org/2000/svg" },
  h("path", { d: "M13.5 5C13.7455 5 13.9496 5.17688 13.9919 5.41012L14 5.5V14.5C14 14.7761 13.7761 15 13.5 15C13.2545 15 13.0504 14.8231 13.0081 14.5899L13 14.5V5.5C13 5.22386 13.2239 5 13.5 5ZM5.64645 5.14645C5.82001 4.97288 6.08944 4.9536 6.28431 5.08859L6.35355 5.14645L10.8536 9.64645C11.0271 9.82001 11.0464 10.0894 10.9114 10.2843L10.8536 10.3536L6.35355 14.8536C6.15829 15.0488 5.84171 15.0488 5.64645 14.8536C5.47288 14.68 5.4536 14.4106 5.58859 14.2157L5.64645 14.1464L9.79289 10L5.64645 5.85355C5.45118 5.65829 5.45118 5.34171 5.64645 5.14645Z", fill: "currentColor" })
);
const closeIcon = h(
  "svg",
  { viewBox: "0 0 20 20", fill: "none", xmlns: "http://www.w3.org/2000/svg" },
  h("path", { d: "M4.089 4.216l.057-.07a.5.5 0 0 1 .638-.057l.07.057L10 9.293l5.146-5.147a.5.5 0 0 1 .638-.057l.07.057a.5.5 0 0 1 .057.638l-.057.07L10.707 10l5.147 5.146a.5.5 0 0 1 .057.638l-.057.07a.5.5 0 0 1-.638.057l-.07-.057L10 10.707l-5.146 5.147a.5.5 0 0 1-.638.057l-.07-.057a.5.5 0 0 1-.057-.638l.057-.07L9.293 10L4.146 4.854a.5.5 0 0 1-.057-.638l.057-.07l-.057.07z", fill: "currentColor" })
);
const style = c([c("body >", [cB("image-container", "position: fixed;")]), cB("image-preview-container", `
 position: fixed;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 display: flex;
 `), cB("image-preview-overlay", `
 z-index: -1;
 position: absolute;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 background: rgba(0, 0, 0, .3);
 `, [fadeInTransition()]), cB("image-preview-toolbar", `
 z-index: 1;
 position: absolute;
 left: 50%;
 transform: translateX(-50%);
 border-radius: var(--n-toolbar-border-radius);
 height: 48px;
 bottom: 40px;
 padding: 0 12px;
 background: var(--n-toolbar-color);
 box-shadow: var(--n-toolbar-box-shadow);
 color: var(--n-toolbar-icon-color);
 transition: color .3s var(--n-bezier);
 display: flex;
 align-items: center;
 `, [cB("base-icon", `
 padding: 0 8px;
 font-size: 28px;
 cursor: pointer;
 `), fadeInTransition()]), cB("image-preview-wrapper", `
 position: absolute;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 display: flex;
 pointer-events: none;
 `, [fadeInScaleUpTransition()]), cB("image-preview", `
 user-select: none;
 -webkit-user-select: none;
 pointer-events: all;
 margin: auto;
 max-height: calc(100vh - 32px);
 max-width: calc(100vw - 32px);
 transition: transform .3s var(--n-bezier);
 `), cB("image", `
 display: inline-flex;
 max-height: 100%;
 max-width: 100%;
 `, [cNotM("preview-disabled", `
 cursor: pointer;
 `), c("img", `
 border-radius: inherit;
 `)])]);
const BLEEDING = 32;
const NImagePreview = defineComponent({
  name: "ImagePreview",
  props: Object.assign(Object.assign({}, imagePreviewSharedProps), { onNext: Function, onPrev: Function, clsPrefix: {
    type: String,
    required: true
  } }),
  setup(props) {
    const themeRef = useTheme("Image", "-image", style, imageLight, props, toRef(props, "clsPrefix"));
    let thumbnailEl = null;
    const previewRef = ref(null);
    const previewWrapperRef = ref(null);
    const previewSrcRef = ref(void 0);
    const showRef = ref(false);
    const displayedRef = ref(false);
    const { localeRef } = useLocale("Image");
    function syncTransformOrigin() {
      const { value: previewWrapper } = previewWrapperRef;
      if (!thumbnailEl || !previewWrapper)
        return;
      const { style: style2 } = previewWrapper;
      const tbox = thumbnailEl.getBoundingClientRect();
      const tx = tbox.left + tbox.width / 2;
      const ty = tbox.top + tbox.height / 2;
      style2.transformOrigin = `${tx}px ${ty}px`;
    }
    function handleKeydown(e) {
      var _a, _b;
      switch (e.key) {
        case " ":
          e.preventDefault();
          break;
        case "ArrowLeft":
          (_a = props.onPrev) === null || _a === void 0 ? void 0 : _a.call(props);
          break;
        case "ArrowRight":
          (_b = props.onNext) === null || _b === void 0 ? void 0 : _b.call(props);
          break;
        case "Escape":
          toggleShow();
          break;
      }
    }
    watch(showRef, (value) => {
      if (value) {
        on("keydown", document, handleKeydown);
      } else
        off("keydown", document, handleKeydown);
    });
    onBeforeUnmount(() => {
      off("keydown", document, handleKeydown);
    });
    let startX = 0;
    let startY = 0;
    let offsetX = 0;
    let offsetY = 0;
    let startOffsetX = 0;
    let startOffsetY = 0;
    let mouseDownClientX = 0;
    let mouseDownClientY = 0;
    let dragging = false;
    function handleMouseMove(e) {
      const { clientX, clientY } = e;
      offsetX = clientX - startX;
      offsetY = clientY - startY;
      beforeNextFrameOnce(derivePreviewStyle);
    }
    function getMoveStrategy(opts) {
      const { mouseUpClientX, mouseUpClientY, mouseDownClientX: mouseDownClientX2, mouseDownClientY: mouseDownClientY2 } = opts;
      const deltaHorizontal = mouseDownClientX2 - mouseUpClientX;
      const deltaVertical = mouseDownClientY2 - mouseUpClientY;
      const moveVerticalDirection = `vertical${deltaVertical > 0 ? "Top" : "Bottom"}`;
      const moveHorizontalDirection = `horizontal${deltaHorizontal > 0 ? "Left" : "Right"}`;
      return {
        moveVerticalDirection,
        moveHorizontalDirection,
        deltaHorizontal,
        deltaVertical
      };
    }
    function getDerivedOffset(moveStrategy) {
      const { value: preview } = previewRef;
      if (!preview)
        return { offsetX: 0, offsetY: 0 };
      const pbox = preview.getBoundingClientRect();
      const { moveVerticalDirection, moveHorizontalDirection, deltaHorizontal, deltaVertical } = moveStrategy || {};
      let nextOffsetX = 0;
      let nextOffsetY = 0;
      if (pbox.width <= window.innerWidth) {
        nextOffsetX = 0;
      } else if (pbox.left > 0) {
        nextOffsetX = (pbox.width - window.innerWidth) / 2;
      } else if (pbox.right < window.innerWidth) {
        nextOffsetX = -(pbox.width - window.innerWidth) / 2;
      } else if (moveHorizontalDirection === "horizontalRight") {
        nextOffsetX = Math.min((pbox.width - window.innerWidth) / 2, startOffsetX - (deltaHorizontal !== null && deltaHorizontal !== void 0 ? deltaHorizontal : 0));
      } else {
        nextOffsetX = Math.max(-((pbox.width - window.innerWidth) / 2), startOffsetX - (deltaHorizontal !== null && deltaHorizontal !== void 0 ? deltaHorizontal : 0));
      }
      if (pbox.height <= window.innerHeight) {
        nextOffsetY = 0;
      } else if (pbox.top > 0) {
        nextOffsetY = (pbox.height - window.innerHeight) / 2;
      } else if (pbox.bottom < window.innerHeight) {
        nextOffsetY = -(pbox.height - window.innerHeight) / 2;
      } else if (moveVerticalDirection === "verticalBottom") {
        nextOffsetY = Math.min((pbox.height - window.innerHeight) / 2, startOffsetY - (deltaVertical !== null && deltaVertical !== void 0 ? deltaVertical : 0));
      } else {
        nextOffsetY = Math.max(-((pbox.height - window.innerHeight) / 2), startOffsetY - (deltaVertical !== null && deltaVertical !== void 0 ? deltaVertical : 0));
      }
      return {
        offsetX: nextOffsetX,
        offsetY: nextOffsetY
      };
    }
    function handleMouseUp(e) {
      off("mousemove", document, handleMouseMove);
      off("mouseup", document, handleMouseUp);
      const { clientX: mouseUpClientX, clientY: mouseUpClientY } = e;
      dragging = false;
      const moveStrategy = getMoveStrategy({
        mouseUpClientX,
        mouseUpClientY,
        mouseDownClientX,
        mouseDownClientY
      });
      const offset = getDerivedOffset(moveStrategy);
      offsetX = offset.offsetX;
      offsetY = offset.offsetY;
      derivePreviewStyle();
    }
    const imageContext = inject(imageContextKey, null);
    function handlePreviewMousedown(e) {
      var _a, _b;
      (_b = (_a = imageContext === null || imageContext === void 0 ? void 0 : imageContext.previewedImgPropsRef.value) === null || _a === void 0 ? void 0 : _a.onMousedown) === null || _b === void 0 ? void 0 : _b.call(_a, e);
      if (e.button !== 0)
        return;
      const { clientX, clientY } = e;
      dragging = true;
      startX = clientX - offsetX;
      startY = clientY - offsetY;
      startOffsetX = offsetX;
      startOffsetY = offsetY;
      mouseDownClientX = clientX;
      mouseDownClientY = clientY;
      derivePreviewStyle();
      on("mousemove", document, handleMouseMove);
      on("mouseup", document, handleMouseUp);
    }
    function handlePreviewDblclick(e) {
      var _a, _b;
      (_b = (_a = imageContext === null || imageContext === void 0 ? void 0 : imageContext.previewedImgPropsRef.value) === null || _a === void 0 ? void 0 : _a.onDblclick) === null || _b === void 0 ? void 0 : _b.call(_a, e);
      const originalImageSizeScale = getOrignalImageSizeScale();
      scale = scale === originalImageSizeScale ? 1 : originalImageSizeScale;
      derivePreviewStyle();
    }
    const scaleRadix = 1.5;
    let scaleExp = 0;
    let scale = 1;
    let rotate = 0;
    function resetScale() {
      scale = 1;
      scaleExp = 0;
    }
    function handleSwitchPrev() {
      var _a;
      resetScale();
      rotate = 0;
      (_a = props.onPrev) === null || _a === void 0 ? void 0 : _a.call(props);
    }
    function handleSwitchNext() {
      var _a;
      resetScale();
      rotate = 0;
      (_a = props.onNext) === null || _a === void 0 ? void 0 : _a.call(props);
    }
    function rotateCounterclockwise() {
      rotate -= 90;
      derivePreviewStyle();
    }
    function rotateClockwise() {
      rotate += 90;
      derivePreviewStyle();
    }
    function getMaxScale() {
      const { value: preview } = previewRef;
      if (!preview)
        return 1;
      const { innerWidth, innerHeight } = window;
      const heightMaxScale = Math.max(1, preview.naturalHeight / (innerHeight - BLEEDING));
      const widthMaxScale = Math.max(1, preview.naturalWidth / (innerWidth - BLEEDING));
      return Math.max(3, heightMaxScale * 2, widthMaxScale * 2);
    }
    function getOrignalImageSizeScale() {
      const { value: preview } = previewRef;
      if (!preview)
        return 1;
      const { innerWidth, innerHeight } = window;
      const heightScale = preview.naturalHeight / (innerHeight - BLEEDING);
      const widthScale = preview.naturalWidth / (innerWidth - BLEEDING);
      if (heightScale < 1 && widthScale < 1) {
        return 1;
      }
      return Math.max(heightScale, widthScale);
    }
    function zoomIn() {
      const maxScale = getMaxScale();
      if (scale < maxScale) {
        scaleExp += 1;
        scale = Math.min(maxScale, Math.pow(scaleRadix, scaleExp));
        derivePreviewStyle();
      }
    }
    function zoomOut() {
      if (scale > 0.5) {
        const originalScale = scale;
        scaleExp -= 1;
        scale = Math.max(0.5, Math.pow(scaleRadix, scaleExp));
        const diff = originalScale - scale;
        derivePreviewStyle(false);
        const offset = getDerivedOffset();
        scale += diff;
        derivePreviewStyle(false);
        scale -= diff;
        offsetX = offset.offsetX;
        offsetY = offset.offsetY;
        derivePreviewStyle();
      }
    }
    function derivePreviewStyle(transition = true) {
      var _a;
      const { value: preview } = previewRef;
      if (!preview)
        return;
      const { style: style2 } = preview;
      const controlledStyle = normalizeStyle((_a = imageContext === null || imageContext === void 0 ? void 0 : imageContext.previewedImgPropsRef.value) === null || _a === void 0 ? void 0 : _a.style);
      let controlledStyleString = "";
      if (typeof controlledStyle === "string") {
        controlledStyleString = controlledStyle + ";";
      } else {
        for (const key in controlledStyle) {
          controlledStyleString += `${kebabCase$1(key)}: ${controlledStyle[key]};`;
        }
      }
      const transformStyle = `transform-origin: center; transform: translateX(${offsetX}px) translateY(${offsetY}px) rotate(${rotate}deg) scale(${scale});`;
      if (dragging) {
        style2.cssText = controlledStyleString + "cursor: grabbing; transition: none;" + transformStyle;
      } else {
        style2.cssText = controlledStyleString + "cursor: grab;" + transformStyle + (transition ? "" : "transition: none;");
      }
      if (!transition) {
        void preview.offsetHeight;
      }
    }
    function toggleShow() {
      showRef.value = !showRef.value;
      displayedRef.value = true;
    }
    function resizeToOrignalImageSize() {
      scale = getOrignalImageSizeScale();
      scaleExp = Math.ceil(Math.log(scale) / Math.log(scaleRadix));
      offsetX = 0;
      offsetY = 0;
      derivePreviewStyle();
    }
    const exposedMethods = {
      setPreviewSrc: (src) => {
        previewSrcRef.value = src;
      },
      setThumbnailEl: (el) => {
        thumbnailEl = el;
      },
      toggleShow
    };
    function withTooltip(node, tooltipKey) {
      if (props.showToolbarTooltip) {
        const { value: theme } = themeRef;
        return h(NTooltip, { to: false, theme: theme.peers.Tooltip, themeOverrides: theme.peerOverrides.Tooltip, keepAliveOnHover: false }, {
          default: () => {
            return localeRef.value[tooltipKey];
          },
          trigger: () => node
        });
      } else {
        return node;
      }
    }
    const cssVarsRef = computed(() => {
      const { common: { cubicBezierEaseInOut }, self: { toolbarIconColor, toolbarBorderRadius, toolbarBoxShadow, toolbarColor } } = themeRef.value;
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-toolbar-icon-color": toolbarIconColor,
        "--n-toolbar-color": toolbarColor,
        "--n-toolbar-border-radius": toolbarBorderRadius,
        "--n-toolbar-box-shadow": toolbarBoxShadow
      };
    });
    const { inlineThemeDisabled } = useConfig();
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("image-preview", void 0, cssVarsRef, props) : void 0;
    return Object.assign({
      previewRef,
      previewWrapperRef,
      previewSrc: previewSrcRef,
      show: showRef,
      appear: isMounted(),
      displayed: displayedRef,
      previewedImgProps: imageContext === null || imageContext === void 0 ? void 0 : imageContext.previewedImgPropsRef,
      handleWheel(e) {
        e.preventDefault();
      },
      handlePreviewMousedown,
      handlePreviewDblclick,
      syncTransformOrigin,
      handleAfterLeave: () => {
        resetScale();
        rotate = 0;
        displayedRef.value = false;
      },
      handleDragStart: (e) => {
        var _a, _b;
        (_b = (_a = imageContext === null || imageContext === void 0 ? void 0 : imageContext.previewedImgPropsRef.value) === null || _a === void 0 ? void 0 : _a.onDragstart) === null || _b === void 0 ? void 0 : _b.call(_a, e);
        e.preventDefault();
      },
      zoomIn,
      zoomOut,
      rotateCounterclockwise,
      rotateClockwise,
      handleSwitchPrev,
      handleSwitchNext,
      withTooltip,
      resizeToOrignalImageSize,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    }, exposedMethods);
  },
  render() {
    var _a, _b;
    const { clsPrefix } = this;
    return h(
      Fragment,
      null,
      (_b = (_a = this.$slots).default) === null || _b === void 0 ? void 0 : _b.call(_a),
      h(LazyTeleport, { show: this.show }, {
        default: () => {
          var _a2;
          if (!(this.show || this.displayed)) {
            return null;
          }
          (_a2 = this.onRender) === null || _a2 === void 0 ? void 0 : _a2.call(this);
          return withDirectives(h(
            "div",
            { class: [
              `${clsPrefix}-image-preview-container`,
              this.themeClass
            ], style: this.cssVars, onWheel: this.handleWheel },
            h(Transition, { name: "fade-in-transition", appear: this.appear }, {
              default: () => this.show ? h("div", { class: `${clsPrefix}-image-preview-overlay`, onClick: this.toggleShow }) : null
            }),
            this.showToolbar ? h(Transition, { name: "fade-in-transition", appear: this.appear }, {
              default: () => {
                if (!this.show)
                  return null;
                const { withTooltip } = this;
                return h(
                  "div",
                  { class: `${clsPrefix}-image-preview-toolbar` },
                  this.onPrev ? h(
                    Fragment,
                    null,
                    withTooltip(h(NBaseIcon, { clsPrefix, onClick: this.handleSwitchPrev }, { default: () => prevIcon }), "tipPrevious"),
                    withTooltip(h(NBaseIcon, { clsPrefix, onClick: this.handleSwitchNext }, { default: () => nextIcon }), "tipNext")
                  ) : null,
                  withTooltip(h(NBaseIcon, { clsPrefix, onClick: this.rotateCounterclockwise }, {
                    default: () => h(RotateCounterclockwiseIcon, null)
                  }), "tipCounterclockwise"),
                  withTooltip(h(NBaseIcon, { clsPrefix, onClick: this.rotateClockwise }, {
                    default: () => h(RotateClockwiseIcon, null)
                  }), "tipClockwise"),
                  withTooltip(h(NBaseIcon, { clsPrefix, onClick: this.resizeToOrignalImageSize }, {
                    default: () => {
                      return h(ResizeSmallIcon, null);
                    }
                  }), "tipOriginalSize"),
                  withTooltip(h(NBaseIcon, { clsPrefix, onClick: this.zoomOut }, { default: () => h(ZoomOutIcon, null) }), "tipZoomOut"),
                  withTooltip(h(NBaseIcon, { clsPrefix, onClick: this.zoomIn }, { default: () => h(ZoomInIcon, null) }), "tipZoomIn"),
                  withTooltip(h(NBaseIcon, { clsPrefix, onClick: this.toggleShow }, { default: () => closeIcon }), "tipClose")
                );
              }
            }) : null,
            h(Transition, {
              name: "fade-in-scale-up-transition",
              onAfterLeave: this.handleAfterLeave,
              appear: this.appear,
              // BUG:
              // onEnter will be called twice, I don't know why
              // Maybe it is a bug of vue
              onEnter: this.syncTransformOrigin,
              onBeforeLeave: this.syncTransformOrigin
            }, {
              default: () => {
                const { previewedImgProps = {} } = this;
                return withDirectives(h(
                  "div",
                  { class: `${clsPrefix}-image-preview-wrapper`, ref: "previewWrapperRef" },
                  h("img", Object.assign({}, previewedImgProps, { draggable: false, onMousedown: this.handlePreviewMousedown, onDblclick: this.handlePreviewDblclick, class: [
                    `${clsPrefix}-image-preview`,
                    previewedImgProps.class
                  ], key: this.previewSrc, src: this.previewSrc, ref: "previewRef", onDragstart: this.handleDragStart }))
                ), [[vShow, this.show]]);
              }
            })
          ), [[zindexable, { enabled: this.show }]]);
        }
      })
    );
  }
});
const imageGroupInjectionKey = createInjectionKey("n-image-group");
const imageGroupProps = imagePreviewSharedProps;
defineComponent({
  name: "ImageGroup",
  props: imageGroupProps,
  setup(props) {
    let currentSrc;
    const { mergedClsPrefixRef } = useConfig(props);
    const groupId = `c${createId()}`;
    const vm = getCurrentInstance();
    const setPreviewSrc = (src) => {
      var _a;
      currentSrc = src;
      (_a = previewInstRef.value) === null || _a === void 0 ? void 0 : _a.setPreviewSrc(src);
    };
    function go(step) {
      if (!(vm === null || vm === void 0 ? void 0 : vm.proxy))
        return;
      const container = vm.proxy.$el.parentElement;
      const imgs = container.querySelectorAll(`[data-group-id=${groupId}]:not([data-error=true])`);
      if (!imgs.length)
        return;
      const index = Array.from(imgs).findIndex((img) => img.dataset.previewSrc === currentSrc);
      if (~index) {
        setPreviewSrc(imgs[(index + step + imgs.length) % imgs.length].dataset.previewSrc);
      } else {
        setPreviewSrc(imgs[0].dataset.previewSrc);
      }
    }
    provide(imageGroupInjectionKey, {
      mergedClsPrefixRef,
      setPreviewSrc,
      setThumbnailEl: (el) => {
        var _a;
        (_a = previewInstRef.value) === null || _a === void 0 ? void 0 : _a.setThumbnailEl(el);
      },
      toggleShow: () => {
        var _a;
        (_a = previewInstRef.value) === null || _a === void 0 ? void 0 : _a.toggleShow();
      },
      groupId
    });
    const previewInstRef = ref(null);
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      previewInstRef,
      next: () => go(1),
      prev: () => go(-1)
    };
  },
  render() {
    return h(NImagePreview, { theme: this.theme, themeOverrides: this.themeOverrides, clsPrefix: this.mergedClsPrefix, ref: "previewInstRef", onPrev: this.prev, onNext: this.next, showToolbar: this.showToolbar, showToolbarTooltip: this.showToolbarTooltip }, this.$slots);
  }
});
const imageProps = Object.assign({ alt: String, height: [String, Number], imgProps: Object, previewedImgProps: Object, lazy: Boolean, intersectionObserverOptions: Object, objectFit: {
  type: String,
  default: "fill"
}, previewSrc: String, fallbackSrc: String, width: [String, Number], src: String, previewDisabled: Boolean, loadDescription: String, onError: Function, onLoad: Function }, imagePreviewSharedProps);
const NImage = defineComponent({
  name: "Image",
  props: imageProps,
  inheritAttrs: false,
  setup(props) {
    const imageRef = ref(null);
    const showErrorRef = ref(false);
    const previewInstRef = ref(null);
    const imageGroupHandle = inject(imageGroupInjectionKey, null);
    const { mergedClsPrefixRef } = imageGroupHandle || useConfig(props);
    const exposedMethods = {
      click: () => {
        if (props.previewDisabled || showErrorRef.value)
          return;
        const mergedPreviewSrc = props.previewSrc || props.src;
        if (imageGroupHandle) {
          imageGroupHandle.setPreviewSrc(mergedPreviewSrc);
          imageGroupHandle.setThumbnailEl(imageRef.value);
          imageGroupHandle.toggleShow();
          return;
        }
        const { value: previewInst } = previewInstRef;
        if (!previewInst)
          return;
        previewInst.setPreviewSrc(mergedPreviewSrc);
        previewInst.setThumbnailEl(imageRef.value);
        previewInst.toggleShow();
      }
    };
    const shouldStartLoadingRef = ref(!props.lazy);
    onMounted(() => {
      var _a;
      (_a = imageRef.value) === null || _a === void 0 ? void 0 : _a.setAttribute("data-group-id", (imageGroupHandle === null || imageGroupHandle === void 0 ? void 0 : imageGroupHandle.groupId) || "");
    });
    onMounted(() => {
      if (isImageSupportNativeLazy) {
        return;
      }
      let unobserve;
      const stopWatchHandle = watchEffect(() => {
        unobserve === null || unobserve === void 0 ? void 0 : unobserve();
        unobserve = void 0;
        if (props.lazy) {
          unobserve = observeIntersection(imageRef.value, props.intersectionObserverOptions, shouldStartLoadingRef);
        }
      });
      onBeforeUnmount(() => {
        stopWatchHandle();
        unobserve === null || unobserve === void 0 ? void 0 : unobserve();
      });
    });
    watchEffect(() => {
      var _a;
      void props.src;
      void ((_a = props.imgProps) === null || _a === void 0 ? void 0 : _a.src);
      showErrorRef.value = false;
    });
    const loadedRef = ref(false);
    provide(imageContextKey, {
      previewedImgPropsRef: toRef(props, "previewedImgProps")
    });
    return Object.assign({
      mergedClsPrefix: mergedClsPrefixRef,
      groupId: imageGroupHandle === null || imageGroupHandle === void 0 ? void 0 : imageGroupHandle.groupId,
      previewInstRef,
      imageRef,
      showError: showErrorRef,
      shouldStartLoading: shouldStartLoadingRef,
      loaded: loadedRef,
      mergedOnClick: (e) => {
        var _a, _b;
        exposedMethods.click();
        (_b = (_a = props.imgProps) === null || _a === void 0 ? void 0 : _a.onClick) === null || _b === void 0 ? void 0 : _b.call(_a, e);
      },
      mergedOnError: (e) => {
        if (!shouldStartLoadingRef.value)
          return;
        showErrorRef.value = true;
        const { onError, imgProps: { onError: imgPropsOnError } = {} } = props;
        onError === null || onError === void 0 ? void 0 : onError(e);
        imgPropsOnError === null || imgPropsOnError === void 0 ? void 0 : imgPropsOnError(e);
      },
      mergedOnLoad: (e) => {
        const { onLoad, imgProps: { onLoad: imgPropsOnLoad } = {} } = props;
        onLoad === null || onLoad === void 0 ? void 0 : onLoad(e);
        imgPropsOnLoad === null || imgPropsOnLoad === void 0 ? void 0 : imgPropsOnLoad(e);
        loadedRef.value = true;
      }
    }, exposedMethods);
  },
  render() {
    var _a, _b;
    const { mergedClsPrefix, imgProps = {}, loaded, $attrs, lazy } = this;
    const placeholderNode = (_b = (_a = this.$slots).placeholder) === null || _b === void 0 ? void 0 : _b.call(_a);
    const loadSrc = this.src || imgProps.src || "";
    const imgNode = h("img", Object.assign(Object.assign({}, imgProps), {
      ref: "imageRef",
      width: this.width || imgProps.width,
      height: this.height || imgProps.height,
      src: isImageSupportNativeLazy ? loadSrc : this.showError ? this.fallbackSrc : this.shouldStartLoading ? loadSrc : void 0,
      alt: this.alt || imgProps.alt,
      "aria-label": this.alt || imgProps.alt,
      onClick: this.mergedOnClick,
      onError: this.mergedOnError,
      onLoad: this.mergedOnLoad,
      // If interseciton observer options is set, do not use native lazy
      loading: isImageSupportNativeLazy && lazy && !this.intersectionObserverOptions ? "lazy" : "eager",
      style: [
        imgProps.style || "",
        placeholderNode && !loaded ? { height: "0", width: "0", visibility: "hidden" } : "",
        { objectFit: this.objectFit }
      ],
      "data-error": this.showError,
      "data-preview-src": this.previewSrc || this.src
    }));
    return h(
      "div",
      Object.assign({}, $attrs, { role: "none", class: [
        $attrs.class,
        `${mergedClsPrefix}-image`,
        (this.previewDisabled || this.showError) && `${mergedClsPrefix}-image--preview-disabled`
      ] }),
      this.groupId ? imgNode : h(NImagePreview, { theme: this.theme, themeOverrides: this.themeOverrides, clsPrefix: mergedClsPrefix, ref: "previewInstRef", showToolbar: this.showToolbar, showToolbarTooltip: this.showToolbarTooltip }, {
        default: () => imgNode
      }),
      !loaded && placeholderNode
    );
  }
});
export {
  NImage as N
};
