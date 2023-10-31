var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
import { N as NDescriptionsItem, a as NDescriptions } from "./DescriptionsItem.js";
import { d as defineComponent, o as openBlock, r as createBlock, w as withCtx, b as createVNode, e as unref, n as createTextVNode, y as toDisplayString, j as NCard, q as createCommentVNode, u as useSettings, c as createElementBlock, m as NTooltip, l as createBaseVNode, F as Fragment, J as watch, D as ref, s as serverUrl } from "./index.js";
import { N as NSlider } from "./Switch.js";
import { N as NInputNumber } from "./InputNumber.js";
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "OutputStats",
  props: {
    genData: {
      type: Object,
      required: true
    }
  },
  setup(__props) {
    return (_ctx, _cache) => {
      return __props.genData.time_taken || __props.genData.seed ? (openBlock(), createBlock(unref(NCard), {
        key: 0,
        title: "Stats"
      }, {
        default: withCtx(() => [
          createVNode(unref(NDescriptions), null, {
            default: withCtx(() => [
              createVNode(unref(NDescriptionsItem), { label: "Total Time" }, {
                default: withCtx(() => [
                  createTextVNode(toDisplayString(__props.genData.time_taken) + "s ", 1)
                ]),
                _: 1
              }),
              createVNode(unref(NDescriptionsItem), { label: "Seed" }, {
                default: withCtx(() => [
                  createTextVNode(toDisplayString(__props.genData.seed), 1)
                ]),
                _: 1
              })
            ]),
            _: 1
          })
        ]),
        _: 1
      })) : createCommentVNode("", true);
    };
  }
});
const _hoisted_1$1 = {
  key: 0,
  class: "flex-container"
};
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1);
const _hoisted_3$1 = {
  key: 1,
  class: "flex-container"
};
const _hoisted_4$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1);
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "BatchSizeInput",
  props: {
    batchSizeObject: {
      type: Object,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    return (_ctx, _cache) => {
      return unref(settings).data.settings.aitDim.batch_size ? (openBlock(), createElementBlock("div", _hoisted_1$1, [
        createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
          trigger: withCtx(() => [
            _hoisted_2$1
          ]),
          default: withCtx(() => [
            createTextVNode(" Number of images to generate in paralel. ")
          ]),
          _: 1
        }),
        createVNode(unref(NSlider), {
          value: props.batchSizeObject.batch_size,
          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => props.batchSizeObject.batch_size = $event),
          min: unref(settings).data.settings.aitDim.batch_size[0],
          max: unref(settings).data.settings.aitDim.batch_size[1],
          style: { "margin-right": "12px" }
        }, null, 8, ["value", "min", "max"]),
        createVNode(unref(NInputNumber), {
          value: props.batchSizeObject.batch_size,
          "onUpdate:value": _cache[1] || (_cache[1] = ($event) => props.batchSizeObject.batch_size = $event),
          size: "small",
          min: unref(settings).data.settings.aitDim.batch_size[0],
          max: unref(settings).data.settings.aitDim.batch_size[1],
          style: { "min-width": "96px", "width": "96px" }
        }, null, 8, ["value", "min", "max"])
      ])) : (openBlock(), createElementBlock("div", _hoisted_3$1, [
        createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
          trigger: withCtx(() => [
            _hoisted_4$1
          ]),
          default: withCtx(() => [
            createTextVNode(" Number of images to generate in paralel. ")
          ]),
          _: 1
        }),
        createVNode(unref(NSlider), {
          value: props.batchSizeObject.batch_size,
          "onUpdate:value": _cache[2] || (_cache[2] = ($event) => props.batchSizeObject.batch_size = $event),
          min: 1,
          max: 9,
          style: { "margin-right": "12px" }
        }, null, 8, ["value"]),
        createVNode(unref(NInputNumber), {
          value: props.batchSizeObject.batch_size,
          "onUpdate:value": _cache[3] || (_cache[3] = ($event) => props.batchSizeObject.batch_size = $event),
          size: "small",
          style: { "min-width": "96px", "width": "96px" }
        }, null, 8, ["value"])
      ]));
    };
  }
});
const _hoisted_1 = {
  key: 0,
  class: "flex-container"
};
const _hoisted_2 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1);
const _hoisted_3 = {
  key: 1,
  class: "flex-container"
};
const _hoisted_4 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1);
const _hoisted_5 = {
  key: 2,
  class: "flex-container"
};
const _hoisted_6 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1);
const _hoisted_7 = {
  key: 3,
  class: "flex-container"
};
const _hoisted_8 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "DimensionsInput",
  props: {
    dimensionsObject: {
      type: Object,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const settings = useSettings();
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        unref(settings).data.settings.aitDim.width ? (openBlock(), createElementBlock("div", _hoisted_1, [
          _hoisted_2,
          createVNode(unref(NSlider), {
            value: props.dimensionsObject.width,
            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => props.dimensionsObject.width = $event),
            min: unref(settings).data.settings.aitDim.width[0],
            max: unref(settings).data.settings.aitDim.width[1],
            step: 64,
            style: { "margin-right": "12px" }
          }, null, 8, ["value", "min", "max"]),
          createVNode(unref(NInputNumber), {
            value: props.dimensionsObject.width,
            "onUpdate:value": _cache[1] || (_cache[1] = ($event) => props.dimensionsObject.width = $event),
            size: "small",
            style: { "min-width": "96px", "width": "96px" },
            min: unref(settings).data.settings.aitDim.width[0],
            max: unref(settings).data.settings.aitDim.width[1],
            step: 64
          }, null, 8, ["value", "min", "max"])
        ])) : (openBlock(), createElementBlock("div", _hoisted_3, [
          _hoisted_4,
          createVNode(unref(NSlider), {
            value: props.dimensionsObject.width,
            "onUpdate:value": _cache[2] || (_cache[2] = ($event) => props.dimensionsObject.width = $event),
            min: 128,
            max: 2048,
            step: 1,
            style: { "margin-right": "12px" }
          }, null, 8, ["value"]),
          createVNode(unref(NInputNumber), {
            value: props.dimensionsObject.width,
            "onUpdate:value": _cache[3] || (_cache[3] = ($event) => props.dimensionsObject.width = $event),
            size: "small",
            style: { "min-width": "96px", "width": "96px" },
            step: 1
          }, null, 8, ["value"])
        ])),
        unref(settings).data.settings.aitDim.height ? (openBlock(), createElementBlock("div", _hoisted_5, [
          _hoisted_6,
          createVNode(unref(NSlider), {
            value: props.dimensionsObject.height,
            "onUpdate:value": _cache[4] || (_cache[4] = ($event) => props.dimensionsObject.height = $event),
            min: unref(settings).data.settings.aitDim.height[0],
            max: unref(settings).data.settings.aitDim.height[1],
            step: 64,
            style: { "margin-right": "12px" }
          }, null, 8, ["value", "min", "max"]),
          createVNode(unref(NInputNumber), {
            value: props.dimensionsObject.height,
            "onUpdate:value": _cache[5] || (_cache[5] = ($event) => props.dimensionsObject.height = $event),
            size: "small",
            style: { "min-width": "96px", "width": "96px" },
            min: unref(settings).data.settings.aitDim.height[0],
            max: unref(settings).data.settings.aitDim.height[1],
            step: 64
          }, null, 8, ["value", "min", "max"])
        ])) : (openBlock(), createElementBlock("div", _hoisted_7, [
          _hoisted_8,
          createVNode(unref(NSlider), {
            value: props.dimensionsObject.height,
            "onUpdate:value": _cache[6] || (_cache[6] = ($event) => props.dimensionsObject.height = $event),
            min: 128,
            max: 2048,
            step: 1,
            style: { "margin-right": "12px" }
          }, null, 8, ["value"]),
          createVNode(unref(NInputNumber), {
            value: props.dimensionsObject.height,
            "onUpdate:value": _cache[7] || (_cache[7] = ($event) => props.dimensionsObject.height = $event),
            size: "small",
            style: { "min-width": "96px", "width": "96px" },
            step: 1
          }, null, 8, ["value"])
        ]))
      ], 64);
    };
  }
});
class BurnerClock {
  constructor(observed_value, settings, callback, timerOverrride = 0, sendInterrupt = true) {
    __publicField(this, "isChanging", ref(false));
    __publicField(this, "timer", null);
    __publicField(this, "timeoutDuration");
    this.observed_value = observed_value;
    this.settings = settings;
    this.callback = callback;
    this.timerOverrride = timerOverrride;
    this.sendInterrupt = sendInterrupt;
    this.timeoutDuration = this.timerOverrride !== 0 ? this.timerOverrride : this.settings.data.settings.frontend.on_change_timer;
    watch(this.observed_value, () => {
      this.handleChange();
    });
  }
  handleChange() {
    if (!this.isChanging.value) {
      this.startTimer();
    } else {
      this.resetTimer();
    }
  }
  startTimer() {
    if (this.timeoutDuration > 0) {
      this.isChanging.value = true;
      this.timer = setTimeout(() => {
        if (this.sendInterrupt) {
          fetch(`${serverUrl}/api/general/interrupt`, {
            method: "POST"
          }).then((res) => {
            if (res.status === 200) {
              this.callback();
              this.isChanging.value = false;
            }
          }).catch((err) => {
            this.isChanging.value = false;
            console.error(err);
          });
        } else {
          this.callback();
          this.isChanging.value = false;
        }
      }, this.timeoutDuration);
    }
  }
  resetTimer() {
    if (this.timer) {
      clearTimeout(this.timer);
    }
    this.timer = setTimeout(() => {
      fetch(`${serverUrl}/api/general/interrupt`, {
        method: "POST"
      }).then((res) => {
        if (res.status === 200) {
          this.callback();
          this.isChanging.value = false;
        }
      }).catch((err) => {
        this.isChanging.value = false;
        console.error(err);
      });
    }, this.timeoutDuration);
  }
  cleanup() {
    if (this.timer) {
      clearTimeout(this.timer);
    }
  }
}
export {
  BurnerClock as B,
  _sfc_main as _,
  _sfc_main$1 as a,
  _sfc_main$2 as b
};
