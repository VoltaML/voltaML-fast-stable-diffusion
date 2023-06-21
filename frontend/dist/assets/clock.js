var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
import { N as NDescriptionsItem, a as NDescriptions } from "./SendOutputTo.vue_vue_type_script_setup_true_lang.js";
import { d as defineComponent, e as openBlock, x as createBlock, w as withCtx, g as createVNode, h as unref, m as createTextVNode, t as toDisplayString, i as NCard, y as createCommentVNode, J as watch, E as ref, s as serverUrl } from "./index.js";
const _sfc_main = /* @__PURE__ */ defineComponent({
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
class BurnerClock {
  constructor(observed_value, settings, callback) {
    __publicField(this, "isChanging", ref(false));
    __publicField(this, "timer", null);
    __publicField(this, "timeoutDuration");
    this.observed_value = observed_value;
    this.settings = settings;
    this.callback = callback;
    this.timeoutDuration = this.settings.data.settings.frontend.on_change_timer;
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
  _sfc_main as _
};
