var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
import { ad as watch, E as ref, s as serverUrl } from "./index.js";
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
  BurnerClock as B
};
