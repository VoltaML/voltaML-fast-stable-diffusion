import { ref, watch } from "vue";
import { serverUrl } from "./env";

export class BurnerClock {
  private isChanging = ref(false);
  private timer: ReturnType<typeof setTimeout> | null = null;
  private timeoutDuration: number;

  constructor(
    private readonly observed_value: any,
    private settings: any,
    private readonly callback: () => void,
    private readonly timerOverrride: number = 0,
    private readonly sendInterrupt: boolean = true
  ) {
    this.timeoutDuration =
      this.timerOverrride !== 0
        ? this.timerOverrride
        : this.settings.data.settings.frontend.on_change_timer;
    watch(this.observed_value, () => {
      this.handleChange();
    });
  }

  private handleChange() {
    if (!this.isChanging.value) {
      this.startTimer();
    } else {
      this.resetTimer();
    }
  }

  private startTimer() {
    if (this.timeoutDuration > 0) {
      this.isChanging.value = true;
      this.timer = setTimeout(() => {
        if (this.sendInterrupt) {
          fetch(`${serverUrl}/api/general/interrupt`, {
            method: "POST",
          })
            .then((res) => {
              if (res.status === 200) {
                this.callback();
                this.isChanging.value = false;
              }
            })
            .catch((err) => {
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

  private resetTimer() {
    if (this.timer) {
      clearTimeout(this.timer);
    }

    this.timer = setTimeout(() => {
      fetch(`${serverUrl}/api/general/interrupt`, {
        method: "POST",
      })
        .then((res) => {
          if (res.status === 200) {
            this.callback();
            this.isChanging.value = false;
          }
        })
        .catch((err) => {
          this.isChanging.value = false;
          console.error(err);
        });
    }, this.timeoutDuration);
  }

  public cleanup() {
    if (this.timer) {
      clearTimeout(this.timer);
    }
  }
}
