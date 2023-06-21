import { Y as cB, $ as cM, X as c, Z as cE, a0 as iconSwitchTransition, aq as cNotM, d as defineComponent, Q as useConfig, a7 as useRtl, a5 as useTheme, T as provide, D as h, aC as flatten, aD as getSlot, V as createInjectionKey, bb as stepsLight, a3 as inject, aZ as throwError, c as computed, a9 as useThemeClass, aB as resolveWrappedSlot, at as resolveSlot, ab as NIconSwitchTransition, a8 as createKey, W as call, au as NBaseIcon, bc as FinishedIcon, bd as ErrorIcon, b as useMessage, u as useState, E as ref, e as openBlock, f as createElementBlock, g as createVNode, w as withCtx, h as unref, j as NSpace, i as NCard, n as createBaseVNode, v as NSlider, r as NSelect, F as NButton, m as createTextVNode, b9 as NModal, s as serverUrl, a as useSettings, x as createBlock, H as NTabPane, I as NTabs } from "./index.js";
import { N as NInputNumber } from "./InputNumber.js";
import { N as NSwitch } from "./Switch.js";
const style = cB("steps", `
 width: 100%;
 display: flex;
`, [cB("step", `
 position: relative;
 display: flex;
 flex: 1;
 `, [cM("disabled", "cursor: not-allowed"), cM("clickable", `
 cursor: pointer;
 `), c("&:last-child", [cB("step-splitor", "display: none;")])]), cB("step-splitor", `
 background-color: var(--n-splitor-color);
 margin-top: calc(var(--n-step-header-font-size) / 2);
 height: 1px;
 flex: 1;
 align-self: flex-start;
 margin-left: 12px;
 margin-right: 12px;
 transition:
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 `), cB("step-content", "flex: 1;", [cB("step-content-header", `
 color: var(--n-header-text-color);
 margin-top: calc(var(--n-indicator-size) / 2 - var(--n-step-header-font-size) / 2);
 line-height: var(--n-step-header-font-size);
 font-size: var(--n-step-header-font-size);
 position: relative;
 display: flex;
 font-weight: var(--n-step-header-font-weight);
 margin-left: 9px;
 transition:
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 `, [cE("title", `
 white-space: nowrap;
 flex: 0;
 `)]), cE("description", `
 color: var(--n-description-text-color);
 margin-top: 12px;
 margin-left: 9px;
 transition:
 color .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 `)]), cB("step-indicator", `
 background-color: var(--n-indicator-color);
 box-shadow: 0 0 0 1px var(--n-indicator-border-color);
 height: var(--n-indicator-size);
 width: var(--n-indicator-size);
 border-radius: 50%;
 display: flex;
 align-items: center;
 justify-content: center;
 transition:
 background-color .3s var(--n-bezier),
 box-shadow .3s var(--n-bezier);
 `, [cB("step-indicator-slot", `
 position: relative;
 width: var(--n-indicator-icon-size);
 height: var(--n-indicator-icon-size);
 font-size: var(--n-indicator-icon-size);
 line-height: var(--n-indicator-icon-size);
 `, [cE("index", `
 display: inline-block;
 text-align: center;
 position: absolute;
 left: 0;
 top: 0;
 font-size: var(--n-indicator-index-font-size);
 width: var(--n-indicator-icon-size);
 height: var(--n-indicator-icon-size);
 line-height: var(--n-indicator-icon-size);
 color: var(--n-indicator-text-color);
 transition: color .3s var(--n-bezier);
 `, [iconSwitchTransition()]), cB("icon", `
 color: var(--n-indicator-text-color);
 transition: color .3s var(--n-bezier);
 `, [iconSwitchTransition()]), cB("base-icon", `
 color: var(--n-indicator-text-color);
 transition: color .3s var(--n-bezier);
 `, [iconSwitchTransition()])])]), cM("vertical", "flex-direction: column;", [cNotM("show-description", [c(">", [cB("step", "padding-bottom: 8px;")])]), c(">", [cB("step", "margin-bottom: 16px;", [c("&:last-child", "margin-bottom: 0;"), c(">", [cB("step-indicator", [c(">", [cB("step-splitor", `
 position: absolute;
 bottom: -8px;
 width: 1px;
 margin: 0 !important;
 left: calc(var(--n-indicator-size) / 2);
 height: calc(100% - var(--n-indicator-size));
 `)])]), cB("step-content", [cE("description", "margin-top: 8px;")])])])])])]);
function stepWithIndex(step, i) {
  if (typeof step !== "object" || step === null || Array.isArray(step)) {
    return null;
  }
  if (!step.props)
    step.props = {};
  step.props.internalIndex = i + 1;
  return step;
}
function stepsWithIndex(steps) {
  return steps.map((step, i) => stepWithIndex(step, i));
}
const stepsProps = Object.assign(Object.assign({}, useTheme.props), { current: Number, status: {
  type: String,
  default: "process"
}, size: {
  type: String,
  default: "medium"
}, vertical: Boolean, "onUpdate:current": [Function, Array], onUpdateCurrent: [Function, Array] });
const stepsInjectionKey = createInjectionKey("n-steps");
const NSteps = defineComponent({
  name: "Steps",
  props: stepsProps,
  setup(props, { slots }) {
    const { mergedClsPrefixRef, mergedRtlRef } = useConfig(props);
    const rtlEnabledRef = useRtl("Steps", mergedRtlRef, mergedClsPrefixRef);
    const themeRef = useTheme("Steps", "-steps", style, stepsLight, props, mergedClsPrefixRef);
    provide(stepsInjectionKey, {
      props,
      mergedThemeRef: themeRef,
      mergedClsPrefixRef,
      stepsSlots: slots
    });
    return {
      mergedClsPrefix: mergedClsPrefixRef,
      rtlEnabled: rtlEnabledRef
    };
  },
  render() {
    const { mergedClsPrefix } = this;
    return h("div", { class: [
      `${mergedClsPrefix}-steps`,
      this.rtlEnabled && `${mergedClsPrefix}-steps--rtl`,
      this.vertical && `${mergedClsPrefix}-steps--vertical`
    ] }, stepsWithIndex(flatten(getSlot(this))));
  }
});
const stepProps = {
  status: String,
  title: String,
  description: String,
  disabled: Boolean,
  // index will be filled by parent steps, not user
  internalIndex: {
    type: Number,
    default: 0
  }
};
const NStep = defineComponent({
  name: "Step",
  props: stepProps,
  setup(props) {
    const NSteps2 = inject(stepsInjectionKey, null);
    if (!NSteps2)
      throwError("step", "`n-step` must be placed inside `n-steps`.");
    const { inlineThemeDisabled } = useConfig();
    const { props: stepsProps2, mergedThemeRef, mergedClsPrefixRef, stepsSlots } = NSteps2;
    const verticalRef = computed(() => {
      return stepsProps2.vertical;
    });
    const mergedStatusRef = computed(() => {
      const { status } = props;
      if (status) {
        return status;
      } else {
        const { internalIndex } = props;
        const { current } = stepsProps2;
        if (current === void 0)
          return "process";
        if (internalIndex < current) {
          return "finish";
        } else if (internalIndex === current) {
          return stepsProps2.status || "process";
        } else if (internalIndex > current) {
          return "wait";
        }
      }
      return "process";
    });
    const cssVarsRef = computed(() => {
      const { value: status } = mergedStatusRef;
      const { size } = stepsProps2;
      const { common: { cubicBezierEaseInOut }, self: { stepHeaderFontWeight, [createKey("stepHeaderFontSize", size)]: stepHeaderFontSize, [createKey("indicatorIndexFontSize", size)]: indicatorIndexFontSize, [createKey("indicatorSize", size)]: indicatorSize, [createKey("indicatorIconSize", size)]: indicatorIconSize, [createKey("indicatorTextColor", status)]: indicatorTextColor, [createKey("indicatorBorderColor", status)]: indicatorBorderColor, [createKey("headerTextColor", status)]: headerTextColor, [createKey("splitorColor", status)]: splitorColor, [createKey("indicatorColor", status)]: indicatorColor, [createKey("descriptionTextColor", status)]: descriptionTextColor } } = mergedThemeRef.value;
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-description-text-color": descriptionTextColor,
        "--n-header-text-color": headerTextColor,
        "--n-indicator-border-color": indicatorBorderColor,
        "--n-indicator-color": indicatorColor,
        "--n-indicator-icon-size": indicatorIconSize,
        "--n-indicator-index-font-size": indicatorIndexFontSize,
        "--n-indicator-size": indicatorSize,
        "--n-indicator-text-color": indicatorTextColor,
        "--n-splitor-color": splitorColor,
        "--n-step-header-font-size": stepHeaderFontSize,
        "--n-step-header-font-weight": stepHeaderFontWeight
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("step", computed(() => {
      const { value: status } = mergedStatusRef;
      const { size } = stepsProps2;
      return `${status[0]}${size[0]}`;
    }), cssVarsRef, stepsProps2) : void 0;
    const handleStepClick = computed(() => {
      if (props.disabled)
        return void 0;
      const { onUpdateCurrent, "onUpdate:current": _onUpdateCurrent } = stepsProps2;
      return onUpdateCurrent || _onUpdateCurrent ? () => {
        if (onUpdateCurrent) {
          call(onUpdateCurrent, props.internalIndex);
        }
        if (_onUpdateCurrent) {
          call(_onUpdateCurrent, props.internalIndex);
        }
      } : void 0;
    });
    return {
      stepsSlots,
      mergedClsPrefix: mergedClsPrefixRef,
      vertical: verticalRef,
      mergedStatus: mergedStatusRef,
      handleStepClick,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender
    };
  },
  render() {
    const { mergedClsPrefix, onRender, handleStepClick, disabled } = this;
    const descriptionNode = resolveWrappedSlot(this.$slots.default, (children) => {
      const mergedDescription = children || this.description;
      if (mergedDescription) {
        return h("div", { class: `${mergedClsPrefix}-step-content__description` }, mergedDescription);
      }
      return null;
    });
    onRender === null || onRender === void 0 ? void 0 : onRender();
    return h(
      "div",
      { class: [
        `${mergedClsPrefix}-step`,
        disabled && `${mergedClsPrefix}-step--disabled`,
        !disabled && handleStepClick && `${mergedClsPrefix}-step--clickable`,
        this.themeClass,
        descriptionNode && `${mergedClsPrefix}-step--show-description`,
        `${mergedClsPrefix}-step--${this.mergedStatus}-status`
      ], style: this.cssVars, onClick: handleStepClick },
      h(
        "div",
        { class: `${mergedClsPrefix}-step-indicator` },
        h(
          "div",
          { class: `${mergedClsPrefix}-step-indicator-slot` },
          h(NIconSwitchTransition, null, {
            default: () => {
              return resolveWrappedSlot(this.$slots.icon, (icon) => {
                const { mergedStatus, stepsSlots } = this;
                return !(mergedStatus === "finish" || mergedStatus === "error") ? icon || h("div", { key: this.internalIndex, class: `${mergedClsPrefix}-step-indicator-slot__index` }, this.internalIndex) : mergedStatus === "finish" ? h(NBaseIcon, { clsPrefix: mergedClsPrefix, key: "finish" }, {
                  default: () => resolveSlot(stepsSlots["finish-icon"], () => [
                    h(FinishedIcon, null)
                  ])
                }) : mergedStatus === "error" ? h(NBaseIcon, { clsPrefix: mergedClsPrefix, key: "error" }, {
                  default: () => resolveSlot(stepsSlots["error-icon"], () => [
                    h(ErrorIcon, null)
                  ])
                }) : null;
              });
            }
          })
        ),
        this.vertical ? h("div", { class: `${mergedClsPrefix}-step-splitor` }) : null
      ),
      h(
        "div",
        { class: `${mergedClsPrefix}-step-content` },
        h(
          "div",
          { class: `${mergedClsPrefix}-step-content-header` },
          h("div", { class: `${mergedClsPrefix}-step-content-header__title` }, resolveSlot(this.$slots.title, () => [this.title])),
          !this.vertical ? h("div", { class: `${mergedClsPrefix}-step-splitor` }) : null
        ),
        descriptionNode
      )
    );
  }
});
const _hoisted_1$1 = { style: { "margin": "16px" } };
const _hoisted_2$1 = { class: "flex-container" };
const _hoisted_3$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Width", -1);
const _hoisted_4$1 = { class: "flex-container" };
const _hoisted_5$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Height", -1);
const _hoisted_6$1 = { class: "flex-container" };
const _hoisted_7$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Batch Size", -1);
const _hoisted_8$1 = { class: "flex-container" };
const _hoisted_9$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "CPU Threads (affects RAM usage)", -1);
const _hoisted_10$1 = { class: "flex-container" };
const _hoisted_11$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Model", -1);
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "AITemplateAccelerate",
  setup(__props) {
    var _a, _b;
    const message = useMessage();
    const global = useState();
    const width = ref(512);
    const height = ref(512);
    const batchSize = ref(1);
    const model = ref("");
    const threads = ref(8);
    const building = ref(false);
    const showUnloadModal = ref(false);
    const modelOptions = computed(() => {
      const options = [];
      for (const model2 of global.state.models) {
        if (model2.backend === "PyTorch" && model2.valid && !model2.name.endsWith(".safetensors") && !model2.name.endsWith(".ckpt")) {
          options.push({
            label: model2.name,
            value: model2.path
          });
        }
      }
      return options;
    });
    model.value = ((_b = (_a = modelOptions.value[0]) == null ? void 0 : _a.value) == null ? void 0 : _b.toString()) ?? "";
    const accelerateUnload = async () => {
      try {
        await fetch(`${serverUrl}/api/models/unload-all`, {
          method: "POST"
        });
        showUnloadModal.value = false;
        await accelerate();
      } catch {
        showUnloadModal.value = false;
        message.error("Failed to unload, check the console for more info.");
      }
    };
    const accelerate = async () => {
      showUnloadModal.value = false;
      building.value = true;
      await fetch(`${serverUrl}/api/generate/generate-aitemplate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model_id: model.value,
          width: width.value,
          height: height.value,
          batch_size: batchSize.value,
          threads: threads.value
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
        createVNode(unref(NCard), { title: "Acceleration progress (around 20 minutes)" }, {
          default: withCtx(() => [
            createVNode(unref(NSpace), {
              vertical: "",
              justify: "center"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSteps), null, {
                  default: withCtx(() => [
                    createVNode(unref(NStep), {
                      title: "UNet",
                      status: unref(global).state.aitBuildStep.unet
                    }, null, 8, ["status"]),
                    createVNode(unref(NStep), {
                      title: "ControlNet UNet",
                      status: unref(global).state.aitBuildStep.controlnet_unet
                    }, null, 8, ["status"]),
                    createVNode(unref(NStep), {
                      title: "CLIP",
                      status: unref(global).state.aitBuildStep.clip
                    }, null, 8, ["status"]),
                    createVNode(unref(NStep), {
                      title: "VAE",
                      status: unref(global).state.aitBuildStep.vae
                    }, null, 8, ["status"]),
                    createVNode(unref(NStep), {
                      title: "Cleanup",
                      status: unref(global).state.aitBuildStep.cleanup
                    }, null, 8, ["status"])
                  ]),
                  _: 1
                })
              ]),
              _: 1
            })
          ]),
          _: 1
        }),
        createVNode(unref(NCard), { style: { "margin-top": "16px" } }, {
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_2$1, [
              _hoisted_3$1,
              createVNode(unref(NSlider), {
                value: width.value,
                "onUpdate:value": _cache[0] || (_cache[0] = ($event) => width.value = $event),
                min: 128,
                max: 2048,
                step: 64,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: width.value,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => width.value = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                step: 64,
                min: 128,
                max: 2048
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_4$1, [
              _hoisted_5$1,
              createVNode(unref(NSlider), {
                value: height.value,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => height.value = $event),
                min: 128,
                max: 2048,
                step: 64,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: height.value,
                "onUpdate:value": _cache[3] || (_cache[3] = ($event) => height.value = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                step: 64,
                min: 128,
                max: 2048
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_6$1, [
              _hoisted_7$1,
              createVNode(unref(NSlider), {
                value: batchSize.value,
                "onUpdate:value": _cache[4] || (_cache[4] = ($event) => batchSize.value = $event),
                min: 1,
                max: 9,
                step: 1,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: batchSize.value,
                "onUpdate:value": _cache[5] || (_cache[5] = ($event) => batchSize.value = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                step: 1,
                min: 1,
                max: 9
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_8$1, [
              _hoisted_9$1,
              createVNode(unref(NSlider), {
                value: threads.value,
                "onUpdate:value": _cache[6] || (_cache[6] = ($event) => threads.value = $event),
                step: 1,
                min: 1,
                max: 64,
                style: { "margin-right": "12px" }
              }, null, 8, ["value"]),
              createVNode(unref(NInputNumber), {
                value: threads.value,
                "onUpdate:value": _cache[7] || (_cache[7] = ($event) => threads.value = $event),
                size: "small",
                style: { "min-width": "96px", "width": "96px" },
                step: 1,
                min: 1,
                max: 64
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_10$1, [
              _hoisted_11$1,
              createVNode(unref(NSelect), {
                value: model.value,
                "onUpdate:value": _cache[8] || (_cache[8] = ($event) => model.value = $event),
                options: modelOptions.value,
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
              disabled: building.value || modelOptions.value.length === 0,
              onClick: _cache[9] || (_cache[9] = ($event) => showUnloadModal.value = true)
            }, {
              default: withCtx(() => [
                createTextVNode("Accelerate")
              ]),
              _: 1
            }, 8, ["loading", "disabled"])
          ]),
          _: 1
        }),
        createVNode(unref(NModal), {
          show: showUnloadModal.value,
          "onUpdate:show": _cache[10] || (_cache[10] = ($event) => showUnloadModal.value = $event),
          preset: "dialog",
          title: "Unload other models",
          width: "400px",
          closable: false,
          "show-close": false,
          content: "Acceleration can be done with the other models loaded as well, but it will take a lot of resources. It is recommended to unload the other models before accelerating. Do you want to unload the other models?",
          "positive-text": "Unload models",
          "negative-text": "Keep models",
          onPositiveClick: accelerateUnload,
          onNegativeClick: accelerate
        }, null, 8, ["show"])
      ]);
    };
  }
});
const _hoisted_1 = { style: { "margin": "16px" } };
const _hoisted_2 = { class: "flex-container" };
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Model", -1);
const _hoisted_4 = { class: "flex-container" };
const _hoisted_5 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Simplify UNet", -1);
const _hoisted_6 = { class: "flex-container" };
const _hoisted_7 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Downcast to FP16", -1);
const _hoisted_8 = /* @__PURE__ */ createBaseVNode("h3", null, "Quantization", -1);
const _hoisted_9 = { class: "flex-container" };
const _hoisted_10 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Text Encoder", -1);
const _hoisted_11 = { class: "flex-container" };
const _hoisted_12 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "UNet", -1);
const _hoisted_13 = { class: "flex-container" };
const _hoisted_14 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "VAE Encoder", -1);
const _hoisted_15 = { class: "flex-container" };
const _hoisted_16 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "VAE Decoder", -1);
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "ONNXAccelerate",
  setup(__props) {
    var _a, _b;
    const message = useMessage();
    const global = useState();
    const conf = useSettings();
    const model = ref("");
    const building = ref(false);
    const showUnloadModal = ref(false);
    const modelOptions = computed(() => {
      const options = [];
      for (const model2 of global.state.models) {
        if (model2.backend === "PyTorch" && model2.valid && !model2.name.endsWith(".safetensors") && !model2.name.endsWith(".ckpt")) {
          options.push({
            label: model2.name,
            value: model2.path
          });
        }
      }
      return options;
    });
    model.value = ((_b = (_a = modelOptions.value[0]) == null ? void 0 : _a.value) == null ? void 0 : _b.toString()) ?? "";
    const accelerateUnload = async () => {
      try {
        await fetch(`${serverUrl}/api/models/unload-all`, {
          method: "POST"
        });
        showUnloadModal.value = false;
        await accelerate();
      } catch {
        showUnloadModal.value = false;
        message.error("Failed to unload, check the console for more info.");
      }
    };
    const accelerate = async () => {
      showUnloadModal.value = false;
      building.value = true;
      await fetch(`${serverUrl}/api/generate/generate-onnx`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model_id: model.value,
          quant_dict: conf.data.settings.onnx.quant_dict,
          simplify_unet: conf.data.settings.onnx.simplify_unet,
          convert_to_fp16: conf.data.settings.onnx.convert_to_fp16
        })
      }).then(() => {
        building.value = false;
      }).catch(() => {
        building.value = false;
        message.error("Failed to accelerate, check the console for more info.");
      });
    };
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NCard), { title: "Acceleration progress (around 5 minutes)" }, {
          default: withCtx(() => [
            createVNode(unref(NSpace), {
              vertical: "",
              justify: "center"
            }, {
              default: withCtx(() => [
                createVNode(unref(NSteps), null, {
                  default: withCtx(() => [
                    createVNode(unref(NStep), {
                      title: "CLIP",
                      status: unref(global).state.onnxBuildStep.clip
                    }, null, 8, ["status"]),
                    createVNode(unref(NStep), {
                      title: "UNet",
                      status: unref(global).state.onnxBuildStep.unet
                    }, null, 8, ["status"]),
                    createVNode(unref(NStep), {
                      title: "VAE",
                      status: unref(global).state.onnxBuildStep.vae
                    }, null, 8, ["status"]),
                    createVNode(unref(NStep), {
                      title: "Cleanup",
                      status: unref(global).state.onnxBuildStep.cleanup
                    }, null, 8, ["status"])
                  ]),
                  _: 1
                })
              ]),
              _: 1
            })
          ]),
          _: 1
        }),
        createVNode(unref(NCard), { style: { "margin-top": "16px" } }, {
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_2, [
              _hoisted_3,
              createVNode(unref(NSelect), {
                value: model.value,
                "onUpdate:value": _cache[0] || (_cache[0] = ($event) => model.value = $event),
                options: modelOptions.value,
                style: { "margin-right": "12px" }
              }, null, 8, ["value", "options"])
            ]),
            createBaseVNode("div", _hoisted_4, [
              _hoisted_5,
              createVNode(unref(NSwitch), {
                value: unref(conf).data.settings.onnx.simplify_unet,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(conf).data.settings.onnx.simplify_unet = $event)
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_6, [
              _hoisted_7,
              createVNode(unref(NSwitch), {
                value: unref(conf).data.settings.onnx.convert_to_fp16,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(conf).data.settings.onnx.convert_to_fp16 = $event)
              }, null, 8, ["value"])
            ]),
            _hoisted_8,
            createBaseVNode("div", _hoisted_9, [
              _hoisted_10,
              createVNode(unref(NSelect), {
                value: unref(conf).data.settings.onnx.quant_dict.text_encoder,
                "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(conf).data.settings.onnx.quant_dict.text_encoder = $event),
                options: [
                  { label: "No quantization", value: "no-quant" },
                  { label: "Unsigned int8 (cpu only)", value: "uint8" },
                  { label: "Signed int8", value: "int8" }
                ],
                style: { "margin-right": "12px" }
              }, null, 8, ["value", "options"])
            ]),
            createBaseVNode("div", _hoisted_11, [
              _hoisted_12,
              createVNode(unref(NSelect), {
                value: unref(conf).data.settings.onnx.quant_dict.unet,
                "onUpdate:value": _cache[4] || (_cache[4] = ($event) => unref(conf).data.settings.onnx.quant_dict.unet = $event),
                options: [
                  { label: "No quantization", value: "no-quant" },
                  { label: "Unsigned int8 (cpu only)", value: "uint8" },
                  { label: "Signed int8", value: "int8" }
                ],
                style: { "margin-right": "12px" }
              }, null, 8, ["value", "options"])
            ]),
            createBaseVNode("div", _hoisted_13, [
              _hoisted_14,
              createVNode(unref(NSelect), {
                value: unref(conf).data.settings.onnx.quant_dict.vae_encoder,
                "onUpdate:value": _cache[5] || (_cache[5] = ($event) => unref(conf).data.settings.onnx.quant_dict.vae_encoder = $event),
                options: [
                  { label: "No quantization", value: "no-quant" },
                  { label: "Unsigned int8 (cpu only)", value: "uint8" },
                  { label: "Signed int8", value: "int8" }
                ],
                style: { "margin-right": "12px" }
              }, null, 8, ["value", "options"])
            ]),
            createBaseVNode("div", _hoisted_15, [
              _hoisted_16,
              createVNode(unref(NSelect), {
                value: unref(conf).data.settings.onnx.quant_dict.vae_decoder,
                "onUpdate:value": _cache[6] || (_cache[6] = ($event) => unref(conf).data.settings.onnx.quant_dict.vae_decoder = $event),
                options: [
                  { label: "No quantization", value: "no-quant" },
                  { label: "Unsigned int8 (cpu only)", value: "uint8" },
                  { label: "Signed int8", value: "int8" }
                ],
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
              disabled: building.value || modelOptions.value.length === 0,
              onClick: _cache[7] || (_cache[7] = ($event) => showUnloadModal.value = true)
            }, {
              default: withCtx(() => [
                createTextVNode("Accelerate")
              ]),
              _: 1
            }, 8, ["loading", "disabled"])
          ]),
          _: 1
        }),
        createVNode(unref(NModal), {
          show: showUnloadModal.value,
          "onUpdate:show": _cache[8] || (_cache[8] = ($event) => showUnloadModal.value = $event),
          preset: "dialog",
          title: "Unload other models",
          width: "400px",
          closable: false,
          "show-close": false,
          content: "Acceleration can be done with the other models loaded as well, but it will take a lot of resources. It is recommended to unload the other models before accelerating. Do you want to unload the other models?",
          "positive-text": "Unload models",
          "negative-text": "Keep models",
          onPositiveClick: accelerateUnload,
          onNegativeClick: accelerate
        }, null, 8, ["show"])
      ]);
    };
  }
});
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "TensorRTAccelerate",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NCard), { title: "Acceleration progress" }, {
        default: withCtx(() => [
          createVNode(unref(NSpace), {
            vertical: "",
            justify: "center"
          }, {
            default: withCtx(() => [
              createVNode(unref(NSteps), { current: 1 }, {
                default: withCtx(() => [
                  createVNode(unref(NStep), {
                    title: "Start",
                    description: "Start the acceleration process by clicking the button next to the model"
                  }),
                  createVNode(unref(NStep), {
                    title: "Convert to ONNX",
                    description: "This process might take a while"
                  }),
                  createVNode(unref(NStep), {
                    title: "Convert to TensorRT",
                    description: "This process might take a while"
                  }),
                  createVNode(unref(NStep), {
                    title: "Package and cleanup",
                    description: "This process might take a while"
                  })
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
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "AccelerateView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NTabs), { type: "segment" }, {
        default: withCtx(() => [
          createVNode(unref(NTabPane), { name: "AITemplate" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$3)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "ONNX" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$2)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "TensorRT" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$1)
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
