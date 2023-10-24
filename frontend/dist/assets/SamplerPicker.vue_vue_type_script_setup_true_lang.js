import { d as defineComponent, a as useSettings, D as ref, c as computed, e as openBlock, f as createElementBlock, n as createBaseVNode, g as createVNode, w as withCtx, h as unref, i as NCard, I as Fragment, L as renderList, E as NButton, m as createTextVNode, t as toDisplayString, bC as convertToTextString, v as createBlock, bG as resolveDynamicComponent, bd as NModal, q as NTooltip, x as NSelect, F as NIcon, C as h } from "./index.js";
import { S as Settings, a as NCheckbox } from "./Settings.js";
import { N as NInputNumber } from "./InputNumber.js";
import { N as NSlider } from "./Switch.js";
const _hoisted_1 = { class: "flex-container" };
const _hoisted_2 = { style: { "margin-left": "12px", "margin-right": "12px", "white-space": "nowrap" } };
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "100px" } }, "Sampler", -1);
const _hoisted_4 = /* @__PURE__ */ createBaseVNode("a", {
  target: "_blank",
  href: "https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
}, "Learn more", -1);
const _hoisted_5 = { class: "flex-container" };
const _hoisted_6 = /* @__PURE__ */ createBaseVNode("p", { style: { "margin-right": "12px", "width": "94px" } }, "Sigmas", -1);
const _hoisted_7 = /* @__PURE__ */ createBaseVNode("b", { class: "highlight" }, 'Only "Default" and "Karras" sigmas work on diffusers samplers (and "Karras" are only applied to KDPM samplers)', -1);
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "SamplerPicker",
  props: {
    type: {
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
    function getValue(param) {
      const val = target.value.sampler_config[target.value[props.type].sampler][param];
      return val;
    }
    function setValue(param, value) {
      target.value.sampler_config[target.value[props.type].sampler][param] = value;
    }
    function resolveComponent(settings2, param) {
      switch (settings2.componentType) {
        case "slider":
          return h(NSlider, {
            min: settings2.min,
            max: settings2.max,
            step: settings2.step,
            value: getValue(param),
            onUpdateValue: (value) => setValue(param, value)
          });
        case "select":
          return h(NSelect, {
            options: settings2.options,
            value: getValue(param),
            onUpdateValue: (value) => setValue(param, value)
          });
        case "boolean":
          return h(NCheckbox, {
            checked: getValue(param),
            onUpdateChecked: (value) => setValue(param, value)
          });
        case "number":
          return h(NInputNumber, {
            min: settings2.min,
            max: settings2.max,
            step: settings2.step,
            value: getValue(param),
            onUpdateValue: (value) => setValue(param, value)
          });
      }
    }
    const target = computed(() => {
      if (props.target === "settings") {
        return settings.data.settings;
      }
      return settings.defaultSettings;
    });
    const computedSettings = computed(() => {
      return target.value.sampler_config[target.value[props.type].sampler] ?? {};
    });
    const sigmaOptions = computed(() => {
      const karras = typeof target.value[props.type].sampler === "string";
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
      if (typeof target.value[props.type].sampler !== "string") {
        if (!["automatic", "karras"].includes(target.value[props.type].sigmas)) {
          return "error";
        } else {
          return void 0;
        }
      }
      return void 0;
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createBaseVNode("div", _hoisted_1, [
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
                      createBaseVNode("p", _hoisted_2, toDisplayString(unref(convertToTextString)(param)), 1),
                      (openBlock(), createBlock(resolveDynamicComponent(
                        resolveComponent(
                          target.value.sampler_config["ui_settings"][param],
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
              _hoisted_3
            ]),
            default: withCtx(() => [
              createTextVNode(" The sampler is the method used to generate the image. Your result may vary drastically depending on the sampler you choose. "),
              _hoisted_4
            ]),
            _: 1
          }),
          createVNode(unref(NSelect), {
            options: unref(settings).scheduler_options,
            filterable: "",
            value: target.value[props.type].sampler,
            "onUpdate:value": _cache[2] || (_cache[2] = ($event) => target.value[props.type].sampler = $event),
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
        createBaseVNode("div", _hoisted_5, [
          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
            trigger: withCtx(() => [
              _hoisted_6
            ]),
            default: withCtx(() => [
              createTextVNode(" Changes the sigmas used in the diffusion process. Can change the quality of the output. "),
              _hoisted_7
            ]),
            _: 1
          }),
          createVNode(unref(NSelect), {
            options: sigmaOptions.value,
            value: target.value[props.type].sigmas,
            "onUpdate:value": _cache[4] || (_cache[4] = ($event) => target.value[props.type].sigmas = $event),
            status: sigmaValidationStatus.value
          }, null, 8, ["options", "value", "status"])
        ])
      ], 64);
    };
  }
});
export {
  _sfc_main as _
};
