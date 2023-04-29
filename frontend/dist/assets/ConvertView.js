import { d as defineComponent, b as useMessage, A as ref, aU as reactive, v as serverUrl, o as openBlock, e as createElementBlock, f as createVNode, w as withCtx, l as createBaseVNode, g as unref, n as NSelect, h as NCard, B as NButton, k as createTextVNode, i as NSpace, a_ as NModal, p as createBlock, D as NTabPane, E as NTabs } from "./index.js";
import { N as NSwitch } from "./Switch.js";
import { _ as _sfc_main$2 } from "./WIP.vue_vue_type_script_setup_true_lang.js";
const _hoisted_1 = { style: { "margin": "16px" } };
const _hoisted_2 = { class: "flex-container" };
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "FP32", -1);
const _hoisted_4 = { class: "flex-container" };
const _hoisted_5 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Output in safetensors format", -1);
const _hoisted_6 = { class: "flex-container" };
const _hoisted_7 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Model", -1);
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
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
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NCard), { style: { "margin-top": "16px" } }, {
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_2, [
              _hoisted_3,
              createVNode(unref(NSwitch), {
                value: use_fp32.value,
                "onUpdate:value": _cache[0] || (_cache[0] = ($event) => use_fp32.value = $event)
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_4, [
              _hoisted_5,
              createVNode(unref(NSwitch), {
                value: safetensors.value,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => safetensors.value = $event)
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_6, [
              _hoisted_7,
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
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ConvertView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NTabs), { type: "segment" }, {
        default: withCtx(() => [
          createVNode(unref(NTabPane), { name: "Convert" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$1)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Merge" }, {
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
