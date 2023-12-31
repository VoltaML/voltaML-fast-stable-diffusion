import { d as defineComponent, B as ref, o as openBlock, g as createBlock, f as unref } from "./index.js";
import { _ as _sfc_main$1 } from "./ModelPopup.vue_vue_type_script_setup_true_lang.js";
import "./DescriptionsItem.js";
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TestView",
  setup(__props) {
    const model = ref(null);
    const showModal = ref(false);
    fetch("https://civitai.com/api/v1/models/7240").then((res) => {
      res.json().then((data) => {
        model.value = data;
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(_sfc_main$1), {
        model: model.value,
        "show-modal": showModal.value,
        "onUpdate:showModal": _cache[0] || (_cache[0] = (e) => showModal.value = e)
      }, null, 8, ["model", "show-modal"]);
    };
  }
});
export {
  _sfc_main as default
};
