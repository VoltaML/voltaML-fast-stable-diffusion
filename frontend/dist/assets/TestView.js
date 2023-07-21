import { d as defineComponent, E as ref } from "./index.js";
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TestView",
  setup(__props) {
    const model = ref(null);
    fetch("https://civitai.com/api/v1/models/7240").then((res) => {
      res.json().then((data) => {
        model.value = data;
      });
    });
    return () => {
    };
  }
});
export {
  _sfc_main as default
};
