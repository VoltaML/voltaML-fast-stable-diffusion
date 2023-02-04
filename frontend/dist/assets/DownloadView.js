import { d as defineComponent, u as useState, D as ref, c as createElementBlock, e as createVNode, w as withCtx, f as unref, j as createBaseVNode, R as Fragment, o as openBlock, i as NInput, as as NButton, l as createTextVNode, h as NSpace, x as serverUrl } from "./index.js";
import { N as NResult } from "./Result.js";
const _hoisted_1 = { style: { "height": "50vh", "display": "inline-flex", "justify-content": "center", "width": "100%" } };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "DownloadView",
  setup(__props) {
    const conf = useState();
    const customModel = ref("");
    function downloadModel() {
      const url = new URL(`${serverUrl}/api/models/download`);
      url.searchParams.append("model", customModel.value);
      console.log(url);
      conf.state.downloading = true;
      customModel.value = "";
      fetch(url, { method: "POST" }).then(() => {
        conf.state.downloading = false;
      }).catch(() => {
        conf.state.downloading = false;
      });
    }
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(unref(NSpace), {
          justify: "end",
          inline: "",
          align: "center",
          class: "install",
          style: { "width": "100%", "margin": "8px" }
        }, {
          default: withCtx(() => [
            createVNode(unref(NInput), {
              value: customModel.value,
              "onUpdate:value": _cache[0] || (_cache[0] = ($event) => customModel.value = $event),
              placeholder: "Custom model",
              style: { "width": "350px" }
            }, null, 8, ["value"]),
            createVNode(unref(NButton), {
              type: "primary",
              bordered: "",
              onClick: downloadModel,
              loading: unref(conf).state.downloading,
              disabled: unref(conf).state.downloading || customModel.value === "",
              secondary: "",
              style: { "margin-right": "16px" }
            }, {
              default: withCtx(() => [
                createTextVNode("Install")
              ]),
              _: 1
            }, 8, ["loading", "disabled"])
          ]),
          _: 1
        }),
        createBaseVNode("div", _hoisted_1, [
          createVNode(unref(NResult), {
            status: "info",
            title: "Download",
            description: "Work in progress"
          })
        ])
      ], 64);
    };
  }
});
export {
  _sfc_main as default
};
