import { d as defineComponent, F as ref, c as computed, e as openBlock, f as createElementBlock, g as createVNode, w as withCtx, h as unref, m as createTextVNode, G as NButton } from "./index.js";
import { N as NDataTable } from "./DataTable.js";
const _hoisted_1 = { class: "main-container" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TestView",
  setup(__props) {
    const loading = ref(false);
    const models = ref();
    const createColumns = () => {
      return [
        {
          title: "Name",
          key: "name"
        },
        {
          title: "ID",
          key: "id"
        },
        {
          title: "NSFW",
          key: "nsfw"
        },
        {
          title: "Type",
          key: "type"
        }
      ];
    };
    const columns = createColumns();
    const data = computed(() => {
      var _a;
      return (_a = models.value) == null ? void 0 : _a.items;
    });
    async function test() {
      loading.value = true;
      const url = new URL("https://civitai.com/api/v1/models");
      url.searchParams.append("query", "Anything");
      const res = await fetch(url);
      models.value = await res.json();
      console.log(models);
      loading.value = false;
    }
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NButton), {
          onClick: test,
          loading: loading.value
        }, {
          default: withCtx(() => [
            createTextVNode("Test")
          ]),
          _: 1
        }, 8, ["loading"]),
        createVNode(unref(NDataTable), {
          columns: unref(columns),
          loading: loading.value,
          data: data.value
        }, null, 8, ["columns", "loading", "data"])
      ]);
    };
  }
});
export {
  _sfc_main as default
};
