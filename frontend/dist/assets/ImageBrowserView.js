import { d as defineComponent, u as useState, c as computed, aV as reactive, v as serverUrl, o as openBlock, p as createBlock, w as withCtx, g as unref, f as createVNode, q as createCommentVNode, N as NGi, l as createBaseVNode, h as NCard, E as NTabs, D as NTabPane, I as NScrollbar, e as createElementBlock, H as renderList, G as Fragment, r as NGrid, k as createTextVNode, t as toDisplayString, _ as _export_sfc } from "./index.js";
import { N as NImage } from "./Image.js";
import { a as NDescriptions, N as NDescriptionsItem } from "./DescriptionsItem.js";
const _hoisted_1 = { style: { "height": "100%", "width": "100%" } };
const _hoisted_2 = ["onClick"];
const _hoisted_3 = ["onClick"];
const _hoisted_4 = ["onClick"];
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ImageBrowserView",
  setup(__props) {
    const global = useState();
    function urlFromPath(path) {
      const url = new URL(path, serverUrl);
      return url.href;
    }
    const imageSrc = computed(() => {
      const url = urlFromPath(global.state.imageBrowser.currentImage.path);
      return url;
    });
    function txt2imgClick(i) {
      global.state.imageBrowser.currentImage = txt2imgData[i];
      console.log(txt2imgData[i].path);
      const url = new URL(`${serverUrl}/api/output/data/`);
      url.searchParams.append("filename", txt2imgData[i].path);
      console.log(url);
      fetch(url).then((res) => res.json()).then((data) => {
        global.state.imageBrowser.currentImageMetadata = data;
      });
    }
    function img2imgClick(i) {
      global.state.imageBrowser.currentImage = img2imgData[i];
      console.log(img2imgData[i].path);
      const url = new URL(`${serverUrl}/api/output/data/`);
      url.searchParams.append("filename", img2imgData[i].path);
      console.log(url);
      fetch(url).then((res) => res.json()).then((data) => {
        global.state.imageBrowser.currentImageMetadata = data;
      });
    }
    function extraClick(i) {
      global.state.imageBrowser.currentImage = extraData[i];
      console.log(extraData[i].path);
      const url = new URL(`${serverUrl}/api/output/data/`);
      url.searchParams.append("filename", extraData[i].path);
      console.log(url);
      fetch(url).then((res) => res.json()).then((data) => {
        global.state.imageBrowser.currentImageMetadata = data;
      });
    }
    const txt2imgData = reactive([]);
    fetch(`${serverUrl}/api/output/txt2img`).then((res) => res.json()).then((data) => {
      data.forEach((item) => {
        txt2imgData.push(item);
      });
      txt2imgData.sort((a, b) => {
        return b.time - a.time;
      });
    });
    const img2imgData = reactive([]);
    fetch(`${serverUrl}/api/output/img2img`).then((res) => res.json()).then((data) => {
      data.forEach((item) => {
        img2imgData.push(item);
      });
      img2imgData.sort((a, b) => {
        return b.time - a.time;
      });
    });
    const extraData = reactive([]);
    fetch(`${serverUrl}/api/output/extra`).then((res) => res.json()).then((data) => {
      data.forEach((item) => {
        extraData.push(item);
      });
      extraData.sort((a, b) => {
        return b.time - a.time;
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NGrid), { cols: "1 850:3" }, {
        default: withCtx(() => [
          createVNode(unref(NGi), {
            span: "2",
            style: { "height": "calc((100vh - 170px) - 24px)", "display": "inline-flex", "justify-content": "center" }
          }, {
            default: withCtx(() => [
              unref(global).state.imageBrowser.currentImage.path !== "" ? (openBlock(), createBlock(unref(NImage), {
                key: 0,
                src: unref(imageSrc),
                "object-fit": "contain",
                style: { "width": "100%", "height": "100%", "justify-content": "center", "margin": "8px" },
                "img-props": { style: { maxWidth: "95%", maxHeight: "95%" } }
              }, null, 8, ["src"])) : createCommentVNode("", true)
            ]),
            _: 1
          }),
          createVNode(unref(NGi), null, {
            default: withCtx(() => [
              createBaseVNode("div", _hoisted_1, [
                createVNode(unref(NCard), null, {
                  default: withCtx(() => [
                    createVNode(unref(NTabs), {
                      type: "segment",
                      style: { "height": "100%" }
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NTabPane), {
                          name: "Txt2Img",
                          style: { "height": "calc(((100vh - 200px) - 53px) - 24px)" }
                        }, {
                          default: withCtx(() => [
                            createVNode(unref(NScrollbar), {
                              trigger: "hover",
                              style: { "height": "100%" }
                            }, {
                              default: withCtx(() => [
                                (openBlock(true), createElementBlock(Fragment, null, renderList(txt2imgData, (i, index) => {
                                  return openBlock(), createElementBlock("span", {
                                    onClick: ($event) => txt2imgClick(index),
                                    key: index,
                                    class: "img-container"
                                  }, [
                                    createVNode(unref(NImage), {
                                      class: "img-slider",
                                      src: urlFromPath(i.path),
                                      lazy: "",
                                      "preview-disabled": "",
                                      style: { "justify-content": "center" }
                                    }, null, 8, ["src"])
                                  ], 8, _hoisted_2);
                                }), 128))
                              ]),
                              _: 1
                            })
                          ]),
                          _: 1
                        }),
                        createVNode(unref(NTabPane), {
                          name: "Img2Img",
                          style: { "height": "calc(((100vh - 200px) - 53px) - 24px)" }
                        }, {
                          default: withCtx(() => [
                            createVNode(unref(NScrollbar), {
                              trigger: "hover",
                              style: { "height": "100%" }
                            }, {
                              default: withCtx(() => [
                                (openBlock(true), createElementBlock(Fragment, null, renderList(img2imgData, (i, index) => {
                                  return openBlock(), createElementBlock("span", {
                                    onClick: ($event) => img2imgClick(index),
                                    key: index,
                                    class: "img-container"
                                  }, [
                                    createVNode(unref(NImage), {
                                      class: "img-slider",
                                      src: urlFromPath(i.path),
                                      lazy: "",
                                      "preview-disabled": "",
                                      style: { "justify-content": "center" }
                                    }, null, 8, ["src"])
                                  ], 8, _hoisted_3);
                                }), 128))
                              ]),
                              _: 1
                            })
                          ]),
                          _: 1
                        }),
                        createVNode(unref(NTabPane), {
                          name: "Extra",
                          style: { "height": "calc(((100vh - 200px) - 53px) - 24px)" }
                        }, {
                          default: withCtx(() => [
                            createVNode(unref(NScrollbar), {
                              trigger: "hover",
                              style: { "height": "100%" }
                            }, {
                              default: withCtx(() => [
                                (openBlock(true), createElementBlock(Fragment, null, renderList(extraData, (i, index) => {
                                  return openBlock(), createElementBlock("span", {
                                    onClick: ($event) => extraClick(index),
                                    key: index,
                                    class: "img-container"
                                  }, [
                                    createVNode(unref(NImage), {
                                      class: "img-slider",
                                      src: urlFromPath(i.path),
                                      lazy: "",
                                      "preview-disabled": "",
                                      style: { "justify-content": "center" }
                                    }, null, 8, ["src"])
                                  ], 8, _hoisted_4);
                                }), 128))
                              ]),
                              _: 1
                            })
                          ]),
                          _: 1
                        })
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                })
              ])
            ]),
            _: 1
          }),
          createVNode(unref(NGi), { span: "3" }, {
            default: withCtx(() => [
              createVNode(unref(NDescriptions), { bordered: "" }, {
                default: withCtx(() => [
                  (openBlock(true), createElementBlock(Fragment, null, renderList(unref(global).state.imageBrowser.currentImageMetadata, (item, key) => {
                    return openBlock(), createBlock(unref(NDescriptionsItem), {
                      label: key.toString(),
                      "content-style": "max-width: 100px",
                      key: item.toString()
                    }, {
                      default: withCtx(() => [
                        createTextVNode(toDisplayString(item), 1)
                      ]),
                      _: 2
                    }, 1032, ["label"]);
                  }), 128))
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
const ImageBrowserView_vue_vue_type_style_index_0_scoped_0f01815b_lang = "";
const ImageBrowserView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-0f01815b"]]);
export {
  ImageBrowserView as default
};
