import { _ as _sfc_main$1, a as NDescriptions, N as NDescriptionsItem } from "./SendOutputTo.vue_vue_type_script_setup_true_lang.js";
import { d as defineComponent, o as openBlock, e as createElementBlock, l as createBaseVNode, u as useState, A as ref, c as computed, aU as reactive, f as createVNode, g as unref, w as withCtx, v as serverUrl, a_ as NModal, N as NGi, p as createBlock, q as createCommentVNode, h as NCard, E as NTabs, D as NTabPane, H as NScrollbar, F as Fragment, G as renderList, r as NGrid, B as NButton, C as NIcon, ba as Download, k as createTextVNode, t as toDisplayString, _ as _export_sfc } from "./index.js";
import { N as NImage } from "./Image.js";
const _hoisted_1$1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode(
  "rect",
  {
    x: "32",
    y: "48",
    width: "448",
    height: "80",
    rx: "32",
    ry: "32",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$1 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M74.45 160a8 8 0 0 0-8 8.83l26.31 252.56a1.5 1.5 0 0 0 0 .22A48 48 0 0 0 140.45 464h231.09a48 48 0 0 0 47.67-42.39v-.21l26.27-252.57a8 8 0 0 0-8-8.83zm248.86 180.69a16 16 0 1 1-22.63 22.62L256 318.63l-44.69 44.68a16 16 0 0 1-22.63-22.62L233.37 296l-44.69-44.69a16 16 0 0 1 22.63-22.62L256 273.37l44.68-44.68a16 16 0 0 1 22.63 22.62L278.62 296z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$1 = [_hoisted_2$1, _hoisted_3$1];
const TrashBin = defineComponent({
  name: "TrashBin",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$1, _hoisted_4$1);
  }
});
const _hoisted_1 = { style: { "margin": "18px" } };
const _hoisted_2 = { style: { "height": "100%", "width": "100%" } };
const _hoisted_3 = ["onClick"];
const _hoisted_4 = ["onClick"];
const _hoisted_5 = ["onClick"];
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ImageBrowserView",
  setup(__props) {
    const global = useState();
    const showDeleteModal = ref(false);
    function urlFromPath(path) {
      const url = new URL(path, serverUrl);
      return url.href;
    }
    const imageSrc = computed(() => {
      const url = urlFromPath(global.state.imageBrowser.currentImage.path);
      return url;
    });
    function deleteImage() {
      const url = new URL(`${serverUrl}/api/output/delete/`);
      url.searchParams.append(
        "filename",
        global.state.imageBrowser.currentImage.path
      );
      fetch(url, { method: "DELETE" }).then((res) => res.json()).then(() => {
        global.state.imageBrowser.currentImage = {
          path: "",
          id: "",
          time: 0
        };
        global.state.imageBrowser.currentImageByte64 = "";
        global.state.imageBrowser.currentImageMetadata = /* @__PURE__ */ new Map();
        refreshImages();
      });
    }
    function downloadImage() {
      const url = urlFromPath(global.state.imageBrowser.currentImage.path);
      fetch(url).then((res) => res.blob()).then((blob) => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = function() {
          const base64data = reader.result;
          if (base64data !== null) {
            const a = document.createElement("a");
            a.href = base64data.toString();
            a.download = global.state.imageBrowser.currentImage.id;
            a.target = "_blank";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
          } else {
            console.log("base64data is null!");
          }
        };
      });
    }
    function setByte64FromImage(path) {
      const url = urlFromPath(path);
      fetch(url).then((res) => res.blob()).then((blob) => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = function() {
          const base64data = reader.result;
          if (base64data !== null) {
            global.state.imageBrowser.currentImageByte64 = base64data.toString();
          } else {
            console.log("base64data is null!");
          }
        };
      });
    }
    function txt2imgClick(i) {
      global.state.imageBrowser.currentImage = txt2imgData[i];
      setByte64FromImage(txt2imgData[i].path);
      const url = new URL(`${serverUrl}/api/output/data/`);
      url.searchParams.append("filename", txt2imgData[i].path);
      fetch(url).then((res) => res.json()).then((data) => {
        global.state.imageBrowser.currentImageMetadata = data;
      });
    }
    function img2imgClick(i) {
      global.state.imageBrowser.currentImage = img2imgData[i];
      setByte64FromImage(img2imgData[i].path);
      const url = new URL(`${serverUrl}/api/output/data/`);
      url.searchParams.append("filename", img2imgData[i].path);
      fetch(url).then((res) => res.json()).then((data) => {
        global.state.imageBrowser.currentImageMetadata = data;
      });
    }
    function extraClick(i) {
      global.state.imageBrowser.currentImage = extraData[i];
      setByte64FromImage(extraData[i].path);
      const url = new URL(`${serverUrl}/api/output/data/`);
      url.searchParams.append("filename", extraData[i].path);
      fetch(url).then((res) => res.json()).then((data) => {
        global.state.imageBrowser.currentImageMetadata = data;
      });
    }
    const txt2imgData = reactive([]);
    const img2imgData = reactive([]);
    const extraData = reactive([]);
    function refreshImages() {
      txt2imgData.splice(0, txt2imgData.length);
      img2imgData.splice(0, img2imgData.length);
      extraData.splice(0, extraData.length);
      fetch(`${serverUrl}/api/output/txt2img`).then((res) => res.json()).then((data) => {
        data.forEach((item) => {
          txt2imgData.push(item);
        });
        txt2imgData.sort((a, b) => {
          return b.time - a.time;
        });
      });
      fetch(`${serverUrl}/api/output/img2img`).then((res) => res.json()).then((data) => {
        data.forEach((item) => {
          img2imgData.push(item);
        });
        img2imgData.sort((a, b) => {
          return b.time - a.time;
        });
      });
      fetch(`${serverUrl}/api/output/extra`).then((res) => res.json()).then((data) => {
        data.forEach((item) => {
          extraData.push(item);
        });
        extraData.sort((a, b) => {
          return b.time - a.time;
        });
      });
    }
    refreshImages();
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NModal), {
          show: showDeleteModal.value,
          "onUpdate:show": _cache[0] || (_cache[0] = ($event) => showDeleteModal.value = $event),
          "mask-closable": false,
          preset: "confirm",
          type: "error",
          title: "Delete Image",
          content: "Do you want to delete this image? This action cannot be undone.",
          "positive-text": "Confirm",
          "negative-text": "Cancel",
          "transform-origin": "center",
          onPositiveClick: deleteImage,
          onNegativeClick: _cache[1] || (_cache[1] = ($event) => showDeleteModal.value = false)
        }, null, 8, ["show"]),
        createVNode(unref(NGrid), {
          cols: "1 850:3",
          "x-gap": "12px"
        }, {
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
            createVNode(unref(NGi), { span: "1" }, {
              default: withCtx(() => [
                createBaseVNode("div", _hoisted_2, [
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
                                    ], 8, _hoisted_3);
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
                                    ], 8, _hoisted_4);
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
                                    ], 8, _hoisted_5);
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
                unref(global).state.imageBrowser.currentImage.path !== "" ? (openBlock(), createBlock(unref(NGrid), {
                  key: 0,
                  cols: "2",
                  "x-gap": "12"
                }, {
                  default: withCtx(() => [
                    createVNode(unref(NGi), null, {
                      default: withCtx(() => [
                        createVNode(_sfc_main$1, {
                          output: unref(global).state.imageBrowser.currentImageByte64
                        }, null, 8, ["output"])
                      ]),
                      _: 1
                    }),
                    createVNode(unref(NGi), null, {
                      default: withCtx(() => [
                        createVNode(unref(NCard), {
                          style: { "margin": "12px 0" },
                          title: "Manage"
                        }, {
                          default: withCtx(() => [
                            createVNode(unref(NGrid), {
                              cols: "4",
                              "x-gap": "4",
                              "y-gap": "4"
                            }, {
                              default: withCtx(() => [
                                createVNode(unref(NGi), null, {
                                  default: withCtx(() => [
                                    createVNode(unref(NButton), {
                                      type: "success",
                                      onClick: downloadImage,
                                      style: { "width": "100%" },
                                      ghost: ""
                                    }, {
                                      icon: withCtx(() => [
                                        createVNode(unref(NIcon), null, {
                                          default: withCtx(() => [
                                            createVNode(unref(Download))
                                          ]),
                                          _: 1
                                        })
                                      ]),
                                      default: withCtx(() => [
                                        createTextVNode("Download")
                                      ]),
                                      _: 1
                                    })
                                  ]),
                                  _: 1
                                }),
                                createVNode(unref(NGi), null, {
                                  default: withCtx(() => [
                                    createVNode(unref(NButton), {
                                      type: "error",
                                      onClick: _cache[2] || (_cache[2] = ($event) => showDeleteModal.value = true),
                                      style: { "width": "100%" },
                                      ghost: ""
                                    }, {
                                      icon: withCtx(() => [
                                        createVNode(unref(NIcon), null, {
                                          default: withCtx(() => [
                                            createVNode(unref(TrashBin))
                                          ]),
                                          _: 1
                                        })
                                      ]),
                                      default: withCtx(() => [
                                        createTextVNode(" Delete")
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
                      ]),
                      _: 1
                    })
                  ]),
                  _: 1
                })) : createCommentVNode("", true)
              ]),
              _: 1
            }),
            createVNode(unref(NGi), { span: "3" }, {
              default: withCtx(() => [
                unref(global).state.imageBrowser.currentImageMetadata.size !== 0 ? (openBlock(), createBlock(unref(NDescriptions), {
                  key: 0,
                  bordered: ""
                }, {
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
                })) : createCommentVNode("", true)
              ]),
              _: 1
            })
          ]),
          _: 1
        })
      ]);
    };
  }
});
const ImageBrowserView_vue_vue_type_style_index_0_scoped_a83deb7b_lang = "";
const ImageBrowserView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-a83deb7b"]]);
export {
  ImageBrowserView as default
};
