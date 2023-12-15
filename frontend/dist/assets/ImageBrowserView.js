import { d as defineComponent, o as openBlock, g as createElementBlock, b as createBaseVNode, b7 as useCssVars, a as useState, u as useSettings, R as inject, y as ref, i as computed, bM as urlFromPath, b9 as reactive, ba as onMounted, p as onUnmounted, e as createVNode, f as unref, w as withCtx, F as Fragment, L as renderList, b8 as themeOverridesKey, s as serverUrl, I as NInput, A as NIcon, bg as NModal, r as NGrid, q as NGi, z as NButton, k as createTextVNode, M as NScrollbar, c as createBlock, bB as convertToTextString, B as toDisplayString, h as createCommentVNode, bN as diffusersSchedulerTuple, _ as _export_sfc } from "./index.js";
import { D as Download, _ as _sfc_main$1 } from "./SendOutputTo.vue_vue_type_script_setup_true_lang.js";
import { N as NImage, T as TrashBin } from "./TrashBin.js";
import { N as NSlider } from "./Slider.js";
import { N as NDescriptionsItem, a as NDescriptions } from "./DescriptionsItem.js";
import "./Switch.js";
const _hoisted_1$1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode(
  "rect",
  {
    x: "48",
    y: "48",
    width: "176",
    height: "176",
    rx: "20",
    ry: "20",
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "stroke-width": "32"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$1 = /* @__PURE__ */ createBaseVNode(
  "rect",
  {
    x: "288",
    y: "48",
    width: "176",
    height: "176",
    rx: "20",
    ry: "20",
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "stroke-width": "32"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$1 = /* @__PURE__ */ createBaseVNode(
  "rect",
  {
    x: "48",
    y: "288",
    width: "176",
    height: "176",
    rx: "20",
    ry: "20",
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "stroke-width": "32"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_5 = /* @__PURE__ */ createBaseVNode(
  "rect",
  {
    x: "288",
    y: "288",
    width: "176",
    height: "176",
    rx: "20",
    ry: "20",
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "stroke-width": "32"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_6 = [_hoisted_2$1, _hoisted_3$1, _hoisted_4$1, _hoisted_5];
const GridOutline = defineComponent({
  name: "GridOutline",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$1, _hoisted_6);
  }
});
const _hoisted_1 = {
  style: { "width": "calc(100vw - 98px)", "height": "48px", "border-bottom": "#505050 1px solid", "margin-top": "52px", "display": "flex", "justify-content": "end", "align-items": "center", "padding-right": "24px", "position": "fixed", "top": "0", "z-index": "1" },
  class: "top-bar"
};
const _hoisted_2 = {
  class: "main-container",
  style: { "margin-top": "114px" }
};
const _hoisted_3 = { class: "image-grid" };
const _hoisted_4 = ["src", "onClick"];
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ImageBrowserView",
  setup(__props) {
    useCssVars((_ctx) => {
      var _a, _b;
      return {
        "0b0e0cc2": unref(settings).data.settings.frontend.image_browser_columns,
        "4fde547c": (_b = (_a = unref(theme)) == null ? void 0 : _a.Card) == null ? void 0 : _b.color
      };
    });
    const global = useState();
    const settings = useSettings();
    const theme = inject(themeOverridesKey);
    const showDeleteModal = ref(false);
    const showImageModal = ref(false);
    const scrollComponent = ref(null);
    const imageLimit = ref(30);
    const itemFilter = ref("");
    const gridColumnRefs = ref([]);
    const imageSrc = computed(() => {
      const url = urlFromPath(global.state.imageBrowser.currentImage.path);
      return url;
    });
    function deleteImage() {
      const url = new URL(`${serverUrl}/api/outputs/delete/`);
      url.searchParams.append(
        "filename",
        global.state.imageBrowser.currentImage.path
      );
      fetch(url, { method: "DELETE" }).then((res) => res.json()).then(() => {
        showImageModal.value = false;
        const index = imgData.findIndex((el) => {
          return el.path === global.state.imageBrowser.currentImage.path;
        });
        imgData.splice(index, 1);
        global.state.imageBrowser.currentImage = {
          path: "",
          id: "",
          time: 0
        };
        global.state.imageBrowser.currentImageByte64 = "";
        global.state.imageBrowser.currentImageMetadata = {};
      });
    }
    function downloadImage() {
      const url = urlFromPath(global.state.imageBrowser.currentImage.path);
      fetch(url, { mode: "no-cors" }).then((res) => res.blob()).then((blob) => {
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
            console.error("base64data is null!");
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
            console.error("base64data is null!");
          }
        };
      });
    }
    const currentColumn = ref(0);
    const currentRowIndex = ref(0);
    function parseMetadataFromString(key, value) {
      value = value.trim();
      if (value.toLowerCase() === "true") {
        return true;
      } else if (value.toLowerCase() === "false") {
        return false;
      } else {
        if (isFinite(+value)) {
          return +value;
        } else {
          return value;
        }
      }
    }
    function imgClick(column_index, item_index) {
      currentRowIndex.value = item_index;
      currentColumn.value = column_index;
      const item = columns.value[column_index][item_index];
      global.state.imageBrowser.currentImage = item;
      setByte64FromImage(item.path);
      const url = new URL(`${serverUrl}/api/outputs/data/`);
      url.searchParams.append("filename", item.path);
      fetch(url).then((res) => res.json()).then((data) => {
        global.state.imageBrowser.currentImageMetadata = JSON.parse(
          JSON.stringify(data),
          (key, value) => {
            if (typeof value === "string") {
              return parseMetadataFromString(key, value);
            }
            return value;
          }
        );
      });
      showImageModal.value = true;
    }
    const imgData = reactive([]);
    const filteredImgData = computed(() => {
      return imgData.filter((item) => {
        if (itemFilter.value === "") {
          return true;
        }
        return item.path.includes(itemFilter.value);
      });
    });
    const computedImgDataLimit = computed(() => {
      return Math.min(filteredImgData.value.length, imageLimit.value);
    });
    const columns = computed(() => {
      const cols = [];
      for (let i = 0; i < settings.data.settings.frontend.image_browser_columns; i++) {
        cols.push([]);
      }
      for (let i = 0; i < computedImgDataLimit.value; i++) {
        cols[i % settings.data.settings.frontend.image_browser_columns].push(
          filteredImgData.value[i]
        );
      }
      return cols;
    });
    async function refreshImages() {
      imgData.splice(0, imgData.length);
      await fetch(`${serverUrl}/api/outputs/txt2img`).then((res) => res.json()).then((data) => {
        data.forEach((item) => {
          imgData.push(item);
        });
      });
      await fetch(`${serverUrl}/api/outputs/img2img`).then((res) => res.json()).then((data) => {
        data.forEach((item) => {
          imgData.push(item);
        });
      });
      await fetch(`${serverUrl}/api/outputs/extra`).then((res) => res.json()).then((data) => {
        data.forEach((item) => {
          imgData.push(item);
        });
      });
      imgData.sort((a, b) => {
        return b.time - a.time;
      });
    }
    const handleScroll = (e) => {
      let element = scrollComponent.value;
      if (element === null) {
        return;
      }
      let minBox = 0;
      for (const col of gridColumnRefs.value) {
        const lastImg = col.childNodes.item(
          col.childNodes.length - 2
        );
        const bottombbox = lastImg.getBoundingClientRect().bottom;
        if (minBox === 0) {
          minBox = bottombbox;
        } else if (bottombbox < minBox) {
          minBox = bottombbox;
        }
      }
      if (minBox - 50 < window.innerHeight) {
        if (imageLimit.value >= filteredImgData.value.length) {
          return;
        }
        imageLimit.value += 30;
      }
    };
    function moveImage(direction) {
      const numColumns = settings.data.settings.frontend.image_browser_columns;
      if (direction === -1) {
        if (currentColumn.value > 0) {
          imgClick(currentColumn.value - 1, currentRowIndex.value);
        } else {
          imgClick(numColumns - 1, currentRowIndex.value - 1);
        }
      } else if (direction === 1) {
        if (currentColumn.value < numColumns - 1) {
          imgClick(currentColumn.value + 1, currentRowIndex.value);
        } else {
          imgClick(0, currentRowIndex.value + 1);
        }
      }
    }
    onMounted(() => {
      window.addEventListener("scroll", handleScroll);
      window.addEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft") {
          moveImage(-1);
        } else if (e.key === "ArrowRight") {
          moveImage(1);
        }
      });
    });
    onUnmounted(() => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft") {
          moveImage(-1);
        } else if (e.key === "ArrowRight") {
          moveImage(1);
        }
      });
    });
    function getNamedSampler(value) {
      const parsed_string = +value;
      for (const objectKey of Object.keys(diffusersSchedulerTuple)) {
        const val = diffusersSchedulerTuple[objectKey];
        if (val === parsed_string) {
          return objectKey;
        }
      }
      return value;
    }
    refreshImages();
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", null, [
        createBaseVNode("div", _hoisted_1, [
          createVNode(unref(NInput), {
            value: itemFilter.value,
            "onUpdate:value": _cache[0] || (_cache[0] = ($event) => itemFilter.value = $event),
            style: { "margin": "0 12px" },
            placeholder: "Filter"
          }, null, 8, ["value"]),
          createVNode(unref(NIcon), {
            style: { "margin-right": "12px" },
            size: "22"
          }, {
            default: withCtx(() => [
              createVNode(unref(GridOutline))
            ]),
            _: 1
          }),
          createVNode(unref(NSlider), {
            style: { "width": "50vw" },
            min: 1,
            max: 10,
            value: unref(settings).data.settings.frontend.image_browser_columns,
            "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(settings).data.settings.frontend.image_browser_columns = $event)
          }, null, 8, ["value"])
        ]),
        createBaseVNode("div", _hoisted_2, [
          createVNode(unref(NModal), {
            show: showDeleteModal.value,
            "onUpdate:show": _cache[2] || (_cache[2] = ($event) => showDeleteModal.value = $event),
            "mask-closable": false,
            preset: "confirm",
            type: "error",
            title: "Delete Image",
            content: "Do you want to delete this image? This action cannot be undone.",
            "positive-text": "Confirm",
            "negative-text": "Cancel",
            "transform-origin": "center",
            onPositiveClick: deleteImage,
            onNegativeClick: _cache[3] || (_cache[3] = ($event) => showDeleteModal.value = false)
          }, null, 8, ["show"]),
          createVNode(unref(NModal), {
            show: showImageModal.value,
            "onUpdate:show": _cache[5] || (_cache[5] = ($event) => showImageModal.value = $event),
            closable: "",
            "mask-closable": "",
            "close-on-esc": "",
            preset: "card",
            style: { "width": "85vw" },
            title: "Image Info",
            id: "image-modal"
          }, {
            default: withCtx(() => [
              createVNode(unref(NGrid), {
                cols: "1 m:2",
                "x-gap": "12",
                "y-gap": "12",
                responsive: "screen"
              }, {
                default: withCtx(() => [
                  createVNode(unref(NGi), null, {
                    default: withCtx(() => [
                      createVNode(unref(NImage), {
                        src: imageSrc.value,
                        "object-fit": "contain",
                        style: { "width": "100%", "height": "auto", "justify-content": "center" },
                        "img-props": { style: { width: "40vw", maxHeight: "70vh" } }
                      }, null, 8, ["src"]),
                      createVNode(unref(NGrid), {
                        cols: "2",
                        "x-gap": "4",
                        "y-gap": "4",
                        style: { "margin-top": "12px" }
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
                                onClick: _cache[4] || (_cache[4] = ($event) => showDeleteModal.value = true),
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
                          }),
                          createVNode(unref(NGi), { span: "2" }, {
                            default: withCtx(() => [
                              createVNode(unref(_sfc_main$1), {
                                output: unref(global).state.imageBrowser.currentImageByte64,
                                card: false,
                                data: unref(global).state.imageBrowser.currentImageMetadata
                              }, null, 8, ["output", "data"])
                            ]),
                            _: 1
                          })
                        ]),
                        _: 1
                      })
                    ]),
                    _: 1
                  }),
                  createVNode(unref(NGi), null, {
                    default: withCtx(() => [
                      createVNode(unref(NScrollbar), null, {
                        default: withCtx(() => [
                          unref(global).state.imageBrowser.currentImageMetadata.size !== 0 ? (openBlock(), createBlock(unref(NDescriptions), {
                            key: 0,
                            column: 2,
                            size: "large"
                          }, {
                            default: withCtx(() => [
                              (openBlock(true), createElementBlock(Fragment, null, renderList(unref(global).state.imageBrowser.currentImageMetadata, (item, key) => {
                                return openBlock(), createBlock(unref(NDescriptionsItem), {
                                  label: unref(convertToTextString)(key.toString()),
                                  "content-style": "max-width: 100px; word-wrap: break-word;",
                                  style: { "margin": "4px" },
                                  key: item.toString()
                                }, {
                                  default: withCtx(() => [
                                    createTextVNode(toDisplayString(key.toString() === "scheduler" ? getNamedSampler(item.toString()) : item), 1)
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
                ]),
                _: 1
              })
            ]),
            _: 1
          }, 8, ["show"]),
          createBaseVNode("div", {
            ref_key: "scrollComponent",
            ref: scrollComponent
          }, [
            createBaseVNode("div", _hoisted_3, [
              (openBlock(true), createElementBlock(Fragment, null, renderList(columns.value, (column, column_index) => {
                return openBlock(), createElementBlock("div", {
                  key: column_index,
                  class: "image-column",
                  ref_for: true,
                  ref_key: "gridColumnRefs",
                  ref: gridColumnRefs
                }, [
                  (openBlock(true), createElementBlock(Fragment, null, renderList(column, (item, item_index) => {
                    return openBlock(), createElementBlock("img", {
                      src: unref(urlFromPath)(item.path),
                      key: item_index,
                      style: { "width": "100%", "height": "auto", "border-radius": "8px", "cursor": "pointer", "margin-bottom": "6px" },
                      onClick: ($event) => imgClick(column_index, item_index)
                    }, null, 8, _hoisted_4);
                  }), 128))
                ]);
              }), 128))
            ])
          ], 512)
        ])
      ]);
    };
  }
});
const ImageBrowserView_vue_vue_type_style_index_0_scoped_c3083c91_lang = "";
const ImageBrowserView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-c3083c91"]]);
export {
  ImageBrowserView as default
};
