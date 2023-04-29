import { _ as _sfc_main$1 } from "./GenerateSection.vue_vue_type_script_setup_true_lang.js";
import { I as ImageUpload } from "./ImageUpload.js";
import { d as defineComponent, u as useState, a as useSettings, b as useMessage, A as ref, c as computed, o as openBlock, e as createElementBlock, f as createVNode, w as withCtx, g as unref, N as NGi, h as NCard, i as NSpace, l as createBaseVNode, n as NSelect, m as NTooltip, k as createTextVNode, j as NInput, br as isRef, t as toDisplayString, r as NGrid, v as serverUrl, s as spaceRegex, x as pushScopeId, y as popScopeId, _ as _export_sfc } from "./index.js";
import { v as v4 } from "./v4.js";
import { N as NSlider } from "./Slider.js";
import { N as NInputNumber } from "./InputNumber.js";
import { N as NSwitch } from "./Switch.js";
const _withScopeId = (n) => (pushScopeId("data-v-eb4929f6"), n = n(), popScopeId(), n);
const _hoisted_1 = { class: "main-container" };
const _hoisted_2 = { class: "flex-container" };
const _hoisted_3 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Sampler", -1));
const _hoisted_4 = { class: "flex-container" };
const _hoisted_5 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Treshold", -1));
const _hoisted_6 = { class: "flex-container" };
const _hoisted_7 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Weighted", -1));
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TaggerView",
  setup(__props) {
    const global = useState();
    const conf = useSettings();
    const messageHandler = useMessage();
    const imageSelectCallback = (base64Image) => {
      conf.data.settings.tagger.image = base64Image;
    };
    const generate = () => {
      global.state.generating = true;
      fetch(`${serverUrl}/api/generate/interrogate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          data: {
            image: conf.data.settings.tagger.image,
            id: v4(),
            strength: conf.data.settings.tagger.treshold
          },
          model: conf.data.settings.tagger.model
        })
      }).then((res) => {
        if (!res.ok) {
          throw new Error(res.statusText);
        }
        global.state.generating = false;
        res.json().then(
          (data) => {
            global.state.tagger.positivePrompt = data.positive;
            global.state.tagger.negativePrompt = data.negative;
            console.log(data);
          }
        );
      }).catch((err) => {
        global.state.generating = false;
        messageHandler.error(err);
        console.log(err);
      });
    };
    const weighted = ref(false);
    function MapToPrompt(map) {
      if (weighted.value) {
        let weightedPrompt = Array();
        for (const [key, value] of map) {
          if (value.toFixed(2) === "1.00") {
            weightedPrompt.push(`${key}`);
            continue;
          } else {
            weightedPrompt.push(`(${key}: ${value.toFixed(2)})`);
          }
        }
        return weightedPrompt.join(", ");
      } else {
        let prompt = Array();
        for (const [key, value] of map) {
          prompt.push(key);
        }
        return prompt.join(", ");
      }
    }
    const computedPrompt = computed(() => {
      const sortedMap = new Map(
        [...global.state.tagger.positivePrompt].sort((a, b) => b[1] - a[1])
      );
      return MapToPrompt(sortedMap);
    });
    const computedNegativePrompt = computed(() => {
      const sortedMap = new Map(
        [...global.state.tagger.negativePrompt].sort((a, b) => b[1] - a[1])
      );
      return MapToPrompt(sortedMap);
    });
    const promptCount = computed(() => {
      return computedPrompt.value.split(spaceRegex).length - 1;
    });
    const negativePromptCount = computed(() => {
      return computedNegativePrompt.value.split(spaceRegex).length - 1;
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NGrid), {
          cols: "1 m:2",
          "x-gap": "12",
          responsive: "screen"
        }, {
          default: withCtx(() => [
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(ImageUpload, {
                  callback: imageSelectCallback,
                  preview: unref(conf).data.settings.tagger.image,
                  style: { "margin-bottom": "12px" },
                  onFileDropped: _cache[0] || (_cache[0] = ($event) => unref(conf).data.settings.tagger.image = $event)
                }, null, 8, ["preview"]),
                createVNode(unref(NCard), { title: "Settings" }, {
                  default: withCtx(() => [
                    createVNode(unref(NSpace), {
                      vertical: "",
                      class: "left-container"
                    }, {
                      default: withCtx(() => [
                        createBaseVNode("div", _hoisted_2, [
                          _hoisted_3,
                          createVNode(unref(NSelect), {
                            options: [
                              {
                                label: "Deepdanbooru",
                                value: "deepdanbooru"
                              },
                              {
                                label: "CLIP",
                                value: "clip"
                              },
                              {
                                label: "Flamingo",
                                value: "flamingo"
                              }
                            ],
                            value: unref(conf).data.settings.tagger.model,
                            "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(conf).data.settings.tagger.model = $event),
                            style: { "flex-grow": "1" }
                          }, null, 8, ["value"])
                        ]),
                        createBaseVNode("div", _hoisted_4, [
                          createVNode(unref(NTooltip), { style: { "max-width": "600px" } }, {
                            trigger: withCtx(() => [
                              _hoisted_5
                            ]),
                            default: withCtx(() => [
                              createTextVNode(" Confidence treshold of the model. The higher the value, the more tokens will be given to you. ")
                            ]),
                            _: 1
                          }),
                          createVNode(unref(NSlider), {
                            value: unref(conf).data.settings.tagger.treshold,
                            "onUpdate:value": _cache[2] || (_cache[2] = ($event) => unref(conf).data.settings.tagger.treshold = $event),
                            min: 0.1,
                            max: 1,
                            style: { "margin-right": "12px" },
                            step: 0.025
                          }, null, 8, ["value", "min", "step"]),
                          createVNode(unref(NInputNumber), {
                            value: unref(conf).data.settings.tagger.treshold,
                            "onUpdate:value": _cache[3] || (_cache[3] = ($event) => unref(conf).data.settings.tagger.treshold = $event),
                            size: "small",
                            style: { "min-width": "96px", "width": "96px" },
                            min: 0.1,
                            max: 1,
                            step: 0.025
                          }, null, 8, ["value", "min", "step"])
                        ])
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
                createVNode(_sfc_main$1, { generate }),
                createVNode(unref(NCard), null, {
                  default: withCtx(() => [
                    createBaseVNode("div", _hoisted_6, [
                      _hoisted_7,
                      createVNode(unref(NSwitch), {
                        value: weighted.value,
                        "onUpdate:value": _cache[4] || (_cache[4] = ($event) => weighted.value = $event)
                      }, null, 8, ["value"])
                    ]),
                    createVNode(unref(NInput), {
                      value: unref(computedPrompt),
                      "onUpdate:value": _cache[5] || (_cache[5] = ($event) => isRef(computedPrompt) ? computedPrompt.value = $event : null),
                      type: "textarea",
                      placeholder: "Prompt",
                      "show-count": ""
                    }, {
                      count: withCtx(() => [
                        createTextVNode(toDisplayString(unref(promptCount)), 1)
                      ]),
                      _: 1
                    }, 8, ["value"]),
                    createVNode(unref(NInput), {
                      value: unref(computedNegativePrompt),
                      "onUpdate:value": _cache[6] || (_cache[6] = ($event) => isRef(computedNegativePrompt) ? computedNegativePrompt.value = $event : null),
                      type: "textarea",
                      placeholder: "Negative prompt",
                      "show-count": ""
                    }, {
                      count: withCtx(() => [
                        createTextVNode(toDisplayString(unref(negativePromptCount)), 1)
                      ]),
                      _: 1
                    }, 8, ["value"])
                  ]),
                  _: 1
                })
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
const TaggerView_vue_vue_type_style_index_0_scoped_eb4929f6_lang = "";
const TaggerView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-eb4929f6"]]);
export {
  TaggerView as default
};
