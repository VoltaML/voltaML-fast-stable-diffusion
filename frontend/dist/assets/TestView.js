import { d as defineComponent, e as openBlock, f as createElementBlock, g as createVNode, h as unref, n as createBaseVNode, k as NInput } from "./index.js";
const _hoisted_1 = { class: "main-container" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TestView",
  setup(__props) {
    function getText(elem) {
      if (elem === null) {
        console.log("Element is null");
        return [0, 0];
      }
      if (elem.tagName === "INPUT" && elem.type === "text") {
        console.log("Selection:", elem.selectionStart, elem.selectionEnd);
        return [
          elem.selectionStart === null ? 0 : elem.selectionStart,
          elem.selectionEnd === null ? 0 : elem.selectionEnd
        ];
      }
      console.log("Element is not input");
      return [0, 0];
    }
    function handleKeyup(e) {
      const values = getText(document.activeElement);
      const boundaryIndexStart = values[0];
      const boundaryIndexEnd = values[1];
      if (e.key === "ArrowUp") {
        e.preventDefault();
        const elem = document.activeElement;
        const current_selection = elem.value.substring(
          boundaryIndexStart,
          boundaryIndexEnd
        );
        const regex = /\(([^:]+([:]?[\s]?)([\d.\d]+))\)/;
        const matches = regex.exec(current_selection);
        if (matches) {
          if (matches) {
            const value = parseFloat(matches[3]);
            const new_value = (value + 0.1).toFixed(1);
            const beforeString = elem.value.substring(0, boundaryIndexStart);
            const afterString = elem.value.substring(boundaryIndexEnd + 1);
            console.log("Before:", beforeString, "After:", afterString);
            const newString = `${beforeString}${current_selection.replace(
              matches[3],
              new_value
            )}${afterString}`;
            console.log("New string", newString);
            elem.value = newString;
            elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd);
          }
        } else if (boundaryIndexStart !== boundaryIndexEnd) {
          const new_inner_string = `(${current_selection}:1.1)`;
          const beforeString = elem.value.substring(0, boundaryIndexStart);
          const afterString = elem.value.substring(boundaryIndexEnd + 1);
          elem.value = `${beforeString}${new_inner_string}${afterString}`;
          elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd + 6);
        } else {
          console.log("No selection, cannot parse for weighting");
        }
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        const elem = document.activeElement;
        const current_selection = elem.value.substring(
          boundaryIndexStart,
          boundaryIndexEnd
        );
        const regex = /\(([^:]+([:]?[\s]?)([\d.\d]+))\)/;
        const matches = regex.exec(current_selection);
        if (matches) {
          if (matches) {
            const value = parseFloat(matches[3]);
            const new_value = Math.max(value - 0.1, 0).toFixed(1);
            const beforeString = elem.value.substring(0, boundaryIndexStart);
            const afterString = elem.value.substring(boundaryIndexEnd + 1);
            console.log("Before:", beforeString, "After:", afterString);
            const newString = `${beforeString}${current_selection.replace(
              matches[3],
              new_value
            )}${afterString}`;
            console.log("New string", newString);
            elem.value = newString;
            elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd);
          }
        } else if (boundaryIndexStart !== boundaryIndexEnd) {
          const new_inner_string = `(${current_selection}:0.9)`;
          const beforeString = elem.value.substring(0, boundaryIndexStart);
          const afterString = elem.value.substring(boundaryIndexEnd + 1);
          elem.value = `${beforeString}${new_inner_string}${afterString}`;
          elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd + 6);
        } else {
          console.log("No selection, cannot parse for weighting");
        }
      }
    }
    function handleKeydown(e) {
      var arrowKeys = [37, 38, 39, 40];
      if (arrowKeys.includes(e.keyCode)) {
        e.preventDefault();
      }
    }
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NInput), {
          value: "nsfw, (worst quality:2.0), (low quality:1.4), (bad-hands:0.2), easynegative, verybadimagenegative_v1.3, badhandv4, ng_deepnegative_v1_75t",
          onKeyup: handleKeyup,
          onKeydown: handleKeydown
        }),
        createBaseVNode("input", {
          type: "text",
          onKeyup: handleKeyup,
          onKeydown: handleKeydown,
          style: { "width": "100%" },
          value: "nsfw, (worst quality:2.0), (low quality:1.4), (bad-hands:0.2), easynegative, verybadimagenegative_v1.3, badhandv4, ng_deepnegative_v1_75t"
        }, null, 32)
      ]);
    };
  }
});
export {
  _sfc_main as default
};
