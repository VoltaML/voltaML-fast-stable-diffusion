<template>
  <div class="main-container">
    <NInput
      value="nsfw, (worst quality:2.0), (low quality:1.4), (bad-hands:0.2), easynegative, verybadimagenegative_v1.3, badhandv4, ng_deepnegative_v1_75t"
      @keyup="handleKeyup"
      @keydown="handleKeydown"
    />
    <input
      type="text"
      @keyup="handleKeyup"
      @keydown="handleKeydown"
      style="width: 100%"
      value="nsfw, (worst quality:2.0), (low quality:1.4), (bad-hands:0.2), easynegative, verybadimagenegative_v1.3, badhandv4, ng_deepnegative_v1_75t"
    />
  </div>
</template>

<script setup lang="ts">
import { NInput } from "naive-ui";

function getText(elem: HTMLInputElement | null) {
  if (elem === null) {
    console.log("Element is null");
    return [0, 0];
  }
  if (elem.tagName === "INPUT" && elem.type === "text") {
    console.log("Selection:", elem.selectionStart, elem.selectionEnd);
    return [
      elem.selectionStart === null ? 0 : elem.selectionStart,
      elem.selectionEnd === null ? 0 : elem.selectionEnd,
    ];
  }
  console.log("Element is not input");
  return [0, 0];
}

function handleKeyup(e: KeyboardEvent) {
  const values = getText(document.activeElement as HTMLInputElement);
  const boundaryIndexStart = values[0];
  const boundaryIndexEnd = values[1];

  // Handle ArrowUp
  if (e.key === "ArrowUp") {
    e.preventDefault();
    const elem = document.activeElement as HTMLInputElement;
    const current_selection = elem.value.substring(
      boundaryIndexStart,
      boundaryIndexEnd
    );

    const regex = /\(([^:]+([:]?[\s]?)([\d.\d]+))\)/;
    const matches = regex.exec(current_selection);

    // Check for value inside if there are parentheses
    if (matches) {
      // We are looking for 1.1 in (selected_text: 1.1), use regex, if there is a value, increment it by 0.1
      if (matches) {
        const value = parseFloat(matches[3]);
        const new_value = (value + 0.1).toFixed(1);

        // Wipe out the things between boundaries and replace it with the new value
        const beforeString = elem.value.substring(0, boundaryIndexStart);
        const afterString = elem.value.substring(boundaryIndexEnd + 1);

        console.log("Before:", beforeString, "After:", afterString);

        const newString = `${beforeString}${current_selection.replace(
          matches[3],
          new_value
        )}${afterString}`;

        console.log("New string", newString);

        elem.value = newString;

        // Set the hightlight as it was lost after the value was changed
        elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd);
      }
    } else if (boundaryIndexStart !== boundaryIndexEnd) {
      // Element has no parentheses, insert them and set value to 1.1
      const new_inner_string = `(${current_selection}:1.1)`;
      const beforeString = elem.value.substring(0, boundaryIndexStart);
      const afterString = elem.value.substring(boundaryIndexEnd + 1);
      elem.value = `${beforeString}${new_inner_string}${afterString}`;

      // Set the hightlight as it was lost after the value was changed
      elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd + 6);
    } else {
      console.log("No selection, cannot parse for weighting");
    }
  }

  // Handle ArrowDown
  if (e.key === "ArrowDown") {
    e.preventDefault();
    const elem = document.activeElement as HTMLInputElement;
    const current_selection = elem.value.substring(
      boundaryIndexStart,
      boundaryIndexEnd
    );

    const regex = /\(([^:]+([:]?[\s]?)([\d.\d]+))\)/;
    const matches = regex.exec(current_selection);

    // Check for value inside if there are parentheses
    if (matches) {
      // We are looking for 0.9 in (selected_text: 0.9), use regex, if there is a value, decrement it by 0.1
      if (matches) {
        const value = parseFloat(matches[3]);
        const new_value = Math.max(value - 0.1, 0).toFixed(1);

        // Wipe out the things between boundaries and replace it with the new value
        const beforeString = elem.value.substring(0, boundaryIndexStart);
        const afterString = elem.value.substring(boundaryIndexEnd + 1);

        console.log("Before:", beforeString, "After:", afterString);

        const newString = `${beforeString}${current_selection.replace(
          matches[3],
          new_value
        )}${afterString}`;

        console.log("New string", newString);

        elem.value = newString;

        // Set the hightlight as it was lost after the value was changed
        elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd);
      }
    } else if (boundaryIndexStart !== boundaryIndexEnd) {
      // Element has no parentheses, insert them and set value to 0.9
      const new_inner_string = `(${current_selection}:0.9)`;
      const beforeString = elem.value.substring(0, boundaryIndexStart);
      const afterString = elem.value.substring(boundaryIndexEnd + 1);
      elem.value = `${beforeString}${new_inner_string}${afterString}`;

      // Set the hightlight as it was lost after the value was changed
      elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd + 6);
    } else {
      console.log("No selection, cannot parse for weighting");
    }
  }
}

function handleKeydown(e: KeyboardEvent) {
  // Prevent arrow keys from moving the cursor
  var arrowKeys = [37, 38, 39, 40];

  if (arrowKeys.includes(e.keyCode)) {
    e.preventDefault();
  }
}
</script>
