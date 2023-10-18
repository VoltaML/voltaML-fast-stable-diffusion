import { serverUrl } from "./env";
import { useWebsocket } from "./store/websockets";

export const spaceRegex = new RegExp("[\\s,]+");
const arrowKeys = [38, 40];
let currentFocus: number = -1;

export function dimensionValidator(value: number) {
  return value % 8 === 0;
}

export function convertToTextString(str: string): string {
  const upper = str.charAt(0).toUpperCase() + str.slice(1);
  return upper.replace(/_/g, " ");
}

function addActive(x: any) {
  if (!x) return false;
  removeActive(x);

  if (currentFocus >= x.length) {
    currentFocus = 0;
  }
  if (currentFocus < 0) {
    currentFocus = x.length - 1;
  }

  x[currentFocus].classList.add("autocomplete-active");
}
function removeActive(x: any) {
  for (let i = 0; i < x.length; i++) {
    x[i].classList.remove("autocomplete-active");
  }
}
function closeAllLists(
  elmnt: HTMLElement | undefined | EventTarget | null,
  input: any
) {
  const x = document.getElementsByClassName("autocomplete-items");
  for (let i = 0; i < x.length; i++) {
    if (elmnt != x[i] && elmnt != input) {
      x[i]?.parentNode?.removeChild(x[i]);
    }
  }
}

export async function startWebsocket(messageProvider: any) {
  const websocketState = useWebsocket();

  const timeout = 1000;

  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  const response = await fetch(`${serverUrl}/api/test/alive`, {
    signal: controller.signal,
  }).catch(() => {
    messageProvider.error("Server is not responding");
  });
  clearTimeout(id);

  if (response === undefined) {
    return;
  }

  if (response.status !== 200) {
    messageProvider.error("Server is not responding");
    return;
  }

  console.log("Starting websocket");
  websocketState.ws_open();
}

export function getTextBoundaries(elem: HTMLInputElement | null) {
  if (elem === null) {
    console.log("Element is null");
    return [0, 0];
  }
  if (
    (elem.tagName === "INPUT" && elem.type === "text") ||
    elem.tagName === "TEXTAREA"
  ) {
    return [
      elem.selectionStart === null ? 0 : elem.selectionStart,
      elem.selectionEnd === null ? 0 : elem.selectionEnd,
    ];
  }
  console.log("Element is not input");
  return [0, 0];
}

export function promptHandleKeyUp(
  e: KeyboardEvent,
  data: any,
  key: string,
  globalState: ReturnType<typeof import("@/store/state")["useState"]>
) {
  // Handle ArrowUp
  if (e.key === "ArrowUp" && e.ctrlKey) {
    const values = getTextBoundaries(
      document.activeElement as HTMLInputElement
    );
    const boundaryIndexStart = values[0];
    const boundaryIndexEnd = values[1];

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
        const afterString = elem.value.substring(boundaryIndexEnd);

        const newString = `${beforeString}${current_selection.replace(
          matches[3],
          new_value
        )}${afterString}`;

        elem.value = newString;
        data[key] = newString;

        // Set the hightlight as it was lost after the value was changed
        elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd);
      }
    } else if (boundaryIndexStart !== boundaryIndexEnd) {
      // Element has no parentheses, insert them and set value to 1.1
      const new_inner_string = `(${current_selection}:1.1)`;
      const beforeString = elem.value.substring(0, boundaryIndexStart);
      const afterString = elem.value.substring(boundaryIndexEnd);

      elem.value = `${beforeString}${new_inner_string}${afterString}`;
      data[key] = `${beforeString}${new_inner_string}${afterString}`;

      // Set the hightlight as it was lost after the value was changed
      elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd + 6);
    } else {
      console.log("No selection, cannot parse for weighting");
    }
  }

  // Handle ArrowDown
  if (e.key === "ArrowDown" && e.ctrlKey) {
    const values = getTextBoundaries(
      document.activeElement as HTMLInputElement
    );
    const boundaryIndexStart = values[0];
    const boundaryIndexEnd = values[1];

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
        const afterString = elem.value.substring(boundaryIndexEnd);

        const newString = `${beforeString}${current_selection.replace(
          matches[3],
          new_value
        )}${afterString}`;

        elem.value = newString;
        data[key] = newString;

        // Set the hightlight as it was lost after the value was changed
        elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd);
      }
    } else if (boundaryIndexStart !== boundaryIndexEnd) {
      // Element has no parentheses, insert them and set value to 0.9
      const new_inner_string = `(${current_selection}:0.9)`;
      const beforeString = elem.value.substring(0, boundaryIndexStart);
      const afterString = elem.value.substring(boundaryIndexEnd);

      elem.value = `${beforeString}${new_inner_string}${afterString}`;
      data[key] = `${beforeString}${new_inner_string}${afterString}`;

      // Set the hightlight as it was lost after the value was changed
      elem.setSelectionRange(boundaryIndexStart, boundaryIndexEnd + 6);
    } else {
      console.log("No selection, cannot parse for weighting");
    }
  }

  // Handle autocomplete
  const input = e.target as HTMLTextAreaElement;

  if (input) {
    const text = input.value;
    const currentTokenStripped = text.split(",").pop()?.trim();

    // close any already open lists of autocompleted values
    closeAllLists(undefined, input);

    if (!currentTokenStripped) {
      return false;
    }

    const toAppend = [];

    // Special autocomplete for lora and similiar
    for (let i = 0; i < globalState.state.autofill_special.length; i++) {
      if (
        globalState.state.autofill_special[i]
          .toLowerCase()
          .includes(currentTokenStripped.toLowerCase())
      ) {
        const b = document.createElement("DIV");
        b.innerText = globalState.state.autofill_special[i];
        b.innerHTML +=
          "<input type='hidden' value='" +
          globalState.state.autofill_special[i] +
          "'>";
        b.addEventListener("click", function () {
          input.value =
            text.substring(0, text.lastIndexOf(",") + 1) +
            globalState.state.autofill_special[i];
          data[key] = input.value;

          closeAllLists(undefined, input);
        });
        toAppend.push(b);
      }
    }

    // Standard autocomplete
    for (let i = 0; i < globalState.state.autofill.length; i++) {
      if (
        globalState.state.autofill[i]
          .toLowerCase()
          .includes(currentTokenStripped.toLowerCase())
      ) {
        if (toAppend.length >= 30) {
          break;
        }

        const b = document.createElement("DIV");
        b.innerText = globalState.state.autofill[i];
        b.innerHTML +=
          "<input type='hidden' value='" + globalState.state.autofill[i] + "'>";
        b.addEventListener("click", function () {
          input.value =
            text.substring(0, text.lastIndexOf(",") + 1) +
            globalState.state.autofill[i];
          data[key] = input.value;

          closeAllLists(undefined, input);
        });
        toAppend.push(b);
      }
    }

    if (toAppend.length === 0) {
      return false;
    }

    const div = document.createElement("DIV");
    div.setAttribute("id", "autocomplete-list");
    div.setAttribute("class", "autocomplete-items");
    input.parentNode?.parentNode?.parentNode?.parentNode?.appendChild(div);
    for (let i = 0; i < toAppend.length; i++) {
      div.appendChild(toAppend[i]);
    }

    // Handle Special keys
    const autocompleteList = document.getElementById("autocomplete-list");
    const x = autocompleteList?.getElementsByTagName("div");
    if (e.key === "ArrowDown") {
      currentFocus++;
      addActive(x);
      e.preventDefault();
    } else if (e.key === "ArrowUp") {
      currentFocus--;
      addActive(x);
      e.preventDefault();
    } else if (e.key === "Enter" || e.key === "Tab") {
      e.stopImmediatePropagation();
      e.preventDefault();
      if (currentFocus > -1) {
        if (x) x[currentFocus].click();
      }
    } else if (e.key === "Escape") {
      closeAllLists(undefined, input);
    }
  }
}

export function promptHandleKeyDown(e: KeyboardEvent) {
  // Prevent arrow keys from moving the cursor

  if (arrowKeys.includes(e.keyCode) && e.ctrlKey) {
    e.preventDefault();
  }

  if (document.getElementById("autocomplete-list")) {
    if (
      e.key === "Enter" ||
      e.key === "Tab" ||
      e.key === "ArrowDown" ||
      e.key === "ArrowUp"
    ) {
      e.preventDefault();
    }
  }
}

export function urlFromPath(path: string) {
  const url = new URL(path, serverUrl);
  return url.href;
}
