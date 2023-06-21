import { serverUrl } from "./env";
import { useWebsocket } from "./store/websockets";

export const spaceRegex = new RegExp("[\\s,]+");
const arrowKeys = [38, 40];

export function dimensionValidator(value: number) {
  return value % 8 === 0;
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

export function promptHandleKeyUp(e: KeyboardEvent, data: any, key: string) {
  console.log(data);

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
}

export function promptHandleKeyDown(e: KeyboardEvent) {
  // Prevent arrow keys from moving the cursor

  if (arrowKeys.includes(e.keyCode)) {
    e.preventDefault();
  }
}
