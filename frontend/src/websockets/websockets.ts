import type { Store, _UnwrapAll } from "pinia";
import type { StateInterface } from "../store/state";

export interface NotificationMessage {
  severity: "success" | "info" | "warning" | "error";
  title: string;
  timestamp: string;
  message: string;
  timeout: number;
  id: number;
}

export interface WebSocketMessage {
  type: string;
  data: any;
}

export function processWebSocket(
  message: WebSocketMessage,
  global: Store<
    "state",
    _UnwrapAll<
      Pick<
        {
          state: StateInterface;
        },
        "state"
      >
    >,
    Pick<
      {
        state: StateInterface;
      },
      never
    >,
    Pick<
      {
        state: StateInterface;
      },
      never
    >
  >
): void {
  switch (message.type) {
    case "test": {
      console.log(message.data);
      break;
    }
    case "progress": {
      global.state.progress = message.data.progress;
      break;
    }
    case "txt2img": {
      console.log(message.data);
      global.state.txt2img.currentImage = message.data.image;
      global.state.progress = message.data.progress;
    }
  }
}
