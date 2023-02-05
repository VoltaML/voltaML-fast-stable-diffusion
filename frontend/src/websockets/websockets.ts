import type { NotificationApiInjection } from "naive-ui/es/notification/src/NotificationProvider";
import type { Store, _UnwrapAll } from "pinia";
import type { StateInterface } from "../store/state";

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
  >,
  notificationProvider: NotificationApiInjection
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
      global.state.txt2img.currentImage = message.data.image
        ? message.data.image
        : global.state.txt2img.currentImage;
      global.state.progress = message.data.progress;
      global.state.current_step = message.data.current_step;
      global.state.total_steps = message.data.total_steps;
      break;
    }
    case "img2img": {
      console.log(message.data);
      global.state.img2img.currentImage = message.data.image
        ? message.data.image
        : global.state.img2img.currentImage;
      global.state.progress = message.data.progress;
      global.state.current_step = message.data.current_step;
      global.state.total_steps = message.data.total_steps;
      break;
    }
    case "notification": {
      message.data.timeout = message.data.timeout || 5000;

      notificationProvider.create({
        type: message.data.severity,
        title: message.data.title,
        content: message.data.message,
        duration: message.data.timeout,
      });
    }
  }
}