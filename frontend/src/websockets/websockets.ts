import type { NotificationApiInjection } from "naive-ui/es/notification/src/NotificationProvider";

export interface WebSocketMessage {
  type: string;
  data: any;
}

export function processWebSocket(
  message: WebSocketMessage,
  global: ReturnType<typeof import("@/store/state")["useState"]>,
  notificationProvider: NotificationApiInjection
): void {
  switch (message.type) {
    case "test": {
      break;
    }
    case "progress": {
      global.state.progress = message.data.progress;
      break;
    }
    case "txt2img": {
      global.state.txt2img.currentImage = message.data.image
        ? message.data.image
        : global.state.txt2img.currentImage;
      global.state.progress = message.data.progress;
      global.state.current_step = message.data.current_step;
      global.state.total_steps = message.data.total_steps;
      break;
    }
    case "img2img": {
      global.state.img2img.currentImage = message.data.image
        ? message.data.image
        : global.state.img2img.currentImage;
      global.state.progress = message.data.progress;
      global.state.current_step = message.data.current_step;
      global.state.total_steps = message.data.total_steps;
      break;
    }
    case "inpainting": {
      global.state.inpainting.currentImage = message.data.image
        ? message.data.image
        : global.state.inpainting.currentImage;
      global.state.progress = message.data.progress;
      global.state.current_step = message.data.current_step;
      global.state.total_steps = message.data.total_steps;
      break;
    }
    case "controlnet": {
      global.state.controlnet.currentImage = message.data.image
        ? message.data.image
        : global.state.controlnet.currentImage;
      global.state.progress = message.data.progress;
      global.state.current_step = message.data.current_step;
      global.state.total_steps = message.data.total_steps;
      break;
    }
    case "notification": {
      console.log(message.data.message);

      if (message.data.timeout === 0) {
        message.data.timeout = null;
      }

      notificationProvider.create({
        type: message.data.severity,
        title: message.data.title,
        content: message.data.message,
        duration: message.data.timeout,
      });
      break;
    }
    case "aitemplate_compile": {
      global.state.aitBuildStep = {
        ...global.state.aitBuildStep,
        ...message.data,
      };
      break;
    }
    case "onnx_compile": {
      global.state.onnxBuildStep = {
        ...global.state.onnxBuildStep,
        ...message.data,
      };
      break;
    }
    case "cluster_stats": {
      global.state.perf_drawer.gpus = message.data;
      break;
    }
    case "token": {
      if (message.data.huggingface === "missing") {
        global.state.secrets.huggingface = "missing";
      }
      break;
    }
    case "refresh_capabilities": {
      global
        .fetchCapabilites()
        .then(() => {
          console.log("Capabilities refreshed");
        })
        .catch((error) => {
          console.error(error);
        });
      break;
    }
    case "log": {
      const messages = message.data.message.split("\n");
      for (const msg of messages) {
        global.state.log_drawer.logs.splice(0, 0, msg);

        // Limit the number of logs to 500
        if (global.state.log_drawer.logs.length > 500) {
          global.state.log_drawer.logs.pop();
        }
      }
      break;
    }
    case "incorrect_settings_value": {
      global.state.settings_diff.default_value = message.data.default_value;
      global.state.settings_diff.current_value = message.data.current_value;
      global.state.settings_diff.key = message.data.key;
      global.state.settings_diff.active = true;
      break;
    }
    default: {
      console.log(message);
    }
  }
}
