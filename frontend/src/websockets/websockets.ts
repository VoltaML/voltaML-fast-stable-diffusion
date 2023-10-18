import type { NotificationApiInjection } from "naive-ui/es/notification/src/NotificationProvider";

export interface WebSocketMessage {
  type: string;
  data: any;
}

function progressForward(
  progress: number,
  global: ReturnType<typeof import("@/store/state")["useState"]>
) {
  if (progress === 0) {
    return 0;
  } else if (global.state.progress <= progress) {
    return progress;
  } else {
    return global.state.progress;
  }
}

function currentStepForward(
  currentStep: number,
  global: ReturnType<typeof import("@/store/state")["useState"]>
) {
  if (currentStep === 0) {
    return 0;
  } else if (global.state.current_step <= currentStep) {
    return currentStep;
  } else {
    return global.state.current_step;
  }
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
      global.state.progress = progressForward(message.data.progress, global);
      global.state.current_step = currentStepForward(
        message.data.current_step,
        global
      );
      global.state.total_steps = message.data.total_steps;
      break;
    }
    case "img2img": {
      global.state.img2img.currentImage = message.data.image
        ? message.data.image
        : global.state.img2img.currentImage;
      global.state.progress = progressForward(message.data.progress, global);
      global.state.current_step = currentStepForward(
        message.data.current_step,
        global
      );
      global.state.total_steps = message.data.total_steps;
      break;
    }
    case "inpainting": {
      global.state.inpainting.currentImage = message.data.image
        ? message.data.image
        : global.state.inpainting.currentImage;
      global.state.progress = progressForward(message.data.progress, global);
      global.state.current_step = currentStepForward(
        message.data.current_step,
        global
      );
      global.state.total_steps = message.data.total_steps;
      break;
    }
    case "controlnet": {
      global.state.controlnet.currentImage = message.data.image
        ? message.data.image
        : global.state.controlnet.currentImage;
      global.state.progress = progressForward(message.data.progress, global);
      global.state.current_step = currentStepForward(
        message.data.current_step,
        global
      );
      global.state.total_steps = message.data.total_steps;
      break;
    }
    case "notification": {
      message.data.timeout = message.data.timeout || 5000;

      console.log(message.data.message);

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
    default: {
      console.log(message);
    }
  }
}
