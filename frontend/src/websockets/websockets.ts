import type { NotificationApiInjection } from "naive-ui/es/notification/src/NotificationProvider";
import type { Store, _UnwrapAll } from "pinia";
import type { StateInterface } from "../store/state";

export interface WebSocketMessage {
  type: string;
  data: any;
}

function progressForward(
  progress: number,
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
    case "image_variations": {
      global.state.imageVariations.currentImage = message.data.image
        ? message.data.image
        : global.state.imageVariations.currentImage;
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
    case "cluster_stats": {
      global.state.perf_drawer.gpus = message.data;
      break;
    }
    default: {
      console.log(message);
    }
  }
}
