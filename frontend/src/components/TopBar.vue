<template>
  <div class="top-bar">
    <NSelect
      style="max-width: 250px; padding-left: 12px; padding-right: 12px"
      :options="generatedModelOptions"
      @update:value="onModelChange"
      :loading="modelsLoading"
      placeholder=""
      :value="
        conf.data.settings.model !== null ? conf.data.settings.model?.name : ''
      "
      :consistent-menu-width="false"
      filterable
    />
    <NButton
      @click="showModal = true"
      :loading="modelsLoading"
      :type="conf.data.settings.model ? 'default' : 'success'"
      >Load Model</NButton
    >
    <NModal
      v-model:show="showModal"
      closable
      mask-closable
      preset="card"
      style="width: 85vw"
      title="Models"
      :auto-focus="false"
    >
      <div v-if="websocketState.readyState === 'CLOSED'">
        <NResult
          title="You are not connected to the server"
          description="Click the button below to reconnect"
          style="
            height: 70vh;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
          "
          status="500"
        >
          <template #footer>
            <NButton type="success" @click="startWebsocket(message)"
              >Reconnect</NButton
            >
          </template>
        </NResult>
      </div>
      <div v-else-if="global.state.models.length === 0">
        <NResult
          title="No models found"
          description="Click on this icon in the LEFT MENU to access the model download page"
          style="
            height: 70vh;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
          "
        >
          <template #icon>
            <NIcon size="64">
              <CubeSharp />
            </NIcon>
          </template>
        </NResult>
      </div>
      <div v-else>
        <div style="display: inline-flex; width: 100%; margin-bottom: 12px">
          <NInput
            v-model:value="filter"
            clearable
            placeholder="Filter Models"
          />
          <NButton
            ghost
            type="success"
            style="margin-left: 4px"
            @click="refreshModels"
            >Refresh</NButton
          >
        </div>
        <NScrollbar>
          <NTabs type="segment" style="height: 70vh">
            <NTabPane name="PyTorch" style="height: 100%">
              <NGrid cols="1 900:3" :x-gap="8" :y-gap="8" style="height: 100%">
                <!-- Models -->
                <NGi>
                  <NCard title="Models" style="height: 100%">
                    <div
                      style="
                        display: inline-flex;
                        width: 100%;
                        align-items: center;
                        justify-content: space-between;
                        border-bottom: 1px solid rgb(66, 66, 71);
                      "
                      v-for="model in pyTorchModels"
                      v-bind:key="model.path"
                    >
                      <p>{{ model.name }}</p>
                      <div style="display: inline-flex">
                        <NButton
                          type="error"
                          ghost
                          @click="unloadModel(model)"
                          v-if="model.state === 'loaded'"
                          >Unload</NButton
                        >
                        <NButton
                          type="success"
                          ghost
                          @click="loadModel(model)"
                          :loading="model.state === 'loading'"
                          v-else
                          >Load</NButton
                        >
                        <NButton
                          type="info"
                          style="margin-left: 4px"
                          ghost
                          @click="selectedModel = model"
                          :disabled="model.state !== 'loaded'"
                          >Select</NButton
                        >
                      </div>
                    </div>
                  </NCard>
                </NGi>

                <!-- LoRA -->
                <NGi>
                  <NCard :title="lora_title">
                    <NCard
                      style="width: 100%; margin-bottom: 8px"
                      title="LoRA strength"
                      header-style="padding-bottom: 0; font-size: 16px"
                    >
                      <div class="flex-container">
                        <p class="slider-label">Text Encoder</p>
                        <NSlider
                          v-model:value="
                            conf.data.settings.api.lora_text_encoder_weight
                          "
                          :min="0.1"
                          :max="1"
                          :step="0.01"
                          style="margin-right: 12px"
                        />
                      </div>

                      <div class="flex-container">
                        <p class="slider-label">UNet</p>
                        <NSlider
                          v-model:value="
                            conf.data.settings.api.lora_unet_weight
                          "
                          :min="0.1"
                          :max="1"
                          :step="0.01"
                          style="margin-right: 12px"
                        />
                      </div>
                    </NCard>
                    <div
                      style="
                        display: inline-flex;
                        width: 100%;
                        align-items: center;
                        justify-content: space-between;
                        border-bottom: 1px solid rgb(66, 66, 71);
                      "
                      v-for="lora in loraModels"
                      v-bind:key="lora.path"
                    >
                      <p>{{ lora.name }}</p>
                      <div style="display: inline-flex">
                        <NButton
                          type="error"
                          ghost
                          disabled
                          v-if="selectedModel?.loras.includes(lora.path)"
                          >Loaded</NButton
                        >
                        <NButton
                          type="success"
                          ghost
                          @click="loadLoRA(lora)"
                          :disabled="selectedModel === undefined"
                          :loading="lora.state === 'loading'"
                          v-else
                          >Load</NButton
                        >
                      </div>
                    </div>
                  </NCard>
                </NGi>

                <!-- Textual Inversions -->
                <NGi>
                  <NCard :title="textual_inversions_title">
                    <NAlert
                      type="warning"
                      show-icon
                      title="Usage of textual inversion"
                    >
                      <b>Ignore the tokens on CivitAI</b>. The name of the
                      inversion that is displayed here will be the actual token
                    </NAlert>
                    <div
                      style="
                        display: inline-flex;
                        width: 100%;
                        align-items: center;
                        justify-content: space-between;
                        border-bottom: 1px solid rgb(66, 66, 71);
                      "
                      v-for="textualInversion in textualInversionModels"
                      v-bind:key="textualInversion.path"
                    >
                      <p>{{ textualInversion.name }}</p>
                      <div style="display: inline-flex">
                        <NButton
                          type="error"
                          ghost
                          disabled
                          v-if="
                            selectedModel?.loras.includes(textualInversion.path)
                          "
                          >Loaded</NButton
                        >
                        <NButton
                          type="success"
                          ghost
                          @click="loadTextualInversion(textualInversion)"
                          :disabled="selectedModel === undefined"
                          :loading="textualInversion.state === 'loading'"
                          v-else
                          >Load</NButton
                        >
                      </div>
                    </div>
                  </NCard>
                </NGi>
              </NGrid>
            </NTabPane>
            <NTabPane name="AITemplate">
              <NCard title="Models" style="height: 100%">
                <div
                  style="
                    display: inline-flex;
                    width: 100%;
                    align-items: center;
                    justify-content: space-between;
                    border-bottom: 1px solid rgb(66, 66, 71);
                  "
                  v-for="model in aitModels"
                  v-bind:key="model.path"
                >
                  <p>{{ model.name }}</p>
                  <div>
                    <NButton
                      type="error"
                      ghost
                      @click="unloadModel(model)"
                      v-if="model.state === 'loaded'"
                      >Unload</NButton
                    >
                    <NButton
                      type="success"
                      ghost
                      @click="loadModel(model)"
                      :loading="model.state === 'loading'"
                      v-else
                      >Load</NButton
                    >
                  </div>
                </div>
              </NCard>
            </NTabPane>
            <NTabPane name="ONNX">
              <NCard title="Models" style="height: 100%">
                <div
                  style="
                    display: inline-flex;
                    width: 100%;
                    align-items: center;
                    justify-content: space-between;
                    border-bottom: 1px solid rgb(66, 66, 71);
                  "
                  v-for="model in onnxModels"
                  v-bind:key="model.path"
                >
                  <p>{{ model.name }}</p>
                  <div>
                    <NButton
                      type="error"
                      ghost
                      @click="unloadModel(model)"
                      v-if="model.state === 'loaded'"
                      >Unload</NButton
                    >
                    <NButton
                      type="success"
                      ghost
                      @click="loadModel(model)"
                      :loading="model.state === 'loading'"
                      v-else
                      >Load</NButton
                    >
                  </div>
                </div>
              </NCard>
            </NTabPane>
            <NTabPane name="Extra">
              <NCard title="Models" style="height: 100%">
                <div
                  style="
                    display: inline-flex;
                    width: 100%;
                    align-items: center;
                    justify-content: space-between;
                    border-bottom: 1px solid rgb(66, 66, 71);
                  "
                  v-for="model in trtModels"
                  v-bind:key="model.path"
                >
                  <p>{{ model.name }}</p>
                  <div>
                    <NButton
                      type="error"
                      ghost
                      @click="unloadModel(model)"
                      v-if="model.state === 'loaded'"
                      >Unload</NButton
                    >
                    <NButton
                      type="success"
                      ghost
                      @click="loadModel(model)"
                      :loading="model.state === 'loading'"
                      v-else
                      >Load</NButton
                    >
                  </div>
                </div>
              </NCard>
            </NTabPane>
          </NTabs>
        </NScrollbar>
      </div>
    </NModal>

    <!-- Progress bar -->
    <div class="progress-container">
      <NProgress
        type="line"
        :percentage="global.state.progress"
        indicator-placement="outside"
        :processing="global.state.progress < 100 && global.state.progress > 0"
        color="#63e2b7"
        :show-indicator="true"
      >
        <NText>
          {{ global.state.current_step }} / {{ global.state.total_steps }}
        </NText>
      </NProgress>
    </div>
    <div style="display: inline-flex; align-items: center">
      <NDropdown :options="dropdownOptions" @select="dropdownSelected">
        <NButton
          :type="websocketState.color"
          quaternary
          icon-placement="left"
          :render-icon="renderIcon(WifiSharp)"
          :loading="websocketState.loading"
          @click="startWebsocket(message)"
          >{{ websocketState.text }}</NButton
        >
      </NDropdown>
      <NButton
        type="success"
        quaternary
        icon-placement="left"
        :render-icon="perfIcon"
        @click="global.state.perf_drawer.enabled = true"
        :disabled="global.state.perf_drawer.enabled"
      />
      <NButton
        quaternary
        icon-placement="left"
        :render-icon="themeIcon"
        style="margin-right: 8px"
        @click="
          conf.data.settings.frontend.theme =
            conf.data.settings.frontend.theme === 'dark' ? 'light' : 'dark'
        "
      />
    </div>
  </div>
</template>

<script lang="ts" setup>
import type { ModelEntry } from "@/core/interfaces";
import {
  NCard,
  NDropdown,
  NGi,
  NGrid,
  NIcon,
  NInput,
  NModal,
  NScrollbar,
  NSelect,
  NSlider,
  NTabPane,
  NTabs,
  NText,
  type DropdownOption,
} from "naive-ui";

import { serverUrl } from "@/env";
import { startWebsocket } from "@/functions";
import { useWebsocket } from "@/store/websockets";
import {
  ContrastSharp,
  CubeSharp,
  PowerSharp,
  SettingsSharp,
  StatsChart,
  SyncSharp,
  WifiSharp,
} from "@vicons/ionicons5";
import { NAlert, NButton, NProgress, NResult, useMessage } from "naive-ui";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { computed, h, ref, type Component, type ComputedRef } from "vue";
import { useRouter } from "vue-router";
import { useSettings } from "../store/settings";
import { useState } from "../store/state";
const router = useRouter();

const websocketState = useWebsocket();
const global = useState();
const conf = useSettings();

const modelsLoading = ref(false);
const filter = ref("");

const filteredModels = computed(() => {
  return global.state.models.filter((model) => {
    return (
      model.path.toLowerCase().includes(filter.value.toLowerCase()) ||
      filter.value === ""
    );
  });
});

const pyTorchModels = computed(() => {
  return filteredModels.value.filter((model) => {
    return model.backend === "PyTorch" && model.valid === true;
  });
});

const aitModels = computed(() => {
  return filteredModels.value.filter((model) => {
    return model.backend === "AITemplate";
  });
});

const onnxModels = computed(() => {
  return filteredModels.value.filter((model) => {
    return model.backend === "ONNX";
  });
});

const trtModels = computed(() => {
  return filteredModels.value.filter((model) => {
    return model.backend === "TensorRT";
  });
});

const loraModels = computed(() => {
  return filteredModels.value.filter((model) => {
    return model.backend === "LoRA";
  });
});

const textualInversionModels = computed(() => {
  return filteredModels.value.filter((model) => {
    return model.backend === "Textual Inversion";
  });
});

function refreshModels() {
  console.log("Refreshing models");
  modelsLoading.value = true;
  fetch(`${serverUrl}/api/models/available`)
    .then((res) => {
      res.json().then((data: Array<ModelEntry>) => {
        // TODO: Lora loaded state isnt updated
        global.state.models.splice(0, global.state.models.length);
        data.forEach((model) => {
          global.state.models.push(model);
        });
        modelsLoading.value = false;
      });
    })
    .then(() => {
      fetch(`${serverUrl}/api/models/loaded`).then((res) => {
        res.json().then((data: Array<ModelEntry>) => {
          // Check if the current model is still loaded, if not, set it to null
          if (conf.data.settings.model) {
            if (
              !data.find((model) => {
                return model.path === conf.data.settings.model?.path;
              })
            ) {
              console.log("Current model is not loaded anymore");
              conf.data.settings.model = null;
            }
          }

          // Update the state of the models
          data.forEach((loadedModel) => {
            const model = global.state.models.find((model) => {
              return model.path === loadedModel.path;
            });
            if (model) {
              // Update all the keys
              Object.assign(model, loadedModel);
            }
          });

          // Set the current model to the first available model if it was null
          if (!conf.data.settings.model) {
            const allLoaded = [
              ...loadedPyTorchModels.value,
              ...loadedAitModels.value,
              ...loadedOnnxModels.value,
              ...loadedExtraModels.value,
            ];

            console.log("All loaded models: ", allLoaded);

            if (allLoaded.length > 0) {
              conf.data.settings.model = allLoaded[0];
              console.log(
                "Set current model to first available model: ",
                conf.data.settings.model
              );
            } else {
              console.log("No models available");
              conf.data.settings.model = null;
            }
          }
          if (conf.data.settings.model) {
            const spl = conf.data.settings.model.name.split("__")[1];
            if (spl) {
              const xspl = spl.split("x");
              const width = parseInt(xspl[0]);
              const height = parseInt(xspl[1]);
              const batch_size = parseInt(xspl[2]);

              conf.data.settings.aitDim.width = width;
              conf.data.settings.aitDim.height = height;
              conf.data.settings.aitDim.batch_size = batch_size;
            } else {
              conf.data.settings.aitDim.width = undefined;
              conf.data.settings.aitDim.height = undefined;
              conf.data.settings.aitDim.batch_size = undefined;
            }
          } else {
            conf.data.settings.aitDim.width = undefined;
            conf.data.settings.aitDim.height = undefined;
            conf.data.settings.aitDim.batch_size = undefined;
          }
        });
      });
    });
}

async function loadModel(model: ModelEntry) {
  model.state = "loading";
  modelsLoading.value = true;
  const load_url = new URL(`${serverUrl}/api/models/load`);
  const params = { model: model.path, backend: model.backend };
  load_url.search = new URLSearchParams(params).toString();

  try {
    await fetch(load_url, {
      method: "POST",
    });
  } catch (e) {
    console.error(e);
  } finally {
    modelsLoading.value = false;
  }
}

async function unloadModel(model: ModelEntry) {
  const load_url = new URL(`${serverUrl}/api/models/unload`);
  const params = { model: model.name };
  load_url.search = new URLSearchParams(params).toString();

  try {
    await fetch(load_url, {
      method: "POST",
    });
  } catch (e) {
    console.error(e);
  }
}

async function loadLoRA(lora: ModelEntry) {
  if (selectedModel.value) {
    try {
      await fetch(`${serverUrl}/api/models/load-lora`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: selectedModel.value.name,
          lora: lora.path,
          unet_weight: conf.data.settings.api.lora_unet_weight,
          text_encoder_weight: conf.data.settings.api.lora_text_encoder_weight,
        }),
      });
      selectedModel.value.loras.push(lora.path);
    } catch (e) {
      console.error(e);
    }
  } else {
    message.error("No model selected");
  }
}

async function loadTextualInversion(textualInversion: ModelEntry) {
  if (selectedModel.value) {
    try {
      await fetch(`${serverUrl}/api/models/load-textual-inversion`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: selectedModel.value.name,
          textual_inversion: textualInversion.path,
        }),
      });
      selectedModel.value.loras.push(textualInversion.path);
    } catch (e) {
      console.error(e);
    }
  } else {
    message.error("No model selected");
  }
}

async function onModelChange(modelStr: string) {
  const modelName = modelStr.split(":")[0];
  const modelBackend = modelStr.split(":")[1];

  const model = global.state.models.find((model) => {
    return model.path === modelName && model.backend === modelBackend;
  });

  if (model) {
    conf.data.settings.model = model;
  } else {
    message.error("Model not found");
  }

  if (conf.data.settings.model) {
    const spl = conf.data.settings.model.name.split("__")[1];
    if (spl) {
      const xspl = spl.split("x");
      const width = parseInt(xspl[0]);
      const height = parseInt(xspl[1]);
      const batch_size = parseInt(xspl[2]);

      conf.data.settings.aitDim.width = width;
      conf.data.settings.aitDim.height = height;
      conf.data.settings.aitDim.batch_size = batch_size;
    } else {
      conf.data.settings.aitDim.width = undefined;
      conf.data.settings.aitDim.height = undefined;
      conf.data.settings.aitDim.batch_size = undefined;
    }
  } else {
    conf.data.settings.aitDim.width = undefined;
    conf.data.settings.aitDim.height = undefined;
    conf.data.settings.aitDim.batch_size = undefined;
  }
}

function resetModels() {
  global.state.models.splice(0, global.state.models.length);
  console.log("Reset models");
}

const perfIcon = () => {
  return h(StatsChart);
};

const themeIcon = () => {
  return h(ContrastSharp);
};

websocketState.onConnectedCallbacks.push(() => {
  refreshModels();
});
websocketState.onDisconnectedCallbacks.push(() => {
  resetModels();
});
websocketState.onRefreshCallbacks.push(() => {
  refreshModels();
});
if (websocketState.readyState === "OPEN") {
  refreshModels();
}

const loadedPyTorchModels = computed(() => {
  return global.state.models.filter((model) => {
    return model.backend === "PyTorch" && model.state === "loaded";
  });
});
const loadedAitModels = computed(() => {
  return global.state.models.filter((model) => {
    return model.backend === "AITemplate" && model.state === "loaded";
  });
});
const loadedOnnxModels = computed(() => {
  return global.state.models.filter((model) => {
    return model.backend === "ONNX" && model.state === "loaded";
  });
});
const loadedExtraModels = computed(() => {
  return global.state.models.filter((model) => {
    return model.backend === "unknown" && model.state === "loaded";
  });
});

const pyTorchOptions: ComputedRef<SelectMixedOption> = computed(() => {
  return {
    type: "group",
    label: "PyTorch",
    key: "pytorch",
    children: loadedPyTorchModels.value.map((model) => {
      return {
        label: model.name,
        value: `${model.path}:PyTorch`,
      };
    }),
  };
});

const aitOptions: ComputedRef<SelectMixedOption> = computed(() => {
  return {
    type: "group",
    label: "AITemplate",
    key: "ait",
    children: loadedAitModels.value.map((model) => {
      return {
        label: model.name,
        value: `${model.path}:AITemplate`,
      };
    }),
  };
});

const onnxOptions: ComputedRef<SelectMixedOption> = computed(() => {
  return {
    type: "group",
    label: "ONNX",
    key: "onnx",
    children: loadedOnnxModels.value.map((model) => {
      return {
        label: model.name,
        value: `${model.path}:ONNX`,
      };
    }),
  };
});

const extraOptions: ComputedRef<SelectMixedOption> = computed(() => {
  return {
    type: "group",
    label: "Extra",
    key: "extra",
    children: loadedExtraModels.value.map((model) => {
      return {
        label: model.name,
        value: `${model.path}:PyTorch`,
      };
    }),
  };
});

const generatedModelOptions: ComputedRef<SelectMixedOption[]> = computed(() => {
  return [
    pyTorchOptions.value,
    aitOptions.value,
    onnxOptions.value,
    extraOptions.value,
  ];
});

const message = useMessage();

const showModal = ref(false);
const selectedModel = ref<ModelEntry>();
const lora_title = computed(() => {
  return `LoRA (${
    selectedModel.value ? selectedModel.value.name : "No model selected"
  })`;
});

const textual_inversions_title = computed(() => {
  return `Textual Inversions (${
    selectedModel.value ? selectedModel.value.name : "No model selected"
  })`;
});

const renderIcon = (icon: Component) => {
  return () => {
    return h(NIcon, null, {
      default: () => h(icon),
    });
  };
};

const dropdownOptions: DropdownOption[] = [
  {
    label: "Reconnect",
    key: "reconnect",
    icon: renderIcon(SyncSharp),
  },
  {
    label: "Settings",
    key: "settings",
    icon: renderIcon(SettingsSharp),
  },
  {
    label: "Shutdown",
    key: "shutdown",
    icon: renderIcon(PowerSharp),
  },
];

async function dropdownSelected(key: string) {
  if (key === "reconnect") {
    await startWebsocket(message);
  } else if (key === "settings") {
    router.push("/settings");
  } else if (key === "shutdown") {
    await fetch(`${serverUrl}/api/general/shutdown`, {
      method: "POST",
    });
  }
}

startWebsocket(message);

const backgroundColor = computed(() => {
  if (conf.data.settings.frontend.theme === "dark") {
    return "#121215";
  } else {
    return "#fff";
  }
});
</script>

<style scoped>
.progress-container {
  margin: 12px;
  flex-grow: 1;
  width: 400px;
}
.top-bar {
  display: inline-flex;
  align-items: center;
  border-bottom: #505050 1px solid;
  padding-top: 10px;
  padding-bottom: 10px;
  width: calc(100% - 64px);
  height: 32px;
  position: fixed;
  top: 0;
  z-index: 1;
  background-color: v-bind(backgroundColor);
}

.logo {
  margin-right: 16px;
  margin-left: 16px;
}
</style>
