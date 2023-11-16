<template>
  <div class="top-bar">
    <NSelect
      style="max-width: 250px; padding-left: 12px; padding-right: 12px"
      :options="generatedModelOptions"
      @update:value="onModelChange"
      :loading="modelsLoading"
      placeholder=""
      :value="
        settings.data.settings.model !== null
          ? settings.data.settings.model?.name
          : ''
      "
      :consistent-menu-width="false"
      filterable
    />
    <NButton
      @click="showModal = true"
      :loading="modelsLoading"
      :type="settings.data.settings.model ? 'default' : 'success'"
    >
      Load Model</NButton
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
          style="
            height: 70vh;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
          "
          status="404"
        >
          <template #footer>
            <NTooltip>
              <template #trigger>
                <NButton
                  type="success"
                  @click="
                    () => {
                      global.state.modelManager.tab = 'civitai';
                      router.push('/models');
                      showModal = false;
                    }
                  "
                  >Get some models</NButton
                >
              </template>

              <img
                src="https://i.imgflip.com/84840n.jpg"
                style="max-width: 30vw; max-height: 30vh"
              />
            </NTooltip>
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
                      <div
                        style="
                          display: flex;
                          flex-direction: row;
                          align-items: center;
                        "
                      >
                        <NTag
                          :type="getModelTag(model.type)[1]"
                          ghost
                          style="margin-right: 8px"
                        >
                          {{ getModelTag(model.type)[0] }}
                        </NTag>
                        <p>{{ model.name }}</p>
                      </div>
                      <div style="display: inline-flex">
                        <NButton
                          type="error"
                          ghost
                          @click="unloadModel(model)"
                          v-if="model.state === 'loaded'"
                          >Unload
                        </NButton>
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
                          @click="global.state.selected_model = model"
                          :disabled="model.state !== 'loaded'"
                          >Select</NButton
                        >
                      </div>
                    </div>
                  </NCard>
                </NGi>

                <!-- VAE -->
                <NGi>
                  <NCard :title="vae_title">
                    <div v-if="global.state.selected_model !== null">
                      <div
                        style="
                          display: inline-flex;
                          width: 100%;
                          align-items: center;
                          justify-content: space-between;
                          border-bottom: 1px solid rgb(66, 66, 71);
                        "
                        v-for="vae in vaeModels"
                        v-bind:key="vae.path"
                      >
                        <p>{{ vae.name }}</p>
                        <div style="display: inline-flex">
                          <NButton
                            type="error"
                            ghost
                            disabled
                            v-if="global.state.selected_model?.vae == vae.path"
                            >Loaded
                          </NButton>
                          <NButton
                            type="success"
                            ghost
                            @click="loadVAE(vae)"
                            :disabled="
                              global.state.selected_model === undefined
                            "
                            :loading="vae.state === 'loading'"
                            v-else
                            >Load</NButton
                          >
                        </div>
                      </div>
                    </div>
                    <div v-else>
                      <NAlert
                        type="warning"
                        show-icon
                        title="No model selected"
                        style="margin-top: 4px"
                      >
                        Please select a model first
                      </NAlert>
                    </div>
                  </NCard>
                </NGi>

                <!-- Textual Inversions -->
                <NGi>
                  <NCard :title="textual_inversions_title">
                    <NAlert
                      type="info"
                      show-icon
                      title="Usage of textual inversion"
                    >
                      <b>Ignore the tokens on CivitAI</b>. The name of the
                      inversion that is displayed here will be the actual token
                      (easynegative.pt -> easynegative)
                    </NAlert>
                    <div v-if="global.state.selected_model !== null">
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
                              global.state.selected_model?.textual_inversions.includes(
                                textualInversion.path
                              )
                            "
                            >Loaded</NButton
                          >
                          <NButton
                            type="success"
                            ghost
                            @click="loadTextualInversion(textualInversion)"
                            :disabled="
                              global.state.selected_model === undefined
                            "
                            :loading="textualInversion.state === 'loading'"
                            v-else
                            >Load</NButton
                          >
                        </div>
                      </div>
                    </div>
                    <div v-else>
                      <NAlert
                        type="warning"
                        show-icon
                        title="No model selected"
                        style="margin-top: 4px"
                      >
                        Please select a model first
                      </NAlert>
                    </div>
                  </NCard>
                </NGi>
              </NGrid>
            </NTabPane>
            <NTabPane name="AITemplate">
              <NScrollbar style="height: 70vh">
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
                        >Unload
                      </NButton>
                      <NButton
                        type="success"
                        ghost
                        @click="loadModel(model)"
                        :loading="model.state === 'loading'"
                        v-else
                      >
                        Load</NButton
                      >
                    </div>
                  </div>
                </NCard>
              </NScrollbar>
            </NTabPane>

            <NTabPane name="ONNX">
              <NScrollbar style="height: 70vh">
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
                        >Unload
                      </NButton>
                      <NButton
                        type="success"
                        ghost
                        @click="loadModel(model)"
                        :loading="model.state === 'loading'"
                        v-else
                      >
                        Load</NButton
                      >
                    </div>
                  </div>
                </NCard>
              </NScrollbar>
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
          :render-icon="renderIcon(Wifi)"
          :loading="websocketState.loading"
          @click="startWebsocket(message)"
        ></NButton>
      </NDropdown>
    </div>
  </div>
</template>

<script lang="ts" setup>
import type { ModelEntry } from "@/core/interfaces";
import {
  NAlert,
  NButton,
  NCard,
  NDropdown,
  NGi,
  NGrid,
  NIcon,
  NInput,
  NModal,
  NProgress,
  NResult,
  NScrollbar,
  NSelect,
  NTabPane,
  NTabs,
  NTag,
  NText,
  NTooltip,
  useMessage,
  type DropdownOption,
} from "naive-ui";

import { serverUrl } from "@/env";
import { startWebsocket } from "@/functions";
import { useWebsocket } from "@/store/websockets";
import {
  DocumentText,
  PowerSharp,
  SettingsSharp,
  StatsChart,
  SyncSharp,
  Wifi,
} from "@vicons/ionicons5";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { computed, h, ref, type Component, type ComputedRef } from "vue";
import { useRouter } from "vue-router";
import { useSettings } from "../store/settings";
import { useState } from "../store/state";
const router = useRouter();

const websocketState = useWebsocket();
const global = useState();
const settings = useSettings();

const modelsLoading = ref(false);
const filter = ref("");

const filteredModels = computed<ModelEntry[]>(() => {
  return global.state.models.filter((model) => {
    return (
      model.path.toLowerCase().includes(filter.value.toLowerCase()) ||
      filter.value === ""
    );
  });
});

const pyTorchModels = computed<ModelEntry[]>(() => {
  return filteredModels.value
    .filter((model) => {
      return model.backend === "PyTorch" && model.valid === true;
    })
    .sort((a, b) => {
      if (a.state === "loaded" && b.state !== "loaded") {
        return -1; // a should come before b
      } else if (a.state !== "loaded" && b.state === "loaded") {
        return 1; // b should come before a
      } else {
        // If 'state' is the same, sort alphabetically by 'name'
        return a.name.localeCompare(b.name);
      }
    });
});

const aitModels = computed<ModelEntry[]>(() => {
  return filteredModels.value
    .filter((model) => {
      return model.backend === "AITemplate";
    })
    .sort((a, b) => {
      if (a.state === "loaded" && b.state !== "loaded") {
        return -1; // a should come before b
      } else if (a.state !== "loaded" && b.state === "loaded") {
        return 1; // b should come before a
      } else {
        // If 'state' is the same, sort alphabetically by 'name'
        return a.name.localeCompare(b.name);
      }
    });
});

const onnxModels = computed<ModelEntry[]>(() => {
  return filteredModels.value
    .filter((model) => {
      return model.backend === "ONNX";
    })
    .sort((a, b) => {
      if (a.state === "loaded" && b.state !== "loaded") {
        return -1; // a should come before b
      } else if (a.state !== "loaded" && b.state === "loaded") {
        return 1; // b should come before a
      } else {
        // If 'state' is the same, sort alphabetically by 'name'
        return a.name.localeCompare(b.name);
      }
    });
});

const manualVAEModels = computed<ModelEntry[]>(() => {
  const selectedModel = global.state.selected_model;
  if (selectedModel?.type === "SDXL") {
    return [
      {
        name: "Default VAE (fp32)",
        path: "default",
        backend: "VAE",
        valid: true,
        state: "not loaded",
        vae: "default",
        textual_inversions: [],
        type: "SDXL",
        stage: "last_stage",
      } as ModelEntry,
      {
        name: "FP16 VAE",
        path: "madebyollin/sdxl-vae-fp16-fix",
        backend: "VAE",
        valid: true,
        state: "not loaded",
        vae: "fp16",
        textual_inversions: [],
        type: "SDXL",
        stage: "last_stage",
      } as ModelEntry,
    ];
  } else {
    return [
      {
        name: "Default VAE",
        path: "default",
        backend: "VAE",
        valid: true,
        state: "not loaded",
        vae: "default",
        textual_inversions: [],
        type: "SD1.x",
        stage: "last_stage",
      } as ModelEntry,
      {
        name: "Tiny VAE (fast)",
        path: "madebyollin/taesd",
        backend: "VAE",
        valid: true,
        state: "not loaded",
        vae: "madebyollin/taesd",
        textual_inversions: [],
        type: "SD1.x",
        stage: "last_stage",
      } as ModelEntry,
      {
        name: "Asymmetric VAE",
        path: "cross-attention/asymmetric-autoencoder-kl-x-1-5",
        backend: "VAE",
        valid: true,
        state: "not loaded",
        vae: "cross-attention/asymmetric-autoencoder-kl-x-1-5",
        textual_inversions: [],
        type: "SD1.x",
        stage: "last_stage",
      } as ModelEntry,
    ];
  }
});

const vaeModels = computed<ModelEntry[]>(() => {
  return [
    ...manualVAEModels.value,
    ...filteredModels.value
      .filter((model) => {
        return model.backend === "VAE";
      })
      .sort((a, b) => {
        return a.name.localeCompare(b.name);
      }),
  ];
});

const textualInversionModels = computed(() => {
  return filteredModels.value
    .filter((model) => {
      return model.backend === "Textual Inversion";
    })
    .sort((a, b) => {
      if (a.state === "loaded" && b.state !== "loaded") {
        return -1; // a should come before b
      } else if (a.state !== "loaded" && b.state === "loaded") {
        return 1; // b should come before a
      } else {
        // If 'state' is the same, sort alphabetically by 'name'
        return a.name.localeCompare(b.name);
      }
    });
});

function refreshModels() {
  console.log("Refreshing models");
  modelsLoading.value = true;
  fetch(`${serverUrl}/api/models/available`)
    .then((res) => {
      if (!res.ok) {
        throw new Error(res.statusText);
      }

      res.json().then((data: Array<ModelEntry>) => {
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
          if (settings.data.settings.model) {
            if (
              !data.find((model) => {
                return model.path === settings.data.settings.model?.path;
              })
            ) {
              console.log("Current model is not loaded anymore");
              settings.data.settings.model = null;
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
          if (!settings.data.settings.model) {
            const allLoaded = [
              ...loadedPyTorchModels.value,
              ...loadedAitModels.value,
              ...loadedOnnxModels.value,
              ...loadedExtraModels.value,
            ];

            console.log("All loaded models: ", allLoaded);

            if (allLoaded.length > 0) {
              settings.data.settings.model = allLoaded[0];
              console.log(
                "Setting current model to first available model: ",
                settings.data.settings.model
              );
            } else {
              console.log("No models available, setting current model to null");
              settings.data.settings.model = null;
            }
          }
          try {
            if (settings.data.settings.model) {
              const spl = settings.data.settings.model.name.split("__")[1];

              const regex = /([\d]+-[\d]+)x([\d]+-[\d]+)x([\d]+-[\d]+)/g;
              const matches = regex.exec(spl);

              if (matches) {
                const width = matches[1].split("-").map((x) => parseInt(x));
                const height = matches[2].split("-").map((x) => parseInt(x));
                const batch_size = matches[3]
                  .split("-")
                  .map((x) => parseInt(x));

                settings.data.settings.aitDim.width = width;
                settings.data.settings.aitDim.height = height;
                settings.data.settings.aitDim.batch_size = batch_size;
              } else {
                throw new Error("Invalid model name for AIT dimensions parser");
              }
            } else {
              throw new Error("No model, cannot parse AIT dimensions");
            }
          } catch (e) {
            settings.data.settings.aitDim.width = undefined;
            settings.data.settings.aitDim.height = undefined;
            settings.data.settings.aitDim.batch_size = undefined;
          }

          const autofillKeys = [];
          for (const model of global.state.models) {
            if (model.backend === "LoRA" || model.backend === "LyCORIS") {
              autofillKeys.push(`<lora:${model.name}:1.0>`);
            }
            /*else if (model.backend === "Textual Inversion") {
              autofillKeys.push(`<ti:${model.name}:1.0>`);
            }*/
          }

          global.state.autofill_special = autofillKeys;
        });
      });
    })
    .catch((e) => {
      message.error(`Failed to refresh models: ${e}`);
      modelsLoading.value = false;
    });
}

async function loadModel(model: ModelEntry) {
  model.state = "loading";
  modelsLoading.value = true;
  const load_url = new URL(`${serverUrl}/api/models/load`);
  const params = {
    model: model.path,
    backend: model.backend,
    type: model.type,
  };
  load_url.search = new URLSearchParams(params).toString();

  fetch(load_url, {
    method: "POST",
  })
    .then((res) => {
      if (!res.ok) {
        throw new Error(res.statusText);
      }

      model.state = "loaded";
      modelsLoading.value = false;
    })
    .catch((e) => {
      message.error(`Failed to load model: ${e}`);
      console.error(e);
      modelsLoading.value = false;
      model.state = "not loaded";
    });
}

async function unloadModel(model: ModelEntry) {
  if (model === global.state.selected_model) {
    global.state.selected_model = null;
  }

  const load_url = new URL(`${serverUrl}/api/models/unload`);
  const params = { model: model.path };
  load_url.search = new URLSearchParams(params).toString();

  try {
    await fetch(load_url, {
      method: "POST",
    });
  } catch (e) {
    console.error(e);
  }
}

async function loadVAE(vae: ModelEntry) {
  if (global.state.selected_model) {
    try {
      await fetch(`${serverUrl}/api/models/load-vae`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: global.state.selected_model.name,
          vae: vae.path,
        }),
      });
      global.state.selected_model.vae = vae.path;
    } catch (e) {
      console.error(e);
    }
  } else {
    message.error("No model selected");
  }
}

async function loadTextualInversion(textualInversion: ModelEntry) {
  if (global.state.selected_model) {
    try {
      await fetch(`${serverUrl}/api/models/load-textual-inversion`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: global.state.selected_model.name,
          textual_inversion: textualInversion.path,
        }),
      });
      global.state.selected_model.textual_inversions.push(
        textualInversion.path
      );
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
    settings.data.settings.model = model;
  } else {
    message.error("Model not found");
  }

  try {
    if (settings.data.settings.model) {
      const spl = settings.data.settings.model.name.split("__")[1];

      const regex = /([\d]+-[\d]+)x([\d]+-[\d]+)x([\d]+-[\d]+)/g;
      const match = spl.match(regex);

      if (match) {
        const width = match[0].split("-").map((x) => parseInt(x));
        const height = match[1].split("-").map((x) => parseInt(x));
        const batch_size = match[2].split("-").map((x) => parseInt(x));

        settings.data.settings.aitDim.width = width;
        settings.data.settings.aitDim.height = height;
        settings.data.settings.aitDim.batch_size = batch_size;
      } else {
        throw new Error("Invalid model name for AIT dimensions parser");
      }
    } else {
      throw new Error("No model, cannot parse AIT dimensions");
    }
  } catch (e) {
    console.warn(e);
    settings.data.settings.aitDim.width = undefined;
    settings.data.settings.aitDim.height = undefined;
    settings.data.settings.aitDim.batch_size = undefined;
  }
}

function resetModels() {
  global.state.models.splice(0, global.state.models.length);
  console.log("Reset models");
}

function getModelTag(
  type: string
): [string, "info" | "warning" | "success" | "error" | "primary"] {
  switch (type) {
    case "SD1.x":
      return [type, "primary"];
    case "SD2.x":
      return [type, "info"];
    case "SDXL":
      return [type, "warning"];
    case "Kandinsky 2.1":
    case "Kandinsky 2.2":
      return ["Kandinsky", "success"];
    default:
      return [type, "error"];
  }
}

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
const vae_title = computed(() => {
  return `VAE (${
    global.state.selected_model
      ? global.state.selected_model.name
      : "No model selected"
  })`;
});

const textual_inversions_title = computed(() => {
  return `Textual Inversions (${
    global.state.selected_model
      ? global.state.selected_model.name
      : "No model selected"
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
    label: "Log",
    key: "log",
    icon: renderIcon(DocumentText),
  },
  {
    label: "Performance",
    key: "performance",
    icon: renderIcon(StatsChart),
  },
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
  switch (key) {
    case "reconnect":
      await startWebsocket(message);
      break;
    case "settings":
      router.push("/settings");
      break;
    case "shutdown":
      await fetch(`${serverUrl}/api/general/shutdown`, {
        method: "POST",
      });
      break;
    case "performance":
      global.state.perf_drawer.enabled = true;
      break;
    case "log":
      global.state.log_drawer.enabled = true;
      break;
  }
}

startWebsocket(message);
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
  padding-top: 10px;
  padding-bottom: 10px;
  width: calc(100% - 64px);
  height: 32px;
  position: fixed;
  top: 0;
  z-index: 1;
}

.logo {
  margin-right: 16px;
  margin-left: 16px;
}
</style>
