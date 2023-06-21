<template>
  <div style="padding: 12px">
    <NInput
      v-model:value="filter"
      style="width: 100%; margin-bottom: 12px"
      placeholder="Filter"
      clearable
    />
    <NGrid cols="3" x-gap="12">
      <NGi>
        <NCard title="Model">
          <NUpload
            multiple
            directory-dnd
            :action="`${serverUrl}/api/models/upload-model`"
            :max="5"
            accept=".ckpt,.safetensors"
            style="
              border-bottom: 1px solid rgb(66, 66, 71);
              padding-bottom: 12px;
            "
          >
            <NUploadDragger
              style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
              "
            >
              <div style="margin-bottom: 12px; display: block">
                <NIcon size="48" :depth="3">
                  <CloudUpload />
                </NIcon>
              </div>
              <NText style="font-size: 24px"> Model </NText>
              <NText style="font-size: 16px">
                Click or drag a model to this area to upload it to the server
              </NText>
            </NUploadDragger>
          </NUpload>
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
              <NDropdown
                :options="createPyTorchOptions(model.path)"
                placement="right"
                @select="handlePyTorchModelAction"
              >
                <NButton :render-icon="renderIcon(Settings)"> </NButton>
              </NDropdown>
            </div>
          </div>
        </NCard>
      </NGi>
      <NGi>
        <NCard title="LoRA">
          <NUpload
            multiple
            directory-dnd
            :action="`${serverUrl}/api/models/upload-model?type=lora`"
            :max="5"
            accept=".ckpt,.safetensors"
            style="
              border-bottom: 1px solid rgb(66, 66, 71);
              padding-bottom: 12px;
            "
          >
            <NUploadDragger
              style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
              "
            >
              <div style="margin-bottom: 12px; display: block">
                <NIcon size="48" :depth="3">
                  <CloudUpload />
                </NIcon>
              </div>
              <NText style="font-size: 24px"> LoRA </NText>

              <NText style="font-size: 16px">
                Click or drag a model to this area to upload it to the server
              </NText>
            </NUploadDragger>
          </NUpload>
          <div
            style="
              display: inline-flex;
              width: 100%;
              align-items: center;
              justify-content: space-between;
              border-bottom: 1px solid rgb(66, 66, 71);
            "
            v-for="model in loraModels"
            v-bind:key="model.path"
          >
            <p>{{ model.name }}</p>
            <div style="display: inline-flex">
              <NDropdown
                :options="createLoraOptions(model.path)"
                placement="right"
                @select="handleLoraModelAction"
              >
                <NButton :render-icon="renderIcon(Settings)"> </NButton>
              </NDropdown>
            </div>
          </div>
        </NCard>
      </NGi>
      <NGi>
        <NCard title="Textual Inversion">
          <NUpload
            multiple
            directory-dnd
            :action="`${serverUrl}/api/models/upload-model?type=textual-inversion`"
            :max="5"
            accept=".pt,.safetensors"
            style="
              border-bottom: 1px solid rgb(66, 66, 71);
              padding-bottom: 12px;
            "
          >
            <NUploadDragger
              style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
              "
            >
              <div style="margin-bottom: 12px; display: block">
                <NIcon size="48" :depth="3">
                  <CloudUpload />
                </NIcon>
              </div>
              <NText style="font-size: 24px"> Textual Inversion </NText>

              <NText style="font-size: 16px">
                Click or drag a model to this area to upload it to the server
              </NText>
            </NUploadDragger>
          </NUpload>
          <div
            style="
              display: inline-flex;
              width: 100%;
              align-items: center;
              justify-content: space-between;
              border-bottom: 1px solid rgb(66, 66, 71);
            "
            v-for="model in textualInversionModels"
            v-bind:key="model.path"
          >
            <p>{{ model.name }}</p>
            <div style="display: inline-flex">
              <NDropdown
                :options="createTextualInversionOptions(model.path)"
                placement="right"
                @select="handleTextualInversionModelAction"
              >
                <NButton :render-icon="renderIcon(Settings)"> </NButton>
              </NDropdown>
            </div>
          </div>
        </NCard>
      </NGi>
    </NGrid>
  </div>
</template>

<script lang="ts" setup>
import { serverUrl } from "@/env";
import { CloudUpload, Settings, TrashBin } from "@vicons/ionicons5";
import {
  NButton,
  NCard,
  NDropdown,
  NGi,
  NGrid,
  NIcon,
  NInput,
  NText,
  NUpload,
  NUploadDragger,
  useMessage,
} from "naive-ui";
import { computed, h, ref, type Component } from "vue";
import { useState } from "../../store/state";

const global = useState();
const filter = ref("");
const message = useMessage();

const renderIcon = (icon: Component) => {
  return () => {
    return h(NIcon, null, {
      default: () => h(icon),
    });
  };
};

const filteredModels = computed(() => {
  return global.state.models
    .filter((model) => {
      return (
        model.path.toLowerCase().includes(filter.value.toLowerCase()) ||
        filter.value === ""
      );
    })
    .sort((a, b) => (a.name.toLowerCase() < b.name.toLowerCase() ? -1 : 1));
});

const pyTorchModels = computed(() => {
  return filteredModels.value.filter((model) => {
    return model.backend === "PyTorch" && model.valid === true;
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

function createPyTorchOptions(model_path: string) {
  return [
    {
      label: "Delete",
      key: `delete:${model_path}`,
      icon: renderIcon(TrashBin),
    },
    // {
    //   label: "Convert",
    //   key: `convert:${model_path}`,
    //   icon: renderIcon(GitCompare),
    // },
    // {
    //   label: "Accelerate",
    //   key: `accelerate:${model_path}`,
    //   icon: renderIcon(PlayForward),
    // },
  ];
}

function createLoraOptions(model_path: string) {
  return [
    {
      label: "Delete",
      key: `delete:${model_path}`,
      icon: renderIcon(TrashBin),
    },
  ];
}

function createTextualInversionOptions(model_path: string) {
  return [
    {
      label: "Delete",
      key: `delete:${model_path}`,
      icon: renderIcon(TrashBin),
    },
  ];
}

function deleteModel(
  model_path: string,
  model_type: "pytorch" | "lora" | "textual-inversion"
) {
  fetch(`${serverUrl}/api/models/delete-model`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model_path: model_path,
      model_type: model_type,
    }),
  })
    .then((response) => response.json())
    .then(() => {
      message.success("Model deleted");
    })
    .catch((error) => {
      message.error(error);
    });
}

function handlePyTorchModelAction(key: string) {
  const [action, model_path] = key.split(":");

  if (action === "delete") {
    deleteModel(model_path, "pytorch");
  } else if (action === "convert") {
    message.success(key);
  } else if (action === "accelerate") {
    message.success(key);
  }
}

function handleLoraModelAction(key: string) {
  const [action, model_path] = key.split(":");
  if (action === "delete") {
    deleteModel(model_path, "lora");
  }
}

function handleTextualInversionModelAction(key: string) {
  const [action, model_path] = key.split(":");
  if (action === "delete") {
    deleteModel(model_path, "textual-inversion");
  }
}
</script>
