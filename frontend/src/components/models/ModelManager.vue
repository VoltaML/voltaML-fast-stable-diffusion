<template>
  <div style="padding: 0px 12px 12px 12px">
    <NInput
      v-model:value="filter"
      style="width: 100%; margin-bottom: 12px"
      placeholder="Filter"
      clearable
    />
    <NGrid cols="1 600:2 900:3" x-gap="8" y-gap="8">
      <NGi
        v-for="key in (
          Object.keys(modelTypes) as Array<keyof typeof modelTypes>
        ).filter((item) => item !== 'AITemplate' && item !== 'ONNX')"
      >
        <NCard :title="key">
          <NUpload
            multiple
            directory-dnd
            :action="`${serverUrl}/api/models/upload-model?type=${modelTypes[key]}`"
            :accept="allowedExtensions"
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
              <NText style="font-size: 24px"> {{ key }} </NText>
              <NText style="font-size: 14px">
                Click or Drag a model here
              </NText>
            </NUploadDragger>
          </NUpload>
        </NCard>
      </NGi>
    </NGrid>

    <NDivider />

    <NGrid style="margin-top: 12px" cols="1 900:2 1100:3" x-gap="12" y-gap="12">
      <NGi
        v-for="modelType in Object.keys(Backends).filter((item) =>
          isNaN(Number(item))
        ) as (keyof typeof modelTypes)[]"
      >
        <NCard :title="modelType" style="width: 100%">
          <div
            style="
              display: inline-flex;
              width: 100%;
              align-items: center;
              justify-content: space-between;
              border-bottom: 1px solid rgb(66, 66, 71);
            "
            v-for="model in filteredModels.filter(
              (item) => item.backend === modelType
            )"
            v-bind:key="model.path"
          >
            <p>{{ model.name }}</p>
            <div style="display: inline-flex">
              <NDropdown
                :options="createOptions(model.path)"
                placement="right"
                @select="
                  (key) => handleAction(key, modelTypes[modelType], model)
                "
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
  NDivider,
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
import { Backends, type ModelEntry } from "../../core/interfaces";
import { useState } from "../../store/state";

const global = useState();
const filter = ref("");
const message = useMessage();

const modelTypes = {
  PyTorch: "models",
  LoRA: "lora",
  LyCORIS: "lycoris",
  "Textual Inversion": "textual-inversion",
  VAE: "vae",
  AITemplate: "aitemplate",
  ONNX: "onnx",
} as const;

const allowedExtensions = ".safetensors,.ckpt,.pth,.pt,.bin";

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

function createOptions(model_path: string) {
  return [
    {
      label: "Delete",
      key: `delete:${model_path}`,
      icon: renderIcon(TrashBin),
    },
  ];
}

async function deleteModel(
  model_path: string,
  model_type: (typeof modelTypes)[keyof typeof modelTypes]
) {
  try {
    const res = await fetch(`${serverUrl}/api/models/delete-model`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model_path: model_path,
        model_type: model_type,
      }),
    });

    await res.json();
    message.success("Model deleted");
  } catch (error) {
    message.error(error as string);
  }
}

async function handleAction(
  key: string,
  modelType: (typeof modelTypes)[keyof typeof modelTypes],
  model: ModelEntry
) {
  const [action, model_path] = key.split(":");
  if (action === "delete") {
    await deleteModel(model_path, modelType);
  }
}
</script>
