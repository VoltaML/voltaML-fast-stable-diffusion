<template>
  <div style="margin: 18px">
    <NCard title="Custom model" segmented>
      <div
        style="
          width: 100%;
          display: inline-flex;
          justify-content: space-between;
          align-items: center;
        "
      >
        <div>Install custom models from Hugging Face</div>
        <div style="display: inline-flex; align-items: center">
          <NInput
            v-model:value="customModel"
            placeholder="andite/anything-v4.0"
            style="width: 350px"
          />
          <NButton
            type="primary"
            bordered
            @click="downloadModel(customModel)"
            :loading="conf.state.downloading"
            :disabled="conf.state.downloading || customModel === ''"
            secondary
            style="margin-right: 16px; margin-left: 4px"
            >Install</NButton
          >
        </div>
      </div>
    </NCard>

    <NCard title="Currated models" style="margin-top: 12px" segmented>
      <NInput
        v-model:value="modelFilter"
        style="margin-bottom: 12px"
        placeholder="Filter"
        clearable
      />
      <NDataTable
        :columns="columns"
        :data="dataRef"
        :pagination="pagination"
        :bordered="true"
        style="padding-bottom: 24px"
      />
    </NCard>
  </div>
</template>

<script lang="ts" setup>
import type { Model } from "@/core/models";
import { serverUrl } from "@/env";
import { Home, Menu } from "@vicons/ionicons5";
import {
  NButton,
  NCard,
  NDataTable,
  NDropdown,
  NIcon,
  NInput,
  useMessage,
  type DataTableColumns,
} from "naive-ui";
import type { DropdownMixedOption } from "naive-ui/es/dropdown/src/interface";
import { computed, h, reactive, ref, type Component, type Ref } from "vue";
import { huggingfaceModelsFile } from "../../env";
import { useState } from "../../store/state";
const conf = useState();
const message = useMessage();

const customModel = ref("");

function downloadModel(model: Ref<string> | string) {
  const url = new URL(`${serverUrl}/api/models/download`);
  const modelName = typeof model === "string" ? model : model.value;
  url.searchParams.append("model", modelName);
  console.log(url);
  conf.state.downloading = true;
  customModel.value = "";
  message.info(`Downloading model: ${modelName}`);
  fetch(url, { method: "POST" })
    .then(() => {
      conf.state.downloading = false;
      message.success(`Downloaded model: ${modelName}`);
    })
    .catch(() => {
      conf.state.downloading = false;
      message.error(`Failed to download model: ${modelName}`);
    });
}

const renderIcon = (
  icon: Component,
  size: "small" | "medium" | "large" = "medium"
) => {
  return () => {
    return h(
      NIcon,
      {
        size: size,
      },
      {
        default: () => h(icon),
      }
    );
  };
};

function getPluginOptions(row: Model) {
  const options: DropdownMixedOption[] = [
    {
      label: "Hugging Face",
      key: "github",
      icon: renderIcon(Home),
      props: {
        onClick: () => window.open(row.huggingface_url, "_blank"),
      },
    },
  ];
  return options;
}

const columns: DataTableColumns<Model> = [
  {
    title: "Name",
    key: "name",
    sorter: "default",
  },
  {
    title: "Repository",
    key: "huggingface_id",
    sorter: "default",
  },
  {
    title: "Download",
    key: "download",
    render(row) {
      return h(
        NButton,
        {
          type: "primary",
          secondary: true,
          round: true,
          block: true,
          bordered: false,
          disabled: conf.state.downloading,
          onClick: () => {
            downloadModel(row.huggingface_id);
          },
        },
        { default: () => "Download" }
      );
    },
  },
  {
    title: "",
    width: 60,
    key: "menu",
    render(row) {
      return h(
        NDropdown,
        {
          trigger: "hover",
          options: getPluginOptions(row),
          disabled: conf.state.downloading,
        },
        { default: renderIcon(Menu) }
      );
    },
    filter: "default",
  },
];

const modelData = reactive<Model[]>([]);
const modelFilter = ref("");

const dataRef = computed(() => {
  if (modelFilter.value !== "") {
    return modelData.filter((model) =>
      model.name.toLowerCase().includes(modelFilter.value.toLowerCase())
    );
  } else {
    return modelData;
  }
});
const pagination = reactive({ pageSize: 10 });

fetch(huggingfaceModelsFile).then((res) => {
  res.json().then((data: { models: Model[] }) => {
    modelData.push(...data["models"]);
  });
});
</script>

<style scoped>
.install {
  width: 100%;
  padding: 10px 0px;
}
</style>
