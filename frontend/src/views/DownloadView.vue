<template>
  <NSpace
    justify="end"
    inline
    align="center"
    class="install"
    style="width: 100%; margin: 8px"
  >
    <NInput
      v-model:value="customModel"
      placeholder="Custom model"
      style="width: 350px"
    />
    <NButton
      type="primary"
      bordered
      @click="downloadModel"
      :loading="conf.state.downloading"
      :disabled="conf.state.downloading || customModel === ''"
      secondary
      style="margin-right: 16px"
      >Install</NButton
    >
  </NSpace>
  <div
    style="
      height: 50vh;
      display: inline-flex;
      justify-content: center;
      width: 100%;
    "
  >
    <NCard>
      <NDataTable
        :columns="columnsRef"
        :data="dataRef"
        :pagination="pagination"
        :bordered="true"
        :remote="true"
        style="padding-bottom: 24px"
    /></NCard>
  </div>
</template>

<script lang="ts" setup>
import { modelData, tagColor, type Model } from "@/core/models";
import { serverUrl } from "@/env";
import { Home, Menu } from "@vicons/ionicons5";
import {
  NButton,
  NCard,
  NDataTable,
  NDropdown,
  NIcon,
  NInput,
  NSpace,
  NTag,
  type DataTableColumns,
} from "naive-ui";
import type {
  Filter,
  FilterOption,
  FilterOptionValue,
} from "naive-ui/es/data-table/src/interface";
import type { DropdownMixedOption } from "naive-ui/es/dropdown/src/interface";
import { h, reactive, ref, type Component } from "vue";
import { useState } from "../store/state";
const conf = useState();

const customModel = ref("");

function downloadModel() {
  const url = new URL(`${serverUrl}/api/models/download`);
  url.searchParams.append("model", customModel.value);
  console.log(url);
  conf.state.downloading = true;
  customModel.value = "";
  fetch(url, { method: "POST" })
    .then(() => {
      conf.state.downloading = false;
    })
    .catch(() => {
      conf.state.downloading = false;
    });
}

const tagsFilterOptions = () => {
  // Return all the tags from the models
  const tagsFilterOptions: FilterOption[] = [];
  const tags: string[] = [];
  modelData.forEach((model) => {
    model.tags.forEach((tag) => {
      if (!tags.includes(tag)) {
        tags.push(tag);
      }
    });
  });

  tags.forEach((tag) => {
    tagsFilterOptions.push({
      label: tag,
      value: tag,
    });
  });

  return tagsFilterOptions;
};

const tagsFilter: Filter<Model> = (value: FilterOptionValue, row: Model) => {
  return row.tags.indexOf(value.toString()) ? false : true;
};

const getTagColor = (tag: string) => {
  return tagColor[tag];
};

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
    title: "Tags",
    key: "tags",
    filterOptions: tagsFilterOptions(),
    filter: tagsFilter,
    render(row) {
      return h(
        NSpace,
        {},
        {
          default: () =>
            row.tags.map((tag) => {
              return h(
                NTag,
                {
                  bordered: true,
                  type: getTagColor(tag),
                },
                { default: () => tag }
              );
            }),
        }
      );
    },
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
          onClick: () => {
            console.log("Download", row.name);
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

const columnsRef = reactive(columns);
const dataRef = reactive(modelData);
const pagination = reactive({ pageSize: 10 });
</script>

<style scoped>
.install {
  width: 100%;
  padding: 10px 0px;
}
</style>
