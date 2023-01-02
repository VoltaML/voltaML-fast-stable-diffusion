<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <NSpace justify="end" inline align="center" class="install">
    <NInput
      v-model:value="customModel"
      placeholder="Custom model"
      style="width: 350px"
    />
    <NButton type="primary" bordered secondary style="margin-right: 24px"
      >Install</NButton
    >
  </NSpace>
  <n-data-table
    :columns="columnsRef"
    :data="dataRef"
    :pagination="pagination"
    :bordered="false"
  />
</template>

<script lang="ts" setup>
import { modelData, tagColor, type Model } from "@/core/models";
import { Home, Menu } from "@vicons/ionicons5";
import {
  NButton,
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

const tagsFilterOptions = () => {
  // Return all the tags from the models
  const tags: FilterOption[] = [];
  modelData.forEach((model) => {
    model.tags.forEach((tag) => {
      if (!tags.includes({ label: tag, value: tag })) {
        tags.push({ label: tag, value: tag });
      }
    });
  });
  return tags;
};

const tagsFilter: Filter<Model> = (value: FilterOptionValue, row: Model) => {
  return row.tags.indexOf(value.toString()) ? false : true;
};

const getTagColor = (tag: string) => {
  return tagColor[tag];
};

const customModel = ref("");

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

const createColumns2 = (): DataTableColumns<Model> => [
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
    title: "Example",
    key: "example",
    render(row) {
      return h(
        NButton,
        {
          type: "tertiary",
          secondary: true,
          round: true,
          block: true,
          bordered: false,
          onClick: () => {
            console.log("Example", row.name);
          },
        },
        { default: () => "Show me" }
      );
    },
  },
  {
    title: "Acceelerate",
    key: "accelerate",
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
            console.log("Accelerate", row.name);
          },
        },
        { default: () => "Accelerate" }
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
        },
        { default: renderIcon(Menu) }
      );
    },
    filter: "default",
  },
];

let columns = createColumns2();

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
