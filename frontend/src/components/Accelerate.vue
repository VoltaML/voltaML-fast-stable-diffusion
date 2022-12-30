<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <NSpace justify="end" inline align="center" class="install">
    <NInput
      v-model:value="customModel"
      placeholder="Custom model"
      style="width: 350px"
    />
    <NButton
      type="primary"
      bordered
      secondary
      @click="installPlugin"
      style="margin-right: 24px"
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
import {
  LogoGithub,
  Menu,
  ReloadCircleOutline,
  TrashOutline,
} from "@vicons/ionicons5";
import {
  NButton,
  NDataTable,
  NDropdown,
  NIcon,
  NInput,
  NSpace,
  useNotification,
  type DataTableColumns,
} from "naive-ui";
import type { Type as ButtonType } from "naive-ui/es/button/src/interface";
import type { DropdownMixedOption } from "naive-ui/es/dropdown/src/interface";
import { h, reactive, ref, type Component } from "vue";
import { serverUrl } from "../env";

const customModel = ref("");

const notification = useNotification();

type Plugin = {
  id: number;
  enabled: boolean;
  repo_url: string;
  name: string;
  author: string;
  stars: number;
  forks: number;
  issues: number;
  license: string;
  traceback: string;
  short_traceback: string;
  exists: boolean;
  empty: boolean;
  row_loading: boolean;
};

const data: Plugin[] = [];

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

async function updateData() {
  const plugin_names = await (await fetch(serverUrl + "/api/plugins")).json();

  dataRef.splice(0, dataRef.length);

  for (const plugin_name of plugin_names) {
    const plugin = await (
      await fetch(`${serverUrl}/api/plugins/status/${plugin_name}`)
    ).json();
    dataRef.push(plugin);
  }
}

function getPluginOptions(row: Plugin) {
  const options: DropdownMixedOption[] = [
    {
      label: "GitHub",
      key: "github",
      icon: renderIcon(LogoGithub),
      props: {
        onClick: () => window.open(row.repo_url),
      },
    },
    {
      label: "Reload",
      key: "reload",
      icon: renderIcon(ReloadCircleOutline),
      props: {
        onClick: async () => {
          await fetch(`${serverUrl}/api/plugins/reload-plugin/${row.name}`, {
            method: "POST",
          })
            .catch(() => {
              notification.error({
                title: "Error",
                description: `Failed to reload ${row.name}`,
                duration: 5000,
              });
            })
            .then(() => {
              notification.success({
                title: "Success",
                description: `Reloaded ${row.name}`,
                duration: 5000,
              });
            });
        },
      },
    },
    {
      label: "Delete",
      key: "delete",
      icon: renderIcon(TrashOutline),
      props: {
        onClick: async () => {
          await fetch(`${serverUrl}/api/plugins/remove-plugin/${row.name}`, {
            method: "POST",
          });
          updateData();
        },
      },
    },
  ];
  return options;
}

const createColumns2 = ({
  openGitHub,
  togglePlugin,
  getStatusButtonType,
  getStatusText,
}: {
  openGitHub: (row: Plugin) => void;
  togglePlugin: (row: Plugin) => void;
  getStatusButtonType: (row: Plugin) => ButtonType;
  getStatusText: (row: Plugin) => string;
}): DataTableColumns<Plugin> => [
  {
    title: "Name",
    key: "name",
    sorter: "default",
  },
  {
    title: "Author",
    key: "author",
    sorter: "default",
  },
  {
    title: "GitHub",
    key: "repo_url",
    render(row) {
      return h(
        NButton,
        {
          type: "info",
          bordered: true,
          secondary: true,
          onClick: () => openGitHub(row),
          target: "_blank",
        },
        { default: () => "GitHub" }
      );
    },
    filter: "default",
  },
  {
    title: "Enabled",
    key: "enabled",
    render(row) {
      return h(
        NButton,
        {
          type: row.enabled ? "success" : "error",
          loading: row.row_loading,
          bordered: true,
          secondary: true,
          block: true,
          strong: true,
          onClick: () => togglePlugin(row),
        },
        { default: () => (row.enabled ? "Enabled" : "Disabled") }
      );
    },
  },
  {
    title: "Status",
    key: "status",
    render(row) {
      return h(
        NButton,
        {
          type: getStatusButtonType(row),
          bordered: true,
          secondary: true,
          block: true,
          style: "cursor: not-allowed",
        },
        { default: () => getStatusText(row) }
      );
    },
    filter: "default",
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

let columns = createColumns2({
  openGitHub(row: Plugin) {
    window.open(row.repo_url);
  },
  togglePlugin(row: Plugin) {
    let data;

    row.row_loading = true;

    if (row.enabled) {
      data = fetch(`${serverUrl}/api/plugins/disable-plugin/${row.name}`, {
        method: "POST",
      });
    } else {
      data = fetch(`${serverUrl}/api/plugins/enable-plugin/${row.name}`, {
        method: "POST",
      });
    }

    data.then((res) => {
      res.json().then((plugin: Plugin) => {
        row.enabled = plugin.enabled;
        row.row_loading = false;
      });
    });
  },
  getStatusButtonType(row: Plugin) {
    if (row.empty) {
      return "error";
    } else if (row.exists) {
      return "success";
    } else {
      return "warning";
    }
  },
  getStatusText(row: Plugin) {
    if (row.empty) {
      return "Empty";
    } else if (row.exists) {
      return "Exists";
    } else {
      return "Missing";
    }
  },
});

function installPlugin() {
  const plugin_name = prompt("Enter plugin url:");

  if (plugin_name) {
    fetch(
      `${serverUrl}/api/plugins/install-plugin?url=${encodeURIComponent(
        plugin_name
      )}`,
      {
        method: "POST",
      }
    ).then(() => {
      updateData();
    });
  }
}

const columnsRef = reactive(columns);
const dataRef = reactive(data);
const pagination = reactive({ pageSize: 10 });
updateData();
</script>

<style scoped>
.install {
  width: 100%;
  padding: 10px 0px;
}
</style>
