<template>
  <div class="navbar">
    <n-layout style="height: 100%" has-sider>
      <n-layout-sider
        bordered
        collapse-mode="width"
        :collapsed-width="64"
        :width="240"
        :collapsed="collapsed"
        @collapse="collapsed = true"
        @expand="collapsed = false"
        style="height: 100%"
      >
        <NSpace vertical justify="space-between" style="height: 100%">
          <n-menu
            :collapsed="collapsed"
            :collapsed-width="64"
            :collapsed-icon-size="22"
            :options="menuOptionsMain"
          />
        </NSpace>
      </n-layout-sider>
    </n-layout>
  </div>
</template>

<script lang="ts" setup>
import {
  Albums,
  Download,
  Duplicate,
  Image,
  Images,
  Speedometer,
  StatsChart,
  Warning,
} from "@vicons/ionicons5";
import type { MenuOption } from "naive-ui";
import { NIcon, NLayout, NLayoutSider, NMenu, NSpace } from "naive-ui";
import type { Component } from "vue";
import { h, ref } from "vue";
import { RouterLink } from "vue-router";

function renderIcon(icon: Component) {
  return () => h(NIcon, null, { default: () => h(icon) });
}

const menuOptionsMain: MenuOption[] = [
  {
    label: () => h(RouterLink, { to: "/" }, { default: () => "Text to Image" }),
    key: "txt2img",
    icon: renderIcon(Image),
  },
  {
    label: () =>
      h(
        RouterLink,
        { to: "/image2image" },
        { default: () => "Image to Image" }
      ),
    key: "image2image",
    icon: renderIcon(Images),
  },
  {
    label: () => h(RouterLink, { to: "/extra" }, { default: () => "Extra" }),
    key: "extra",
    icon: renderIcon(Duplicate),
  },
  {
    label: () =>
      h(RouterLink, { to: "/download" }, { default: () => "Download" }),
    key: "download",
    icon: renderIcon(Download),
  },
  {
    label: () =>
      h(RouterLink, { to: "/accelerate" }, { default: () => "Accelerate" }),
    key: "plugins",
    icon: renderIcon(Speedometer),
  },
  {
    label: () =>
      h(
        RouterLink,
        { to: "/imageBrowser" },
        { default: () => "Image Browser" }
      ),
    key: "imageBrowser",
    icon: renderIcon(Albums),
  },
  {
    label: () => h(RouterLink, { to: "/stats" }, { default: () => "Stats" }),
    key: "stats",
    icon: renderIcon(StatsChart),
  },
  {
    label: () => h(RouterLink, { to: "/test" }, { default: () => "Test" }),
    key: "test",
    icon: renderIcon(Warning),
  },
];

let collapsed = ref(true);
</script>

<style>
.navbar {
  position: fixed;
  top: 0;
  left: 0;
  height: 100%;
  z-index: 1000;
}
</style>
