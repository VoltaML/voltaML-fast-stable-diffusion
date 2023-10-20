<template>
  <NConfigProvider :theme="theme" :theme-overrides="overrides" class="main">
    <NThemeEditor v-if="settings.data.settings.frontend.enable_theme_editor" />
    <NNotificationProvider placement="bottom-right" :max="3">
      <NLoadingBarProvider>
        <NMessageProvider>
          <SecretsHandlerVue />
          <CollapsileNavbarVue />
          <TopBarVue />
          <InitHandler />
          <routerContainerVue style="margin-top: 52px" />
          <PerformanceDrawer />
        </NMessageProvider>
      </NLoadingBarProvider>
    </NNotificationProvider>
  </NConfigProvider>
</template>

<script setup lang="ts">
import {
  NConfigProvider,
  NLoadingBarProvider,
  NMessageProvider,
  NNotificationProvider,
  NThemeEditor,
  darkTheme,
  lightTheme,
  type GlobalThemeOverrides,
} from "naive-ui";
import { computed } from "vue";
import CollapsileNavbarVue from "./components/CollapsibleNavbar.vue";
import InitHandler from "./components/InitHandler.vue";
import PerformanceDrawer from "./components/PerformanceDrawer.vue";
import SecretsHandlerVue from "./components/SecretsHandler.vue";
import TopBarVue from "./components/TopBar.vue";
import routerContainerVue from "./router/router-container.vue";
import { useSettings } from "./store/settings";

const settings = useSettings();

const theme = computed(() => {
  if (settings.data.settings.frontend.theme === "dark") {
    document.body.style.backgroundColor = "#121215";
    return darkTheme;
  } else {
    document.body.style.backgroundColor = "white";
    return lightTheme;
  }
});

const backgroundColor = computed(() => {
  if (settings.data.settings.frontend.theme === "dark") {
    return "#121215";
  } else {
    return "#fff";
  }
});

const overrides: GlobalThemeOverrides = {
  common: {
    fontSize: "15px",
    fontWeight: "600",
  },
};
</script>

<style>
.main {
  background-color: v-bind(backgroundColor);
}

.autocomplete {
  position: relative;
  display: inline-block;
}
.autocomplete-items {
  position: absolute;
  z-index: 99;
  background-color: v-bind("theme.common.popoverColor");
  border-radius: v-bind("theme.common.borderRadius");
  padding: 2px;
}
.autocomplete-items div {
  padding: 8px;
  cursor: pointer;
  border-radius: v-bind("theme.common.borderRadius");
}
.autocomplete-active {
  background-color: v-bind("theme.common.pressedColor") !important;
  color: v-bind("theme.common.primaryColorHover") !important;
}
#autocomplete-list {
  max-height: min(600px, 70vh);
  overflow-y: auto;
}
</style>
