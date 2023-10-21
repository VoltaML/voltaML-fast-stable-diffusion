<template>
  <NConfigProvider :theme="theme" :theme-overrides="overrides" class="main">
    <NThemeEditor v-if="settings.data.settings.frontend.enable_theme_editor">
      <NNotificationProvider placement="bottom-right" :max="3">
        <NLoadingBarProvider>
          <NMessageProvider>
            <div id="background"></div>
            <SecretsHandlerVue />
            <CollapsileNavbarVue />
            <TopBarVue />
            <InitHandler />
            <routerContainerVue style="margin-top: 52px" />
            <PerformanceDrawer />
          </NMessageProvider>
        </NLoadingBarProvider>
      </NNotificationProvider>
    </NThemeEditor>
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
import { computed, ref, watch } from "vue";
import CollapsileNavbarVue from "./components/CollapsibleNavbar.vue";
import InitHandler from "./components/InitHandler.vue";
import PerformanceDrawer from "./components/PerformanceDrawer.vue";
import SecretsHandlerVue from "./components/SecretsHandler.vue";
import TopBarVue from "./components/TopBar.vue";
import { serverUrl } from "./env";
import routerContainerVue from "./router/router-container.vue";
import { useSettings } from "./store/settings";

const settings = useSettings();

type ExtendedThemeOverrides = GlobalThemeOverrides & {
  volta: {
    base: "light" | "dark" | undefined;
    blur: string | undefined;
    backgroundImage: string | undefined;
  };
};

const overrides = ref<ExtendedThemeOverrides | null>(null);
const theme = computed(() => {
  if (overrides.value?.volta?.base === "light") {
    return lightTheme;
  } else {
    return darkTheme;
  }
});

function updateTheme() {
  fetch(`${serverUrl}/themes/${settings.data.settings.frontend.theme}.json`)
    .then((res) => res.json())
    .then((data) => {
      overrides.value = data;
    });
}

updateTheme();
watch(() => settings.data.settings.frontend.theme, updateTheme);

const backgroundImage = computed(() =>
  overrides.value?.volta?.backgroundImage
    ? `url(${overrides.value?.volta?.backgroundImage})`
    : undefined
);
const blur = computed(() => `blur(${overrides.value?.volta?.blur ?? "6px"})`);
</script>

<style lang="scss">
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
  background-color: v-bind("theme.common.pressedColor");
  color: v-bind("theme.common.primaryColorHover");
}
#autocomplete-list {
  max-height: min(600px, 70vh);
  overflow-y: auto;
}

.n-card {
  backdrop-filter: v-bind(blur);
}

.navbar {
  .n-layout {
    backdrop-filter: v-bind(blur);
  }

  .n-layout-toggle-button {
    backdrop-filter: v-bind(blur);
  }
}

.top-bar {
  backdrop-filter: v-bind(blur);
  background-color: v-bind("overrides?.Card?.color");
}

.navbar {
  backdrop-filter: v-bind(blur);
}

#background {
  width: 100vw;
  height: 100vh;
  position: fixed;
  background-image: v-bind(backgroundImage);
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
  top: 0;
  left: 0;
  z-index: -99;
}
</style>
