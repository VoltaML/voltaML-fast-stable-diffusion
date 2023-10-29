<template>
  <div class="main-container">
    <NCard>
      <NTabs>
        <NTabPane name="Autoload">
          <AutoloadSettings />
        </NTabPane>
        <NTabPane name="Files & Saving">
          <FilesSettings />
        </NTabPane>
        <NTabPane name="Optimizations">
          <OptimizationSettings />
        </NTabPane>
        <NTabPane name="Reproducibility & Generation">
          <ReproducibilitySettings />
        </NTabPane>
        <NTabPane name="Live preview & UI">
          <UISettings />
        </NTabPane>
        <NTabPane name="Defaults">
          <FrontendSettings />
        </NTabPane>
        <NTabPane name="Bot">
          <BotSettings />
        </NTabPane>
        <NTabPane name="General">
          <GeneralSettings />
        </NTabPane>
        <NTabPane name="Extra">
          <ExtraSettings />
        </NTabPane>
        <NTabPane name="Theme">
          <ThemeSettings />
        </NTabPane>
        <NTabPane name="NSFW">
          <NSFWSettings />
        </NTabPane>

        <template #suffix>
          <NButton
            type="error"
            ghost
            style="margin-right: 12px"
            @click="resetSettings"
            >Reset Settings</NButton
          >
          <NButton type="success" ghost @click="saveSettings" :loading="saving"
            >Save Settings</NButton
          >
        </template>
      </NTabs>
    </NCard>
  </div>
</template>

<script lang="ts" setup>
import AutoloadSettings from "@/components/settings/AutoloadSettings.vue";
import BotSettings from "@/components/settings/BotSettings.vue";
import ThemeSettings from "@/components/settings/DefaultsSettings/ThemeSettings.vue";
import ExtraSettings from "@/components/settings/ExtraSettings.vue";
import FilesSettings from "@/components/settings/FilesSettings.vue";
import FrontendSettings from "@/components/settings/FrontendSettings.vue";
import GeneralSettings from "@/components/settings/GeneralSettings.vue";
import NSFWSettings from "@/components/settings/NSFWSettings.vue";
import OptimizationSettings from "@/components/settings/OptimizationSettings.vue";
import ReproducibilitySettings from "@/components/settings/ReproducibilitySettings.vue";
import UISettings from "@/components/settings/UISettings.vue";
import { serverUrl } from "@/env";
import { defaultSettings } from "@/settings";
import { useSettings } from "@/store/settings";
import {
  NButton,
  NCard,
  NTabPane,
  NTabs,
  useMessage,
  useNotification,
} from "naive-ui";
import { onUnmounted, ref } from "vue";

const message = useMessage();
const settings = useSettings();
const notification = useNotification();

const saving = ref(false);

function resetSettings() {
  // Deepcopy and assign
  Object.assign(
    settings.defaultSettings,
    JSON.parse(JSON.stringify(defaultSettings))
  );

  message.warning(
    "Settings were reset to default values, please save them if you want to keep them"
  );
}

function saveSettings() {
  saving.value = true;

  fetch(`${serverUrl}/api/settings/save`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(settings.defaultSettings),
  }).then((res) => {
    if (res.status === 200) {
      message.success("Settings saved successfully");
    } else {
      res.json().then((data) => {
        message.error("Error while saving settings");
        notification.create({
          title: "Error while saving settings",
          content: data.message,
          type: "error",
        });
      });
    }

    saving.value = false;
  });
}

onUnmounted(() => {
  saveSettings();
});
</script>
