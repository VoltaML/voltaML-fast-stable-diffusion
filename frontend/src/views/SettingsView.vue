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
import {
  AutoloadSettings,
  BotSettings,
  FilesSettings,
  FrontendSettings,
  GeneralSettings,
  NSFWSettings,
  OptimizationSettings,
  ReproducibilitySettings,
  ThemeSettings,
  UISettings,
} from "@/components";
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

  settings
    .saveSettings()
    .then(() => {
      message.success("Settings saved");
    })
    .catch((e) => {
      message.error("Failed to save settings");
      notification.create({
        title: "Failed to save settings",
        content: e,
        type: "error",
      });
    })
    .finally(() => {
      saving.value = false;
    });
}

onUnmounted(() => {
  saveSettings();
});
</script>
