<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <div class="flex-container">
    <div class="slider-label">
      <p>Enabled</p>
    </div>
    <NSwitch v-model:value="target[props.tab].scalecrafter.enabled" />
  </div>
  <NSpace
    vertical
    class="left-container"
    v-if="target[props.tab].scalecrafter.enabled"
  >
    <NAlert type="warning">
      Only works with <b class="highlight">Automatic</b> and
      <b class="highlight">Karras</b> sigmas
    </NAlert>

    <div class="flex-container">
      <NTooltip style="max-width: 600px">
        <template #trigger>
          <p class="slider-label">Disperse</p>
        </template>
        May generate more unique images.
        <b class="highlight">
          However, this comes at the cost of increased vram usage, generally in
          the range of 3-4x.
        </b>
      </NTooltip>
      <NSwitch v-model:value="target[props.tab].scalecrafter.disperse" />
    </div>

    <div class="flex-container">
      <NTooltip style="max-width: 600px">
        <template #trigger>
          <p class="slider-label">Unsafe resolutions</p>
        </template>
        Allow generating with unique resolutions that don't have configs ready
        for them, or clamp them (really, force them) to the closest resolution.
      </NTooltip>
      <NSwitch
        v-model:value="target[props.tab].scalecrafter.unsafe_resolutions"
      />
    </div>
  </NSpace>
</template>

<script setup lang="ts">
import type { ISettings } from "@/settings";
import { useSettings } from "@/store/settings";
import type { InferenceTabs } from "@/types";
import { NAlert, NSpace, NSwitch, NTooltip } from "naive-ui";
import { computed, type PropType } from "vue";

const settings = useSettings();

const props = defineProps({
  tab: {
    type: String as PropType<InferenceTabs>,
    required: true,
  },
  target: {
    type: String as PropType<"settings" | "defaultSettings">,
    required: false,
    default: "settings",
  },
});

const target = computed<ISettings>(() => {
  if (props.target === "settings") {
    return settings.data.settings;
  }

  return settings.defaultSettings;
});
</script>
