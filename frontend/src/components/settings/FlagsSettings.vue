<template>
  <NCard title="Hi-res fix">
    <NForm>
      <NFormItem label="Scale" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.flags.highres.scale"
        />
      </NFormItem>
      <NFormItem label="Scaling Mode" label-placement="left">
        <NSelect
          :options="[
            {
              label: 'Nearest',
              value: 'nearest',
            },
            {
              label: 'Linear',
              value: 'linear',
            },
            {
              label: 'Bilinear',
              value: 'bilinear',
            },
            {
              label: 'Bicubic',
              value: 'bicubic',
            },
            {
              label: 'Bislerp',
              value: 'bislerp',
            },
            {
              label: 'Nearest Exact',
              value: 'nearest-exact',
            },
          ]"
          v-model:value="
            settings.defaultSettings.flags.highres.latent_scale_mode
          "
        />
      </NFormItem>
      <NFormItem label="Strength" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.flags.highres.strength"
        />
      </NFormItem>
      <NFormItem label="Steps" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.flags.highres.steps"
        />
      </NFormItem>
      <NFormItem label="Antialiased" label-placement="left">
        <NSwitch
          v-model:value="settings.defaultSettings.flags.highres.antialiased"
        />
      </NFormItem>
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import {
  NCard,
  NForm,
  NFormItem,
  NInputNumber,
  NSelect,
  NSwitch,
} from "naive-ui";
import { useSettings } from "../../store/settings";

const settings = useSettings();
</script>
