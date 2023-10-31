<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <div>
    <NInput
      v-model:value="settings.data.settings[props.tab].prompt"
      type="textarea"
      placeholder="Prompt"
      class="prompt"
      show-count
      @keyup="
        promptHandleKeyUp(
          $event,
          settings.data.settings[props.tab],
          'prompt',
          state
        )
      "
      @keydown="promptHandleKeyDown"
    >
      <template #suffix>
        <NTooltip>
          <template #trigger>
            <NIcon style="margin-top: 10px">
              <SettingsOutline />
            </NIcon>
          </template>

          <NForm :show-feedback="false">
            <NFormItem label="Prompt-to-Prompt preprocessing" class="form-item">
              <NSwitch
                v-model:value="settings.data.settings.api.prompt_to_prompt"
              />
            </NFormItem>
            <NFormItem label="Prompt-to-Prompt model" class="form-item">
              <NSelect
                filterable
                :consistent-menu-width="false"
                :options="[
                  {
                    value: 'lllyasviel/Fooocus-Expansion',
                    label: 'lllyasviel/Fooocus-Expansion',
                  },
                  {
                    value: 'daspartho/prompt-extend',
                    label: 'daspartho/prompt-extend',
                  },
                  {
                    value: 'succinctly/text2image-prompt-generator',
                    label: 'succinctly/text2image-prompt-generator',
                  },
                  {
                    value: 'Gustavosta/MagicPrompt-Stable-Diffusion',
                    label: 'Gustavosta/MagicPrompt-Stable-Diffusion',
                  },
                  {
                    value:
                      'Ar4ikov/gpt2-medium-650k-stable-diffusion-prompt-generator',
                    label:
                      'Ar4ikov/gpt2-medium-650k-stable-diffusion-prompt-generator',
                  },
                ]"
                v-model:value="
                  settings.data.settings.api.prompt_to_prompt_model
                "
              />
            </NFormItem>
            <NFormItem label="Prompt-to-Prompt device" class="form-item">
              <NSelect
                :options="[
                  {
                    value: 'gpu',
                    label: 'On-Device',
                  },
                  {
                    value: 'cpu',
                    label: 'CPU',
                  },
                ]"
                v-model:value="
                  settings.data.settings.api.prompt_to_prompt_device
                "
              />
            </NFormItem>
          </NForm>
        </NTooltip>
      </template>
      <template #count>{{ promptCount }}</template>
    </NInput>
    <NInput
      v-model:value="settings.data.settings[props.tab].negative_prompt"
      type="textarea"
      placeholder="Negative prompt"
      show-count
      @keyup="
        promptHandleKeyUp(
          $event,
          settings.data.settings[props.tab],
          'negative_prompt',
          state
        )
      "
      @keydown="promptHandleKeyDown"
    >
      <template #count>{{ negativePromptCount }}</template>
    </NInput>
  </div>
</template>

<script lang="ts" setup>
import {
  promptHandleKeyDown,
  promptHandleKeyUp,
  spaceRegex,
} from "@/functions";
import { useSettings } from "@/store/settings";
import { useState } from "@/store/state";
import type { InferenceTabs } from "@/types";
import { SettingsOutline } from "@vicons/ionicons5";
import {
  NForm,
  NFormItem,
  NIcon,
  NInput,
  NSelect,
  NSwitch,
  NTooltip,
} from "naive-ui";
import { computed, type PropType } from "vue";

const settings = useSettings();
const state = useState();

const props = defineProps({
  tab: {
    type: String as PropType<InferenceTabs>,
    required: true,
  },
});

const promptCount = computed(() => {
  return settings.data.settings[props.tab].prompt.split(spaceRegex).length - 1;
});
const negativePromptCount = computed(() => {
  return (
    settings.data.settings[props.tab].negative_prompt.split(spaceRegex).length -
    1
  );
});
</script>

<style lang="scss">
.prompt {
  .n-input__suffix {
    display: block;
  }
}
</style>

<style scoped>
.form-item {
  margin-bottom: 10px;
}
</style>
