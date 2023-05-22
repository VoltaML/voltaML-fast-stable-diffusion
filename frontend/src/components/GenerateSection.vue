<template>
  <!-- Generate button -->
  <NCard style="margin-bottom: 12px">
    <NGrid cols="2" x-gap="24">
      <NGi>
        <NButton
          type="success"
          ref="generateButton"
          @click="props.generate"
          :disabled="
            !props.doNotDisableGenerate &&
            (global.state.generating ||
              conf.data.settings.model?.name === '' ||
              conf.data.settings.model?.name === undefined)
          "
          :loading="global.state.generating"
          style="width: 100%"
          ghost
          >Generate
          <template #icon>
            <NIcon>
              <Play />
            </NIcon>
          </template>
        </NButton>
      </NGi>
      <NGi>
        <NButton
          type="error"
          @click="interrupt"
          style="width: 100%"
          ghost
          :disabled="!global.state.generating"
          >Interrupt
          <template #icon>
            <NIcon>
              <Skull />
            </NIcon>
          </template>
        </NButton>
      </NGi>
    </NGrid>
    <NAlert
      style="margin-top: 12px"
      v-if="
        conf.data.settings.model?.name === '' ||
        conf.data.settings.model?.name === undefined
      "
      type="warning"
      title="No model loaded"
      :bordered="false"
    >
    </NAlert>
  </NCard>
</template>

<script lang="ts" setup>
import { serverUrl } from "@/env";
import { useSettings } from "@/store/settings";
import { useState } from "@/store/state";
import { Play, Skull } from "@vicons/ionicons5";
import { NAlert, NButton, NCard, NGi, NGrid, NIcon } from "naive-ui";
import type { MaybeArray } from "naive-ui/es/_utils";
import { onMounted, onUnmounted, ref, type PropType } from "vue";

const global = useState();
const conf = useSettings();

const generateButton = ref<HTMLElement | null>(null);

onMounted(() => {
  window.addEventListener("keydown", handleKeyDown);
});

onUnmounted(() => {
  window.removeEventListener("keydown", handleKeyDown);
});

function handleKeyDown(e: KeyboardEvent) {
  // Press the generate button if ctrl+enter is pressed
  if (e.key === "Enter" && e.ctrlKey) {
    e.preventDefault();
    if (global.state.generating) {
      return;
    }
    const fn = props.generate as Function;
    fn(e as unknown as MouseEvent);
  }

  if (e.key === "Escape") {
    e.preventDefault();
    interrupt();
  }
}

function interrupt() {
  fetch(`${serverUrl}/api/general/interrupt`, {
    method: "POST",
  }).then((res) => {
    if (res.status === 200) {
      global.state.generating = false;
    }
  });
}

const props = defineProps({
  generate: {
    type: Function as unknown as PropType<MaybeArray<(e: MouseEvent) => void>>,
    required: true,
  },
  doNotDisableGenerate: {
    type: Boolean,
    default: false,
  },
});
</script>
