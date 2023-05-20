<template>
  <div style="margin: 16px">
    <NCard title="Acceleration progress (around 20 minutes)">
      <NSpace vertical justify="center">
        <NSteps>
          <NStep title="UNet" :status="global.state.aitBuildStep.unet" />
          <NStep
            title="ControlNet UNet"
            :status="global.state.aitBuildStep.controlnet_unet"
          />
          <NStep title="CLIP" :status="global.state.aitBuildStep.clip" />
          <NStep title="VAE" :status="global.state.aitBuildStep.vae" />
          <NStep
            title="Cleanup"
            :status="global.state.aitBuildStep.cleanup"
          /> </NSteps></NSpace
    ></NCard>

    <NCard style="margin-top: 16px">
      <!-- Model select -->
      <div class="flex-container">
        <p class="slider-label">Model</p>
        <NSelect
          v-model:value="model"
          :options="modelOptions"
          style="margin-right: 12px"
        />
      </div>

      <!-- Simplify UNet -->
      <div class="flex-container">
        <p class="slider-label">Simplify UNet</p>
        <NSwitch v-model:value="conf.data.settings.onnx.simplify_unet" />
      </div>

      <!-- Quantization -->
      <h3>Quantization</h3>
      <div class="flex-container">
        <p class="slider-label">Text Encoder</p>
        <NSelect
          v-model:value="proxyQuantDict.text_encoder"
          :options="[
            { label: 'None', value: 'null' },
            { label: 'True', value: 'true' },
            { label: 'False', value: 'false' },
          ]"
          style="margin-right: 12px"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">UNet</p>
        <NSelect
          v-model:value="proxyQuantDict.unet"
          :options="[
            { label: 'None', value: 'null' },
            { label: 'True', value: 'true' },
            { label: 'False', value: 'false' },
          ]"
          style="margin-right: 12px"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">VAE Encoder</p>
        <NSelect
          v-model:value="proxyQuantDict.vae_encoder"
          :options="[
            { label: 'None', value: 'null' },
            { label: 'True', value: 'true' },
            { label: 'False', value: 'false' },
          ]"
          style="margin-right: 12px"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">VAE Decoder</p>
        <NSelect
          v-model:value="proxyQuantDict.vae_decoder"
          :options="[
            { label: 'None', value: 'null' },
            { label: 'True', value: 'true' },
            { label: 'False', value: 'false' },
          ]"
          style="margin-right: 12px"
        />
      </div>
    </NCard>

    <NSpace vertical justify="center" style="width: 100%" align="center">
      <NButton
        style="margin-top: 16px; padding: 0 92px"
        type="success"
        ghost
        :loading="building"
        :disabled="building || modelOptions.length === 0"
        @click="showUnloadModal = true"
        >Accelerate</NButton
      >
    </NSpace>

    <NModal
      v-model:show="showUnloadModal"
      preset="dialog"
      title="Unload other models"
      width="400px"
      :closable="false"
      :show-close="false"
      content="Acceleration can be done with the other models loaded as well, but it will take a lot of resources. It is recommended to unload the other models before accelerating. Do you want to unload the other models?"
      positive-text="Unload models"
      negative-text="Keep models"
      @positive-click="accelerateUnload"
      @negative-click="accelerate"
    />
  </div>
</template>

<script lang="ts" setup>
import { serverUrl } from "@/env";
import { useState } from "@/store/state";
import {
  NButton,
  NCard,
  NModal,
  NSelect,
  NSpace,
  NStep,
  NSteps,
  NSwitch,
  useMessage,
  type SelectOption,
} from "naive-ui";
import { computed, reactive, ref, watch } from "vue";
import { useSettings } from "../store/settings";

const message = useMessage();
const global = useState();
const conf = useSettings();

const model = ref("");

const building = ref(false);
const showUnloadModal = ref(false);

const proxyQuantDict: {
  text_encoder: "null" | "true" | "false";
  vae_encoder: "null" | "true" | "false";
  vae_decoder: "null" | "true" | "false";
  unet: "null" | "true" | "false";
} = reactive({
  text_encoder: "null",
  vae_encoder: "null",
  vae_decoder: "null",
  unet: "null",
});

function quantDictStrToValue(value: "null" | "true" | "false"): boolean | null {
  if (value === "null") {
    return null;
  } else if (value === "true") {
    return true;
  } else if (value === "false") {
    return false;
  }
  return null;
}

watch(proxyQuantDict, () => {
  console.log("Change detected in proxyQuantDict");
  conf.data.settings.onnx.quant_dict = {
    text_encoder: quantDictStrToValue(proxyQuantDict.text_encoder),
    vae_encoder: quantDictStrToValue(proxyQuantDict.vae_encoder),
    vae_decoder: quantDictStrToValue(proxyQuantDict.vae_decoder),
    unet: quantDictStrToValue(proxyQuantDict.unet),
  };
});

const modelOptions = computed(() => {
  const options: SelectOption[] = [];
  for (const model of global.state.models) {
    if (model.backend === "PyTorch" && model.valid) {
      options.push({
        label: model.name,
        value: model.path,
      });
    }
  }
  return options;
});

model.value = modelOptions.value[0]?.value?.toString() ?? "";

const accelerateUnload = async () => {
  try {
    await fetch(`${serverUrl}/api/models/unload-all`, {
      method: "POST",
    });

    showUnloadModal.value = false;
    await accelerate();
  } catch {
    showUnloadModal.value = false;
    message.error("Failed to unload, check the console for more info.");
  }
};

const accelerate = async () => {
  showUnloadModal.value = false;
  building.value = true;
  await fetch(`${serverUrl}/api/generate/generate-onnx`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model_id: model.value,
      quant_dict: conf.data.settings.onnx.quant_dict,
      simplify_unet: conf.data.settings.onnx.simplify_unet,
    }),
  })
    .then(() => {
      building.value = false;
    })
    .catch(() => {
      building.value = false;
      message.error("Failed to accelerate, check the console for more info.");
    });
};
</script>
