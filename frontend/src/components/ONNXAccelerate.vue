<template>
  <div style="margin: 16px">
    <NCard title="Acceleration progress (around 5 minutes)">
      <NSpace vertical justify="center">
        <NSteps>
          <NStep title="CLIP" :status="global.state.onnxBuildStep.clip" />
          <NStep title="UNet" :status="global.state.onnxBuildStep.unet" />
          <NStep title="VAE" :status="global.state.onnxBuildStep.vae" />
          <NStep
            title="Cleanup"
            :status="global.state.onnxBuildStep.cleanup"
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

      <!-- Downcast to FP16 operations -->
      <div class="flex-container">
        <p class="slider-label">Downcast to FP16</p>
        <NSwitch v-model:value="conf.data.settings.onnx.convert_to_fp16" />
      </div>

      <!-- Quantization -->
      <h3>Quantization</h3>
      <div class="flex-container">
        <p class="slider-label">Text Encoder</p>
        <NSelect
          v-model:value="conf.data.settings.onnx.quant_dict.text_encoder"
          :options="[
            { label: 'No quantization', value: 'no-quant' },
            { label: 'Unsigned int8 (cpu only)', value: 'uint8' },
            { label: 'Signed int8', value: 'int8' },
          ]"
          style="margin-right: 12px"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">UNet</p>
        <NSelect
          v-model:value="conf.data.settings.onnx.quant_dict.unet"
          :options="[
            { label: 'No quantization', value: 'no-quant' },
            { label: 'Unsigned int8 (cpu only)', value: 'uint8' },
            { label: 'Signed int8', value: 'int8' },
          ]"
          style="margin-right: 12px"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">VAE Encoder</p>
        <NSelect
          v-model:value="conf.data.settings.onnx.quant_dict.vae_encoder"
          :options="[
            { label: 'No quantization', value: 'no-quant' },
            { label: 'Unsigned int8 (cpu only)', value: 'uint8' },
            { label: 'Signed int8', value: 'int8' },
          ]"
          style="margin-right: 12px"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">VAE Decoder</p>
        <NSelect
          v-model:value="conf.data.settings.onnx.quant_dict.vae_decoder"
          :options="[
            { label: 'No quantization', value: 'no-quant' },
            { label: 'Unsigned int8 (cpu only)', value: 'uint8' },
            { label: 'Signed int8', value: 'int8' },
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
import { computed, ref } from "vue";
import { useSettings } from "../store/settings";

const message = useMessage();
const global = useState();
const conf = useSettings();

const model = ref("");

const building = ref(false);
const showUnloadModal = ref(false);

const modelOptions = computed(() => {
  const options: SelectOption[] = [];
  for (const model of global.state.models) {
    if (
      model.backend === "PyTorch" &&
      model.valid &&
      !model.name.endsWith(".safetensors") &&
      !model.name.endsWith(".ckpt")
    ) {
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
      convert_to_fp16: conf.data.settings.onnx.convert_to_fp16,
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
