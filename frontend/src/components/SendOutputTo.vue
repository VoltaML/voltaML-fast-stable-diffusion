<template>
  <!-- Generate button -->
  <NCard style="margin: 12px 0" title="Send To" v-if="output && card">
    <NGrid cols="4" x-gap="4" y-gap="4">
      <NGi>
        <NButton type="default" @click="toImg2Img" style="width: 100%" ghost
          >Img2Img</NButton
        >
      </NGi>
      <NGi>
        <NButton type="default" @click="toControlNet" style="width: 100%" ghost
          >ControlNet</NButton
        >
      </NGi>
      <NGi>
        <NButton type="default" @click="toInpainting" style="width: 100%" ghost
          >Inpainting</NButton
        >
      </NGi>
      <NGi>
        <NButton type="default" @click="toUpscale" style="width: 100%" ghost
          >Upscale</NButton
        >
      </NGi>
    </NGrid>
  </NCard>
  <NGrid cols="4" x-gap="4" y-gap="4" v-else-if="output">
    <NGi>
      <NButton type="default" @click="toImg2Img" style="width: 100%" ghost
        >Img2Img</NButton
      >
    </NGi>
    <NGi>
      <NButton type="default" @click="toControlNet" style="width: 100%" ghost
        >ControlNet</NButton
      >
    </NGi>
    <NGi>
      <NButton type="default" @click="toInpainting" style="width: 100%" ghost
        >Inpainting</NButton
      >
    </NGi>
    <NGi>
      <NButton type="default" @click="toUpscale" style="width: 100%" ghost
        >Upscale</NButton
      >
    </NGi>
  </NGrid>
</template>

<script lang="ts" setup>
import { useState } from "@/store/state";
import { NButton, NCard, NGi, NGrid } from "naive-ui";
import { useRouter } from "vue-router";
import { useSettings } from "../store/settings";
const router = useRouter();

const conf = useSettings();
const state = useState();

const props = defineProps({
  output: {
    type: String,
    required: true,
  },
  card: {
    type: Boolean,
    default: true,
  },
});

async function toImg2Img() {
  conf.data.settings.img2img.image = props.output;
  state.state.img2img.tab = "Image to Image";
  await router.push("/image2image");
}

async function toControlNet() {
  conf.data.settings.controlnet.image = props.output;
  state.state.img2img.tab = "ControlNet";
  await router.push("/image2image");
}

async function toInpainting() {
  conf.data.settings.inpainting.image = props.output;
  state.state.img2img.tab = "Inpainting";
  await router.push("/image2image");
}

async function toUpscale() {
  conf.data.settings.upscale.image = props.output;
  state.state.extra.tab = "Upscale";
  await router.push("/extra");
}
</script>
