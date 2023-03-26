<template>
  <NButton @click="show = true">Show Modal</NButton>
  <NModal v-model:show="show" closable mask-closable>
    <NCard title="Models" style="width: 70vw">
      <NTabs type="segment" style="height: 70vh">
        <NTabPane name="PyTorch" style="height: 100%">
          <NCard title="Models" style="height: 100%">
            <div
              style="
                display: inline-flex;
                width: 100%;
                align-items: center;
                justify-content: space-between;
                border-bottom: 1px solid rgb(66, 66, 71);
              "
              v-for="model in models"
              v-bind:key="model"
            >
              <p>{{ model }}</p>
              <NButton type="success" ghost @click="selectedModel = model"
                >Select</NButton
              >
            </div>
          </NCard>
        </NTabPane>
        <NTabPane name="AITemplate">
          <div>Model 2</div>
        </NTabPane>
        <NTabPane name="TensorRT">
          <div>Model 2</div>
        </NTabPane>
        <NTabPane name="Extra">
          <div>Model 2</div>
        </NTabPane>
      </NTabs>
      <div style="display: inline-flex; width: 100%; margin-top: 12px">
        <NInput
          placeholder="Model Name"
          disabled
          style="justify-self: start; margin-right: 12px"
          v-bind:value="selectedModel"
        />
        <NSelect placeholder="GPU" style="margin-right: 12px" />
        <NButton style="justify-self: end" type="success" ghost>Load</NButton>
      </div>
    </NCard>
  </NModal>
</template>

<script lang="ts" setup>
import {
  NButton,
  NCard,
  NInput,
  NModal,
  NSelect,
  NTabPane,
  NTabs,
} from "naive-ui";
import { ref } from "vue";

const models = ["Model 1", "Model 2", "Model 3", "Model 4"];
const selectedModel = ref("");
const show = ref(true);
</script>

<style scoped></style>
