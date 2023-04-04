<template>
  <NButton @click="show = true">Show Modal</NButton>
  <NModal v-model:show="show" closable mask-closable>
    <NCard title="Models" style="width: 70vw">
      <NTabs type="segment" style="height: 70vh">
        <NTabPane name="PyTorch" style="height: 100%">
          <NGrid :cols="2" :x-gap="8" style="height: 100%">
            <NGi>
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
                  <div>
                    <NButton type="success" ghost @click="selectedModel = model"
                      >Load</NButton
                    >
                    <NButton
                      type="info"
                      style="margin-left: 8px"
                      ghost
                      @click="selectedModel = model"
                      >Select</NButton
                    >
                  </div>
                </div>
              </NCard>
            </NGi>
            <NGi>
              <NCard :title="lora_title" style="height: 100%">
                <p>{{ selectedModel }}</p>
              </NCard>
            </NGi>
          </NGrid>
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
    </NCard>
  </NModal>
</template>

<script lang="ts" setup>
import { NButton, NCard, NGi, NGrid, NModal, NTabPane, NTabs } from "naive-ui";
import { computed, ref } from "vue";

const models = ["Model 1", "Model 2", "Model 3", "Model 4"];
const selectedModel = ref("");
const show = ref(true);
const lora_title = computed(() => {
  return `LoRA (${
    selectedModel.value ? selectedModel.value : "No model selected"
  })`;
});
</script>

<style scoped></style>
