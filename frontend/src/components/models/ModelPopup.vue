<template>
  <NModal
    v-model:show="show"
    :title="model?.name as string + ' (by ' + model?.creator.username as string + ')' || 'Loading...'"
    preset="card"
    style="width: 90vw"
  >
    <NTabs justify-content="start" type="bar" v-model:value="tabValue" animated>
      <NTabPane
        v-for="subModel in props.model?.modelVersions"
        :name="subModel.name"
        :key="subModel.id"
        style="display: flex; flex-direction: column"
      >
        <NGrid cols="2">
          <NGi>
            <!-- Gallery -->
            <NCarousel
              style="height: 70vh; width: 100%"
              draggable
              :slides-per-view="2"
              effect="card"
              dot-type="line"
            >
              <div
                v-for="image in subModel.images"
                :key="image.hash"
                style="border-radius: 20px; overflow: hidden"
              >
                <img
                  :src="image.url"
                  :style="{
                    width: '100%',
                    filter: image.nsfw !== 'None' ? 'blur(12px)' : 'none',
                  }"
                />
              </div>
            </NCarousel>
          </NGi>
          <NGi>
            <!-- Main Content -->
            <NCard
              :title="subModel.name"
              style="height: auto"
              segmented
              hoverable
              :content-style="{
                paddingBottom: '8px',
              }"
            >
              <template #footer>
                <i
                  >Data provided by <a href="https://civitai.com">CivitAI</a>,
                  go and support them</i
                >
              </template>
              <div style="height: 90%">
                <NRate :value="subModel.stats.rating" allow-half readonly />
                <div style="line-height: 32px">
                  <NTag
                    v-for="tag in model?.tags"
                    :key="tag"
                    style="margin-right: 4px"
                  >
                    {{ tag }}
                  </NTag>
                </div>
                <NDescriptions
                  label-placement="left"
                  label-align="left"
                  bordered
                  :columns="1"
                  style="margin-top: 8px"
                >
                  <NDescriptionsItem label="Base Model">
                    {{ subModel.baseModel }}
                  </NDescriptionsItem>
                  <NDescriptionsItem label="Downloads">
                    {{ subModel.stats.downloadCount.toLocaleString() }}
                  </NDescriptionsItem>
                  <NDescriptionsItem label="Keywords">
                    {{
                      subModel.trainedWords.length !== 0
                        ? subModel.trainedWords
                        : "No keywords"
                    }}
                  </NDescriptionsItem>
                  <NDescriptionsItem label="Last Updated">
                    {{ dateFormat.format(new Date(subModel.updatedAt)) }}
                  </NDescriptionsItem>
                  <NDescriptionsItem label="Created">
                    {{ dateFormat.format(new Date(subModel.createdAt)) }}
                  </NDescriptionsItem>
                </NDescriptions>
              </div>
              <div
                style="
                  width: 100%;
                  display: inline-flex;
                  height: 40px;
                  align-items: center;
                  margin-top: 8px;
                "
              >
                <NSelect :options="generateDownloadOptions(subModel.files)" />
                <NButton style="margin-left: 4px" type="primary" ghost>
                  Download
                </NButton>
              </div>
            </NCard>
          </NGi>
        </NGrid>
      </NTabPane>
    </NTabs>
  </NModal>
</template>

<script setup lang="ts">
import type { ICivitAIModel, IFile } from "@/civitai";
import {
  NButton,
  NCard,
  NCarousel,
  NDescriptions,
  NDescriptionsItem,
  NGi,
  NGrid,
  NModal,
  NRate,
  NSelect,
  NTabPane,
  NTabs,
  NTag,
  type TabsInst,
} from "naive-ui";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { nextTick, ref, watch } from "vue";

const show = ref(true);

const props = defineProps<{
  model: ICivitAIModel | null;
}>();

const tabValue = ref<string | number>("");
const tabsInstRef = ref<TabsInst | null>(null);

const dateFormat = new Intl.DateTimeFormat(navigator.language, {
  year: "numeric",
  month: "long",
  day: "numeric",
  minute: "numeric",
  hour: "numeric",
});

watch(props, (newProps) => {
  if (newProps.model) {
    tabValue.value = newProps.model.modelVersions[0].name;
  }
  nextTick(() => {
    tabsInstRef.value?.syncBarPosition();
  });
});

function generateDownloadOptions(submodel: IFile[]): SelectMixedOption[] {
  return submodel.map((file) => ({
    label: file.name,
    value: file.downloadUrl,
  }));
}
</script>
