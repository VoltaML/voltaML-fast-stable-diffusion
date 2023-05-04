<template>
  <div style="padding: 12px">
    <NGrid cols="3" x-gap="12">
      <NGi>
        <NCard title="Model">
          <NUpload
            multiple
            directory-dnd
            :action="`${serverUrl}/api/models/upload-model`"
            :max="5"
            accept=".ckpt,.safetensors"
          >
            <NUploadDragger
              style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
              "
            >
              <div style="margin-bottom: 12px; display: block">
                <NIcon size="48" :depth="3">
                  <CloudUpload />
                </NIcon>
              </div>
              <NText style="font-size: 24px"> Model </NText>
              <NText style="font-size: 16px">
                Click or drag a model to this area to upload it to the server
              </NText>
            </NUploadDragger>
          </NUpload>
        </NCard>
      </NGi>
      <NGi>
        <NCard title="LoRA">
          <NUpload
            multiple
            directory-dnd
            :action="`${serverUrl}/api/models/upload-model?type=lora`"
            :max="5"
            accept=".ckpt,.safetensors"
          >
            <NUploadDragger
              style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
              "
            >
              <div style="margin-bottom: 12px; display: block">
                <NIcon size="48" :depth="3">
                  <CloudUpload />
                </NIcon>
              </div>
              <NText style="font-size: 24px"> LoRA </NText>

              <NText style="font-size: 16px">
                Click or drag a model to this area to upload it to the server
              </NText>
            </NUploadDragger>
          </NUpload>
        </NCard>
      </NGi>
      <NGi>
        <NCard title="Textual Inversion">
          <NUpload
            multiple
            directory-dnd
            :action="`${serverUrl}/api/models/upload-model?type=textual-inversion`"
            :max="5"
            accept=".pt"
          >
            <NUploadDragger
              style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
              "
            >
              <div style="margin-bottom: 12px; display: block">
                <NIcon size="48" :depth="3">
                  <CloudUpload />
                </NIcon>
              </div>
              <NText style="font-size: 24px"> Textual Inversion </NText>

              <NText style="font-size: 16px">
                Click or drag a model to this area to upload it to the server
              </NText>
            </NUploadDragger>
          </NUpload>
        </NCard>
      </NGi>
    </NGrid>
  </div>
</template>

<script lang="ts" setup>
import { serverUrl } from "@/env";
import { CloudUpload } from "@vicons/ionicons5";
import {
  NCard,
  NGi,
  NGrid,
  NIcon,
  NText,
  NUpload,
  NUploadDragger,
} from "naive-ui";
</script>
