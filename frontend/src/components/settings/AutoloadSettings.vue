<template>
  <NForm>
    <NFormItem label="Model" label-placement="left">
      <NSelect
        multiple
        filterable
        :options="autoloadModelOptions"
        v-model:value="settings.defaultSettings.api.autoloaded_models"
      >
      </NSelect>
    </NFormItem>

    <NFormItem label="Textual Inversions" label-placement="left">
      <NSelect
        multiple
        filterable
        :options="textualInversionOptions"
        v-model:value="
          settings.defaultSettings.api.autoloaded_textual_inversions
        "
      >
      </NSelect>
    </NFormItem>

    <NCard title="VAE">
      <div style="width: 100%">
        <div
          v-for="model of availableModels"
          :key="model.name"
          style="display: flex; flex-direction: row; margin-bottom: 4px"
        >
          <NText style="width: 50%">
            {{ model.name }}
          </NText>
          <NSelect
            filterable
            :options="autoloadVaeOptions"
            v-model:value="autoloadVaeValue(model.path).value"
          />
        </div>
      </div>
    </NCard>
  </NForm>
</template>

<script lang="ts" setup>
import { NCard, NForm, NFormItem, NSelect, NText } from "naive-ui";
import { computed } from "vue";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const settings = useSettings();
const global = useState();

const textualInversions = computed(() => {
  return global.state.models.filter((model) => {
    return model.backend === "Textual Inversion";
  });
});

const textualInversionOptions = computed(() => {
  return textualInversions.value.map((model) => {
    return {
      value: model.path,
      label: model.name,
    };
  });
});

const availableModels = computed(() => {
  return global.state.models.filter((model) => {
    return (
      model.backend === "AITemplate" ||
      model.backend === "PyTorch" ||
      model.backend === "ONNX"
    );
  });
});

const availableVaes = computed(() => {
  return global.state.models.filter((model) => {
    return model.backend === "VAE";
  });
});

const autoloadModelOptions = computed(() => {
  return availableModels.value.map((model) => {
    return {
      value: model.path,
      label: model.name,
    };
  });
});

const autoloadVaeOptions = computed(() => {
  const arr = availableVaes.value.map((model) => {
    return {
      value: model.path,
      label: model.name,
    };
  });
  arr.push({ value: "default", label: "Default" });

  return arr;
});

const autoloadVaeValue = (model: string) => {
  return computed({
    get: () => {
      return settings.defaultSettings.api.autoloaded_vae[model] ?? "default";
    },
    set: (value: string) => {
      if (!value || value === "default") {
        delete settings.defaultSettings.api.autoloaded_vae[model];
      } else {
        console.log("Setting", model, "to", value);
        settings.defaultSettings.api.autoloaded_vae[model] = value;
      }
    },
  });
};
</script>
