<template>
  <NCard title="Acceleration progress">
    <NSpace vertical justify="center">
      <NSteps :current="1">
        <NStep
          title="Start"
          description="Start the acceleration process by clicking the button next to the model"
        />
        <NStep
          title="Convert to ONNX"
          description="This process might take a while"
        />
        <NStep
          title="Convert to TensorRT"
          description="This process might take a while"
        />
        <NStep
          title="Package and cleanup"
          description="This process might take a while"
        /> </NSteps></NSpace
  ></NCard>
</template>

<script lang="ts" setup>
import { NCard, NSpace, NStep, NSteps } from "naive-ui";
</script>
