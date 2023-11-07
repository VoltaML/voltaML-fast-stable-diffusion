<!-- eslint-disable vue/valid-template-root -->
<template></template>

<script setup lang="ts">
import { useState } from "@/store/state";

console.log(
  `
  ██╗   ██╗ █████╗ ██╗   ████████╗ █████╗ ███╗   ███╗██╗     
  ██║   ██║██╔══██╗██║   ╚══██╔══╝██╔══██╗████╗ ████║██║     
  ╚██╗ ██╔╝██║  ██║██║      ██║   ███████║██╔████╔██║██║     
   ╚████╔╝ ██║  ██║██║      ██║   ██╔══██║██║╚██╔╝██║██║     
    ╚██╔╝  ╚█████╔╝███████╗ ██║   ██║  ██║██║ ╚═╝ ██║███████╗
     ╚═╝    ╚════╝ ╚══════╝ ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
  `
);

const global = useState();

global.fetchCapabilites().then(() => {
  console.log("Capabilities successfully fetched from the server");
});

global.fetchAutofill();
</script>
