import type { Capabilities } from "@/core/interfaces";
import { serverUrl } from "@/env";

export const defaultCapabilities: Capabilities = {
  supported_backends: [["CPU", "cpu"]],
  supported_precisions_cpu: ["float32"],
  supported_precisions_gpu: ["float32"],
  supported_torch_compile_backends: ["inductor"],
  supported_self_attentions: [
    ["Cross-Attention", "cross-attention"],
    ["Subquadratic Attention", "subquadratic"],
    ["Multihead Attention", "multihead"],
  ],
  has_tensorfloat: false,
  has_tensor_cores: false,
  supports_xformers: false,
  supports_triton: false,
  supports_int8: false,
};

export async function getCapabilities() {
  try {
    const response = await fetch(`${serverUrl}/api/hardware/capabilities`);
    if (response.status !== 200) {
      console.error("Server is not responding");
      return defaultCapabilities;
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(error);
    return defaultCapabilities;
  }
}
