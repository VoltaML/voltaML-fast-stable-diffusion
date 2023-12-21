import { useMediaQuery } from "@vueuse/core";

export const isLargeScreen = useMediaQuery("(min-width: 1000px)");
