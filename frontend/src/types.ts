import type { GlobalThemeOverrides } from "naive-ui";

export type ExtendedThemeOverrides = GlobalThemeOverrides & {
  volta: {
    base: "light" | "dark" | undefined;
    blur: string | undefined;
    backgroundImage: string | undefined;
  };
};
