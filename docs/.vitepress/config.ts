import { defineConfig } from "vitepress";

export default defineConfig({
	title: "VoltaML documentation",
	description: "VoltaML fast Stable Diffusion",
	lang: "en-US",
	appearance: "dark",
	lastUpdated: true,
	base: "/voltaML-fast-stable-diffusion/",
	themeConfig: {
		editLink: {
			pattern:
				"https://github.com/VoltaML/voltaML-fast-stable-diffusion/edit/main/docs/:path",
		},
		socialLinks: [
			{
				icon: "github",
				link: "https://github.com/voltaML/voltaML-fast-stable-diffusion",
			},
		],
		sidebar: [
			{
				text: "Introduction",
				items: [
					{ text: "Introduction", link: "/" },
					{ text: "Installation", link: "/installation" },
				],
			},
			{
				text: "WebUI",
				items: [{ text: "WebUI", link: "/webui/" }],
			},
			{
				text: "Bot",
				items: [{ text: "Bot", link: "/bot/" }],
			},
			{
				text: "API",
				items: [{ text: "API", link: "/api/" }],
			},
		],
	},
	cleanUrls: "with-subfolders",
});
