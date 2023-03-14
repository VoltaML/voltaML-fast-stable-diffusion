import { defineConfig } from "vitepress";

export default defineConfig({
	title: "VoltaML documentation",
	description: "VoltaML fast Stable Diffusion",
	lang: "en-US",
	appearance: "dark",
	lastUpdated: true,
	base: "/voltaML-fast-stable-diffusion/",
	themeConfig: {
		nav: [
			{ text: "Home", link: "/" },
			{ text: "Docs", link: "/introduction" },
		],
		editLink: {
			pattern:
				"https://github.com/VoltaML/voltaML-fast-stable-diffusion/edit/main/docs/:path",
		},
		logo: "/volta-rounded.webp",
		socialLinks: [
			{
				icon: "github",
				link: "https://github.com/voltaML/voltaML-fast-stable-diffusion",
			},
			{
				icon: "discord",
				link: "https://discord.gg/pY5SVyHmWm",
			},
		],
		sidebar: [
			{
				text: "Introduction",
				items: [{ text: "Introduction", link: "/introduction" }],
				collapsed: false,
			},
			{
				text: "Installation",
				items: [
					{ text: "Docker", link: "/installation/docker" },
					{ text: "Local", link: "/developers/pytorch" },
				],
				collapsed: false,
			},
			{
				text: "WebUI",
				items: [
					{ text: "Text to Image", link: "/webui/txt2img" },
					{ text: "Image to Image", link: "/webui/img2img" },
					{ text: "Extra", link: "/webui/extra" },
					{ text: "Download", link: "/webui/download" },
					{ text: "Accelerate", link: "/webui/accelerate" },
					{ text: "Image browser", link: "/webui/imagebrowser" },
					{ text: "Convert", link: "/webui/convert" },
					{ text: "Settings", link: "/webui/settings" },
				],
				collapsed: false,
			},
			{
				text: "Bot",
				items: [{ text: "Bot", link: "/bot/" }],
				collapsed: false,
			},
			{
				text: "API",
				items: [{ text: "API", link: "/api/" }],
				collapsed: false,
			},
			{
				text: "Developers",
				items: [
					{ text: "PyTorch", link: "/developers/pytorch" },
					{
						text: "Frontend",
						link: "/developers/frontend",
					},
					{
						text: "Documentation",
						link: "/developers/documentation",
					},
				],
				collapsed: false,
			},
			{
				text: "Troubleshooting",
				items: [
					{ text: "Linux", link: "/troubleshooting/linux" },
					{
						text: "Windows",
						link: "/troubleshooting/windows",
					},
					{
						text: "Docker",
						link: "/troubleshooting/docker",
					},
				],
				collapsed: false,
			},
		],
		algolia: {
			appId: "M9XJK5W9ML",
			apiKey: "8447ad2a43b65f2c280b8c883c76dc3f",
			indexName: "voltaml-fast-stable-diffusion",
		},
	},
	cleanUrls: true,
});
