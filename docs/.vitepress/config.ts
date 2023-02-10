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
			{
				icon: "discord",
				link: "https://discord.gg/pY5SVyHmWm",
			},
		],
		sidebar: [
			{
				text: "Introduction",
				items: [{ text: "Introduction", link: "/" }],
			},
			{
				text: "Installation",
				items: [
					{ text: "Windows", link: "/installation/windows" },
					{ text: "Linux", link: "/installation/linux" },
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
			},
		],
		algolia: {
			appId: "8WLJEL7XVD",
			apiKey: "19809754944322d77cacde9bc57875aa",
			indexName: "VoltaML docs",
		},
	},
});
