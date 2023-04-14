import { defineConfig } from "vitepress";

export default defineConfig({
	title: "VoltaML",
	description: "Easy to use, yet feature-rich WebUI",
	lang: "en-US",
	appearance: "dark",
	lastUpdated: true,
	base: "/voltaML-fast-stable-diffusion/",
	head: [
		[
			"link",
			{
				rel: "shortcut icon",
				type: "image/x-icon",
				href: "/voltaML-fast-stable-diffusion/favicon.ico",
			},
		],
		[
			"meta",
			{
				property: "og:image",
				content: "/voltaML-fast-stable-diffusion/volta-og-image.webp",
			},
		],
	],
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
				items: [
					{ text: "Introduction", link: "/getting-started/introduction" },
					{ text: "Features", link: "/getting-started/features" },
					{ text: "First image", link: "/getting-started/first-image" },
				],
				collapsed: false,
			},
			{
				text: "Installation",
				items: [
					{ text: "Local", link: "/installation/local" },
					{ text: "Docker", link: "/installation/docker" },
					{ text: "Old", link: "/installation/old" },
				],
				collapsed: false,
			},
			{
				text: "WebUI",
				items: [
					{ text: "Models", link: "/webui/models" },
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
				items: [
					{ text: "Setup", link: "/bot/setup" },
					{
						text: "Commands",
						link: "/bot/commands",
					},
				],
				collapsed: false,
			},
			{
				text: "Experimental",
				items: [
					{ text: "xFormers", link: "/experimental/xformers" },
					{ text: "Settings", link: "/experimental/settings" },
					{
						text: "Safetensors/CKPT support",
						link: "/experimental/checkpoints",
					},
				],
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
					{
						text: "Testing",
						link: "/developers/testing",
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
	ignoreDeadLinks: "localhostLinks",
	markdown: {
		theme: "one-dark-pro",
		lineNumbers: true,
	},
});
