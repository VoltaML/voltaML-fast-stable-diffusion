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
		search: {
			provider: "local",
		},
		nav: [
			{ text: "Home", link: "/" },
			{ text: "Docs", link: "/getting-started/introduction" },
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
			{
				icon: "linkedin",
				link: "https://www.linkedin.com/in/tom%C3%A1%C5%A1-nov%C3%A1k-5a163321b/",
			},
			{
				icon: "linkedin",
				link: "https://www.linkedin.com/in/márton-kissik/",
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
					{ text: "Windows", link: "/installation/windows" },
					{ text: "Linux", link: "/installation/linux" },
					{ text: "WSL", link: "/installation/wsl" },
					{ text: "Vast.ai", link: "/installation/vast" },
					{ text: "Docker", link: "/installation/docker" },
					{ text: "Old", link: "/installation/old" },
					{ text: "xFormers", link: "/installation/xformers" },
				],
				collapsed: false,
			},
			{
				text: "Basics",
				items: [
					{ text: "Models", link: "/basics/models" },
					{ text: "LoRA", link: "/basics/lora" },
					{ text: "Textual Inversion", link: "/basics/textual-inversion" },
					{
						text: "AITemplate Acceleration",
						link: "/basics/aitemplate",
					},
				],
				collapsed: false,
			},
			{
				text: "WebUI",
				items: [
					{ text: "Models", link: "/webui/models" },
					{ text: "Image to Image", link: "/webui/img2img" },
					{ text: "Extra", link: "/webui/extra" },
					{ text: "Downloading Models", link: "/webui/download" },
					{ text: "Image browser", link: "/webui/imagebrowser" },
					{ text: "Convert", link: "/webui/convert" },
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
				text: "Settings",
				items: [{ text: "Settings", link: "/settings/settings" }],
			},
			{
				text: "Experimental",
				items: [],
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
					{ text: "API", link: "/developers/api" },
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
	},
	cleanUrls: true,
	ignoreDeadLinks: "localhostLinks",
	markdown: {
		theme: "one-dark-pro",
		lineNumbers: true,
	},
});
