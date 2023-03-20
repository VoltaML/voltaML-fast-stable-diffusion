# Testing

::: warning
Automated CI/CD pipeline is still in development. We need more coverage, so, if you can make some useful ones, I will be very thankful as I never worked on any large scale project that required it.
:::

This repository includes automated tests to sanity check if the project is still working as expected.

## Installing requirements

Please make sure that you have all the dependencies installed before proceeding with testing. You can install them with this command:

```bash
pip install -r requirements/tests.txt
```

## Running with Yarn (or NPM)

::: tip
If you do not have Yarn installed, you can run all of these commands with NPM as well by replacing the word `yarn` with `npm`

Or you can install yarn with `npm i -g yarn`
:::

Recommended way of starting tests is by running this command:

```bash
yarn test
```

Or you can run all tests (including slow ones like AIT model compilation)

```bash
yarn test:full
```

## Running tests manually

To run tests manually, open the `package.json` file. It should include a JSON like object like this:

::: warning
This preview is not updated. Please check your local installation for the updated version
:::

```json
{
	"name": "voltaml-fast-stable-diffusion",
	"version": "0.0.3",
	"description": "Fast Stable Diffusion",
	"main": "index.js",
	"license": "GPL-3.0",
	"private": false,
	"devDependencies": {
		"vitepress": "^1.0.0-alpha.36",
		"vue": "^3.2.45"
	},
	"scripts": {
		"docs:dev": "vitepress dev docs",
		"docs:build": "vitepress build docs",
		"docs:preview": "vitepress preview docs",
		"test": "pytest -x", // [!code focus]
		"test:full": "pytest --run-optional-tests=slow --cov=core -x" // [!code focus]
	}
}
```

Feel free to copy-paste these lines into your terminal to run the test
