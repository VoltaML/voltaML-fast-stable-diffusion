# Documentation

This is will show you how to edit our documentation and how to properly contribute while outlining some rules for us.

## Rules

::: warning
Please read the rules before you start editing the documentation.
:::

- All images will be in WEBP format with maximal 90% image quality
- Images will be of sensible resolution (no 4k or higher resolution images)
- English only
- Grammatically correct when possible
- Keep it simple

## How to edit

All documentation is written in Markdown and is located in the `docs` folder. You can edit it directly on GitHub or you can clone the repository and edit it locally.

Edits on GitHub will create a Pull Request with the changes and they will be waiting for review.

Once the change is reviewed and approved it will be merged into the branch and will be deployed by our CI/CD pipeline.

## Running documentation locally

Clone the repository

```bash
git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion.git
```

Install dependencies

```bash
yarn install
```

Run the documentation

```bash
yarn docs:dev
```

You should now be able to access the documentation on `http://localhost:5173/voltaML-fast-stable-diffusion/`
