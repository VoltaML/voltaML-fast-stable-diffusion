# Frontend

This is the documentation for setting up the WebUI for local development.

## 1. Clone the repository

```bash
git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion --branch experimental
```

## 2. Move into the frontend directory

```bash
cd voltaML-fast-stable-diffusion/frontend
```

## 3. Install dependencies

::: warning
Node.js version 18+ installed is required. (16 might work as well but it's not tested)
:::

::: info
If you are using Linux, you might need to use `sudo` before the command.
:::

Install yarn if you don't have it already.

```bash
npm install -g yarn
```

Install dependencies

```bash
yarn install
```

## 4. Run the development server

```bash
yarn dev
```

## 5. Open the WebUI

Open [http://127.0.0.1:5173/](http://127.0.0.1:5173/) with your browser to see the result.
