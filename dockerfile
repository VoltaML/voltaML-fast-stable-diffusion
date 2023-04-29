FROM stax124/aitemplate:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install curl -y
RUN curl -sL https://deb.nodesource.com/setup_18.x | bash

RUN apt install nodejs -y

RUN npm i -g yarn
RUN apt install time git -y
RUN pip install --upgrade pip

WORKDIR /app

COPY requirements /app/requirements

RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements/api.txt
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements/bot.txt
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements/pytorch.txt
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements/interrogation.txt
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install python-dotenv
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -U xformers

COPY . /app

RUN --mount=type=cache,mode=0755,target=/app/frontend/node_modules cd frontend && yarn install && yarn build
RUN rm -rf frontend/node_modules

# Run the server
RUN chmod +x scripts/start.sh
ENTRYPOINT ["bash", "./scripts/start.sh"]
