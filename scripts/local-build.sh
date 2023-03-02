docker build -t voltaml_local -f ./dockerfiles/dockerfile.local . \
&& docker tag voltaml_local:latest stax124/volta_diffusion_webui:local \
&& docker rmi $(docker images --filter "dangling=true" -q --no-trunc)