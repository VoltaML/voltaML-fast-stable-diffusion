docker build -t voltaml_network -f ./dockerfiles/dockerfile.network . \
&& docker tag voltaml_network:latest voltaml/volta_diffusion_webui:latest \
&& docker rmi $(docker images --filter "dangling=true" -q --no-trunc)