docker build -t voltaml_network -f ./dockerfiles/network . && docker rmi $(docker images --filter "dangling=true" -q --no-trunc)
docker tag voltaml_network:latest voltaml/volta_diffusion_webui:latest