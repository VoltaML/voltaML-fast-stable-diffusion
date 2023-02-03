docker build -t voltaml_local -f ./dockerfile/local . && docker rmi $(docker images --filter "dangling=true" -q --no-trunc)
docker tag voltaml_local:latest voltaml/volta_diffusion_webui:local