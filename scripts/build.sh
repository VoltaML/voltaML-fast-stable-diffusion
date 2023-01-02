docker build -t voltaml -f ./Dockerfile . && docker rmi $(docker images --filter "dangling=true" -q --no-trunc)
docker tag voltaml:latest voltaml/volta_diffusion_webui:latest