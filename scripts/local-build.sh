docker build -t voltaml_local -f ./dockerfiles/local . \
&& docker tag voltaml_local:latest voltaml/volta_diffusion_webui:local \
&& docker rmi $(docker images --filter "dangling=true" -q --no-trunc)