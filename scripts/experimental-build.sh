docker build -t voltaml_experimental -f ./dockerfiles/experimental . \
&& docker tag voltaml_experimental:latest voltaml/volta_diffusion_webui:experimental \
&& docker rmi $(docker images --filter "dangling=true" -q --no-trunc)