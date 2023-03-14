docker build -t volta -f ./dockerfiles/dockerfile.local . \
&& docker tag volta:latest stax124/volta:experimental \
&& docker rmi $(docker images --filter "dangling=true" -q --no-trunc)