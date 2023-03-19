docker rm $(docker ps -aq) -f
docker rmi $(docker images --filter "dangling=true" -q --no-trunc)