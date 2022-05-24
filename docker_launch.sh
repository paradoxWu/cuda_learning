docker run --gpus '"device=0"' -it --rm --net=host --ipc=host \
                    -e DISPLAY=$DISPLAY \
                    --privileged \
                    -e XAUTHORITY=$XAUTH \
                    -u root \
                    --entrypoint=bash \
                    --ulimit core=-1 \
                    --security-opt seccomp=unconfined \
                    -v ${PWD}:/cuda_learning \
                    -w /cuda_learning \
                    781f4835ff4e