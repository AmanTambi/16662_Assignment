xhost +local:root
docker container prune -f

# Resolve X11 authority file (XAUTH is often unset; fall back to XAUTHORITY)
: "${XAUTH:=${XAUTHORITY:-$HOME/.Xauthority}}"

# Detect NVIDIA GPU — skip --gpus all if none present (so script runs on any PC)
GPU_FLAG=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_FLAG="--gpus all"
fi

docker run --privileged --rm -it \
    --name="team12" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$XAUTH:$XAUTH:ro" \
    --network host \
    -v "$(pwd)/src/devel_packages:/home/ros_ws/src/devel_packages" \
    -v "$(pwd)/data:/home/ros_ws/data" \
    -v "$(pwd)/guide_mode.py:/home/ros_ws/guide_mode.py" \
    -v "$(pwd)/reset_joints.py:/home/ros_ws/reset_joints.py" \
    -v "/etc/timezone:/etc/timezone:ro" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev:/dev" \
    -v "/dev/bus/usb:/dev/bus/usb" \
    $GPU_FLAG \
    team12 bash