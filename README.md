# 16-662 Robot Autonomy

Dockerized ROS Noetic workspace for Franka Panda manipulation with MoveIt, frankapy, Azure Kinect, and Intel RealSense support.

## What's in the image

- ROS Noetic desktop-full (Ubuntu 20.04)
- MoveIt, franka-ros, panda_moveit_config
- frankapy + autolab_core
- PyTorch (CUDA 11.8)
- Azure Kinect SDK (k4a 1.4.1) + ROS driver
- Intel RealSense SDK (librealsense2) + `realsense2-camera` ROS wrapper
- easy_handeye, aruco_ros

## Prerequisites (host)

- **Docker** — https://docs.docker.com/engine/install/ubuntu/
- **NVIDIA Container Toolkit** (only if NVIDIA GPU present) — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  - The run script auto-detects GPU and skips `--gpus all` if none is found, so the image works on CPU-only PCs.
- Add your user to the `docker` group so you don't need `sudo` — https://docs.docker.com/engine/install/linux-postinstall

No host-side install is needed for RealSense, Azure Kinect, udev rules, or librealsense — everything lives inside the container.

## Setup

### 1. Clone

```bash
git clone <this-repo-url>
cd 16662_RobotAutonomy
```

### 2. Build the image

```bash
docker build -t frankapy_docker .
```

First build takes ~20–30 min and produces a ~25 GB image. Subsequent builds reuse cached layers.

### 3. Run

```bash
./run_docker.sh
```

Drops you into a `bash` shell inside the container at `/home/ros_ws`. The script:
- forwards X11 (GUIs like RViz, `realsense-viewer` work on host display)
- mounts `/dev` (USB cameras, Franka interface)
- enables host networking (robot comms)
- auto-attaches GPU if available

To open a second shell into the running container:

```bash
./terminal_docker.sh
```

## Using the Intel RealSense camera

Plug the camera into a **USB 3.0 port** (blue port; depth streaming fails on USB 2). Then:

**Option A — GUI test:**
```bash
realsense-viewer
```

**Option B — ROS driver + RViz:**
```bash
# terminal 1 (inside container)
roslaunch realsense2_camera rs_camera.launch \
    enable_depth:=true enable_color:=true align_depth:=true

# terminal 2 (./terminal_docker.sh)
rqt_image_view /camera/color/image_raw
# or:
rosrun rviz rviz
```

In RViz set Fixed Frame to `camera_link`, then `Add` → `Image` (`/camera/color/image_raw`) and `PointCloud2` (`/camera/depth/color/points`).

## Using the Franka robot

Replace `[control-pc-name]` with e.g. `iam-snowwhite`.

1. **Unlock joints** — ssh to the control PC, open Chrome to `https://172.16.0.2/desk/`, click **Click to unlock joints**:
   ```bash
   ssh -X student@[control-pc-name]
   google-chrome
   ```

2. **Start roscore** on the control PC:
   ```bash
   roscore
   ```

3. **Start frankapy control bridge** (from your frankapy directory):
   ```bash
   bash ./bash_scripts/start_control_pc.sh -u student -i [control-pc-name]
   ```
   Launches 3 terminals. Kill and rerun to reset.

4. **Run the Docker container:**
   ```bash
   bash run_docker.sh
   ```

5. **Launch MoveIt:**
   ```bash
   bash terminal_docker.sh
   roslaunch manipulation demo_frankapy.launch
   ```

6. **Run demo script:**
   ```bash
   bash terminal_docker.sh
   rosrun manipulation demo_moveit.py
   ```

## Portability to other PCs

The image is fully self-contained. To move it to another machine:

```bash
# on source PC
docker save frankapy_docker | gzip > frankapy_docker.tar.gz

# copy the ~8–10 GB tarball to the target PC, then:
docker load < frankapy_docker.tar.gz
./run_docker.sh
```

Target PC only needs Docker + (optional) NVIDIA toolkit. No RealSense, Azure Kinect, ROS, or udev setup required on the host.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `./run_docker.sh` exits with no prompt | `$XAUTH` was empty; the script now falls back to `$XAUTHORITY`. If still broken, run `bash -x ./run_docker.sh` to see the failing docker command. |
| `realsense-viewer` shows no device | Camera was plugged in after container start. Exit, replug, re-run `./run_docker.sh`. |
| RealSense depth stream is black | Using USB 2.0 port — switch to USB 3 (blue). |
| GUI apps fail to open | Run `xhost +local:root` on host (the script does this automatically). |
| Files created inside container are root-owned on host | `bash claim_files.sh` restores ownership in the `data/` shared volume. |
| `gpg: NO_PUBKEY` during build | Intel rotated the repo key. The Dockerfile fetches the current key (`FB0B24895113F120`) via HTTPS from Ubuntu's keyserver. |
| Build fails at `pip3 install numpy` with `X509_V_FLAG_NOTIFY_POLICY` error | Upstream `osrf/ros:noetic` has a broken `python3-openssl` / `cryptography` combo. The Dockerfile upgrades `pyopenssl` before pip calls to fix this. |

## Utility scripts

- `run_docker.sh` — build a fresh container (removes any previous one named `frankapy_docker`)
- `terminal_docker.sh` — attach another bash shell to the running container
- `claim_files.sh` — chown files in `data/` back to your host user
- `guide_mode.py`, `reset_joints.py` — Franka utility scripts, mounted into the container

## Credits

Based on [vib2810/16662_RobotAutonomy](https://github.com/vib2810/16662_RobotAutonomy). Additions in this fork: Intel RealSense SDK, portable run script (XAUTH fallback, GPU auto-detect), build-breakage fixes for current Intel GPG keys and upstream Python SSL.
