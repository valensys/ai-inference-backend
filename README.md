# Prepare Ubuntu (tested on 22.04 and 24.04)

### Install packages

```bash
sudo apt install libeigen3-dev
sudo apt install nlohmann-json3-dev
```

### Move to directory /usr/local/src and clone dependencies

```bash
git clone https://github.com/nadjieb/cpp-mjpeg-streamer.git
git clone -b 4.7.0 https://github.com/opencv/opencv.git
git clone -b poco-1.13.3-release https://github.com/pocoproject/poco.git
```
Also download folder with pre-built libraries
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz
```

### Build and install libraries inside /usr/local/src directory

MJPG streamer
```bash
cd cpp-mjpeg-streamer && mkdir build && cd build
cmake ..
make && make install
cd ../../
```

POCO
```bash
cd poco && mkdir cmake_build && cd cmake_build
cmake ..
make && make install
cd ../../
```

OpenCV
```bash
cd opencv && mkdir build && cd build
cmake -DCMAKE_CXX_STANDARD=17 ..
make && make install
cd ../../
```

Extract the archive with OnnxRuntime and move it to /usr/local/onnxruntime
```bash
tar xfv onnxruntime-linux-x64-1.17.3.tgz -C /usr/local
ln -s /usr/local/onnxruntime-linux-x64-1.17.3 /usr/local/onnxruntime
```

### Move to /opt directory and clone a branch of "recognition" repository there
``` bash
cd /opt
git clone -b analytics-deploy-cpu --single-branch --depth=1 https://gitlab.controlsystems.ru/gis/video-analytics/recognition.git
```

### Build backend
```bash
cd /opt/recognition
mkdir build && cd build
cmake ..
make
```

### Set up system variable LD_LIBRARY_PATH (if it's not set yet)

```bash
export LD_LIBRARY_PATH=/usr/local/lib
```

### Change config file and launch binary
Check recognition/configs/main.cfg and change "host" and "port" according to requirements.
For example assign: host=10.128.0.9, port=5200
```bash
./Analytics
```
