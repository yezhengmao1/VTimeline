#!/bin/bash

# build 3rd
cd libs/spdlog
[ -d "build" ] && rm -rf build
mkdir build && cd build && cmake ../ && make -j4 && make install && cd ../../../

# build cupti extension so
[ -d "build" ] && rm -rf build
[ -e "libvtimeline.so" ] && rm libvtimeline.so.0.0.1
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release ../ && make && mv libvtimeline.so.0.0.1 ../libvtimeline.so && cd ../

# build the wheel
mv libvtimeline.so src/vtimeline/
pip wheel --no-build-isolation . -w dist/

# clean
rm src/vtimeline/libvtimeline.so