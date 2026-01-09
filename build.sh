#!/bin/bash
if [ ! -d  "build" ];then
   mkdir build
else
   rm -rf build
   mkdir build
fi
cd build
echo "Begin compiling"
cmake .. -DCMAKE_INSTALL_PREFIX=../ -DCMAKE_PREFIX_PATH=/usr/local/libtorch
make
make install
cd ..