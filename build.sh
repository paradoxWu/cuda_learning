#!/bin/bash
if [ ! -d  "build" ];then
   mkdir build
else
   rm -rf build
   mkdir build
fi
cd build
echo "Begin compiling"
cmake .. -DCMAKE_INSTALL_PREFIX=../
make
make install
cd ..