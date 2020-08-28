cmake_minimum_required(VERSION 3.1)


project( yolocv )

if (MINGW)
  set(OpenCV_DIR "D:/OpenCV/3.4.5/mingw530_32/install")
elseif (MSVC)
  set(OpenCV_DIR "D:/OpenCV/3.4.5/vs2015/install/x64/vc14/lib")
endif (MINGW)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )


set (CMAKE_CXX_STANDARD 11)


#add_executable( yolo3cv342 yolo3cv342.cpp )

#target_link_libraries( yolo3cv342 ${OpenCV_LIBS} )

#cmake . -G "Visual Studio 14 Win64"
#msbuild OpenCV_YOLO.sln
#cd Debug
#yolo3cv342 --config=/.../yolov3.cfg --model=/.../yolov3.weights --classes=/.../coco.names --input=/.../img.jpg --rgb --width=608 --height=608


add_executable( yolo2cv340 yolo2cv340.cpp )

target_link_libraries( yolo2cv340 ${OpenCV_LIBS} )

#cmake . -G "Unix Makefiles"
#make
#yolo2cv340 -cfg=/.../yolov2.cfg  -model=/.../yolov2.weights  -class_names=/.../coco.names  -source=/.../img.jpg  -min_confidence=0.20


add_executable( yolo3cv345 yolo3cv345.cpp common.hpp )

target_link_libraries( yolo3cv345 ${OpenCV_LIBS} )

#yolo3cv345 --config=D:\Cygwin\home\asen\.yolo\yolov3.cfg --model=D:\Cygwin\home\asen\.yolo\yolov3.weights --classes=D:\Cygwin\home\asen\.yolo\coco.names --input=D:\Cygwin\home\asen\work\Soft\AI\images\street4.jpg --width=416 --height=416 --scale=0.00392 --rgb --thr=0.25
