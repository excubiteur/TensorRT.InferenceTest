Detects faces on image file "image.jpg".
Rectangles are drawn on faces in "image.jpg" and saved as "results.jpg"

To run:
Open a terminal in this folder and do the following:
make clean
make
./Serialize.exe
./InferFromEngine.exe
Open "results.jpg" and see if rectangles are drawn over the correct locations.

Works on Nano Jetpack 4.3
Does not work on Xavier NX Jetpack 4.4 GA (see rectangles in the wrong places)


https://forums.developer.nvidia.com/t/xavier-nx-jetpack-4-4-ga-trt-gives-wrong-results-for-a-specific-caffe-model/141529
