
# change the build dir to wherever it is on your system
BUILD_DIR="/home/mitch/projects/HAMC/build"

cd ${BUILD_DIR}
mkdir -p Transpose_output/

PATHS[0]=${BUILD_DIR}/Transpose/Dataset/0
PATHS[1]=${BUILD_DIR}/Transpose/Dataset/1
PATHS[2]=${BUILD_DIR}/Transpose/Dataset/2
PATHS[3]=${BUILD_DIR}/Transpose/Dataset/3
PATHS[4]=${BUILD_DIR}/Transpose/Dataset/4
PATHS[5]=${BUILD_DIR}/Transpose/Dataset/5
PATHS[6]=${BUILD_DIR}/Transpose/Dataset/6
PATHS[7]=${BUILD_DIR}/Transpose/Dataset/7

count=0
for j in ${PATHS[@]}
do
   file=output$((count)).txt
   ./Transpose_test -e $j/output.raw -i $j/input0.raw > Transpose_output/$file
 count=$((count+1))
done
