# Set up the environment
source /opt/intel/openvino/bin/setupvars.sh


python3 face_recognition_demo.py \
  -i /dev/video0 \
  -m_fd intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml \
  -m_lm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
  -m_reid intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml \
#  --verbose \
#  -fg "/home/amazin/face_gallery" \
#  --run_detector 
  
 
