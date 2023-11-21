from pupil_apriltags import Detector
import cv2
imagepath = 'calibration/left_april_0.jpg'
img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
# img = cv2.flip(cv2.flip(img, 0), 1)
# print(img)
print(img.shape)
at_detector = Detector(
   families="tagStandard41h12",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)
# p[981.409 532.97]  f[1365.82 1366.29]
x = at_detector.detect(img, estimate_tag_pose=True, 
    camera_params=(1365.82, 1366.29, 981.409, 532.97), 
    tag_size=0.114)

x = at_detector.detect(img, estimate_tag_pose=True, 
    camera_params=(1364.88, 1364.17, 970.216, 582.394), 
    tag_size=0.114)
# x = at_detector.detect(img, estimate_tag_pose=True, 
#     camera_params=(), 
#     tag_size=0.114)


# [  f[]

### z is 96.5
### y is 114.7
### x is 0 
# pose_R = [[-0.83343159  0.54600808 -0.08524648]
#  [-0.26791058 -0.53412818 -0.80182979]
#  [-0.48333809 -0.64543184  0.59144064]]
# pose_t = [[ 0.03970234]
#  [-0.27958421]
#  [ 1.35536968]]

print(x)
# import pdb; pdb.set_trace()