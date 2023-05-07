import cv2
import sys

# get information values from user
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# read input (images like .png .jpg ..) with BGR2GRAY
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect the faces from image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

print("Found {0} faces!".format(len(faces)))

# draw rectangle around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)

# run with command $python face_detect.py [imagename.extention]
# ex: python face_detect.py input.jpg
