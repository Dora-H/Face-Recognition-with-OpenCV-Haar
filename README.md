# Face-Recognition-with-OpenCV-Haar
Simple detects front faces, noses, eyes by using OpenCV Harr. 

## Requirements
● Python 3    
● cv2   


## Class
FaceRecognition


## Functions
● classifiers  
● show


## Create __init__
#### define paths of all required harr xml :
    def __init__(self):
        self.pathA = 'path of your harr eye.xml'
        self.pathB = 'path of your harr face.xml'
        self.pathC = 'path of your harr nose.xml'
        self.pathF = 'path of this code .jpg'


## Run codes
#### 1. Call the main finction to work, start.
    if __name__ == "__main__":
        run = FaceRecognition()
        run.classifiers()

#### 2. load OpenCV face detector :
    def classifiers(self):
        eye_cascade = cv2.CascadeClassifier(self.pathA)
        nose_cascade = cv2.CascadeClassifier(self.pathC)
        face_cascade = cv2.CascadeClassifier(self.pathB)

##### read image
        image = cv2.imread(self.pathF)

##### check image height, width(to resize)
        height, width = image.shape[0], image.shape[1]
        adj_height, adj_width = int(height * 0.8), int(width * 0.4)
        adj_image = cv2.resize(image, (adj_height, adj_width))

##### list to hold all subject faces
        faces_holder = []
        
##### using Haar Cascade Classfiers
        faces = face_cascade.detectMultiScale(adj_image, 1.08, 5, minSize=(30, 30))
        
        for left, top, width, height in faces:
            # extract the faces area
            faces = adj_image[top:top+height, left:left+width]
            cv2.rectangle(adj_image, (left, top), (left+width, top+height),
                          (13, 230, 23), 2)

            # detect eyes area from faces area
            eyes = eye_cascade.detectMultiScale(faces, 1.12, 2,
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            for(ex, ey, ew, eh) in eyes:
                cv2.rectangle(faces, (ex, ey), (ex+ew, ew+(eh//2)), (23, 13, 277), 2)

            # detect noses area from faces area
            noses = nose_cascade.detectMultiScale(faces, 1.2, 5,
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
            for (nx, ny, nw, nh) in noses:
                cv2.rectangle(faces, (nx, ny), (nx+nw, ny+(nh//2)), (255, 0, 255), 2)

            if faces is not None:
                # add face to list of faces_holder
                faces_holder.append(faces)
        self.show(faces_holder, adj_image)
        

## 3. call show() function to show result :
    def show(self, holder, image):
        # print total faces (text color:white)
        text = f"Total faces: {len(holder)}"
        cv2.putText(image, text, (20, 620), cv2.FONT_HERSHEY_PLAIN, 2.8,
                    (255, 255, 255), 4)

        # display an image window to show the image
        cv2.imshow('Faces & Noses & Eyes Recognition', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
![Result](https://user-images.githubusercontent.com/70878758/130394409-953c5ab4-a3ed-460d-961e-f5d6a914104d.jpg)
