import cv2


class FaceRecognition(object):
    def __init__(self):
        self.pathA = 'path of your harr eye.xml'
        self.pathB = 'path of your harr face.xml'
        self.pathC = 'path of your harr nose.xml'
        self.pathF = 'path of this code .jpg'

    def show(self, holder, image):
        text = f"Total faces: {len(holder)}"
        cv2.putText(image, text, (20, 620), cv2.FONT_HERSHEY_PLAIN, 2.8,
                    (255, 255, 255), 4)

        cv2.imshow('Faces & Noses & Eyes Recognition', image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def classifiers(self):
        eye_cascade = cv2.CascadeClassifier(self.pathA)
        nose_cascade = cv2.CascadeClassifier(self.pathC)
        face_cascade = cv2.CascadeClassifier(self.pathB)

        image = cv2.imread(self.pathF)
        height, width = image.shape[0], image.shape[1]
        adj_height, adj_width = int(height * 0.8), int(width * 0.4)
        adj_image = cv2.resize(image, (adj_height, adj_width))

        faces_holder = []
        faces = face_cascade.detectMultiScale(adj_image, 1.08, 5, minSize=(30, 30))
        
        for left, top, width, height in faces:
            faces = adj_image[top:top+height, left:left+width]
            cv2.rectangle(adj_image, (left, top), (left+width, top+height),
                          (13, 230, 23), 2)

            eyes = eye_cascade.detectMultiScale(faces, 1.12, 2, flags=cv2.CASCADE_SCALE_IMAGE)
            for(ex, ey, ew, eh) in eyes:
                cv2.rectangle(faces, (ex, ey), (ex+ew, ew+(eh//2)), (23, 13, 277), 2)

            noses = nose_cascade.detectMultiScale(faces, 1.2, 5, flags=cv2.CASCADE_SCALE_IMAGE)
            for (nx, ny, nw, nh) in noses:
                cv2.rectangle(faces, (nx, ny), (nx+nw, ny+(nh//2)), (255, 0, 255), 2)

            if faces is not None:
                # add face to list of faces_holder
                faces_holder.append(faces)

        self.show(faces_holder, adj_image)


if __name__ == "__main__":
    run = FaceRecognition()
    run.classifiers()



