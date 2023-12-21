import cv2
import tensorflow as tf

# load the model
model = tf.keras.models.load_model("FMD_VGG19_D1.h5")

# live face mask detection
cap = cv2.VideoCapture(0)

def detect_face_mask(img):
    y_pred = model.predict(img.reshape(1,224,224,3))
    return y_pred

def draw_label(image, label,col, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):

    height, width, _ = image.shape

    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size

    text_x = (width - text_width) // 2
    text_y = height - 20 

    rectangle_height = text_height + 10  
    rectangle_y = text_y - text_height - 5
    cv2.rectangle(image, (text_x - 5, rectangle_y), (text_x + text_width + 5, rectangle_y + rectangle_height),
                  (0, 0, 0), -1)  
    cv2.putText(image, label, (text_x, text_y), font, font_scale, col, font_thickness, cv2.LINE_AA)

haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    cord = haar.detectMultiScale(img)
    return cord

while True:
    ret,frame = cap.read()

    img = cv2.resize(frame,(224,224))

    y_pred = detect_face_mask(img)

    faceDet = detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))

    for x,y,w,h in faceDet:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)  

    if y_pred[0][1] <= 0.2:
        draw_label(frame,"mask",(0,255,0))
    else:
        draw_label(frame,"no mask",(0,0,255))


    cv2.imshow("window",frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()