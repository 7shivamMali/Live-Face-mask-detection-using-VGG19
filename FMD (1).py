import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# load the model
model = tf.keras.models.load_model("FMD_VGG19_D1.h5")

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

input_img = cv2.imread('sample images/without_mask_2007.jpg')
input_img = cv2.resize(input_img,(224,224))
pred = detect_face_mask(input_img)
input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
if pred == 0:
    draw_label(input_img,"mask",(0,255,0))
else:
    draw_label(input_img,"no mask",(255,0,0))

plt.imshow(input_img)
plt.axis('off') 
plt.show()
