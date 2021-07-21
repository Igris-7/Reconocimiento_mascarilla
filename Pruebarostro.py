import cv2
import os
from mtcnn.mtcnn import MTCNN

direccion = 'D:/Python/Examen/Fotos'
dire_img = os.listdir(direccion)
print ("Nombres:",dire_img)


reconocimiento = cv2.face.LBPHFaceRecognizer_create()

reconocimiento.read('ModeloRostros.xml')

detector = MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == False : break
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    copia = frame.copy()
    copia2 = gris.copy()

    caras = detector.detect_faces(copia)

    for i in range(len(caras)):
        x1,y1, ancho, alto = caras[i]['box']
        x2,y2 = x1+ancho , y1+alto
        cara_reg = copia2[y1:y2 , x1:x2]
        cara_rec = cv2.resize(cara_reg, (150,200), interpolation = cv2.INTER_CUBIC)
        resultado = reconocimiento.predict(cara_rec)

        if resultado[0]== 0:
            cv2.putText(frame,"Mascarilla Ailton",(x1,y1-5),2,1.3,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x1,y1),(x1+ancho, y1+alto), (0,255,0),2)
        elif resultado[0]== 1:
            cv2.putText(frame,"Mascarilla David",(x1,y1-5),2,1.3,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x1,y1),(x1+ancho, y1+alto), (0,255,0),2)
        elif resultado[0]== 2:
            cv2.putText(frame,"No Mascarilla Ailton",(x1,y1-5),2,1.3,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x1,y1),(x1+ancho, y1+alto), (0,255,0),2)
        elif resultado[0]== 3:
            cv2.putText(frame,"No Mascarilla David",(x1,y1-5),2,1.3,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x1,y1),(x1+ancho, y1+alto), (0,255,0),2)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,0)
        
    cv2.imshow("Reconocimiento", frame)

    t=cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
        
