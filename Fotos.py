import cv2
from matplotlib import pyplot
import os
import imutils
from mtcnn.mtcnn import MTCNN

Nombres = 'Con_Mascarilla_Ailton'
Carpeta = 'D:/Python/Examen/Fotos' #Cambia a la ruta donde hayas almacenado Data

CarpetaFotos = Carpeta + '/' + Nombres

if not os.path.exists(Nombres):
	print('Carpeta creada: ',Nombres)
	os.makedirs(CarpetaFotos)

cap = cv2.VideoCapture(0)
detector = MTCNN()
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    copia = frame.copy()

    caras = detector.detect_faces(copia)

    for i in range(len(caras)):
        x1,y1,ancho,alto = caras[i]['box']
        x2,y2 = x1 + ancho, y1 + alto
        cara_reg = frame[y1:y2, x1:x2]
        cara_reg = cv2.resize(cara_reg,(150,200), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(CarpetaFotos + "/rostro_{}.jpg".format(count),cara_reg)
        count = count +1
    cv2.imshow("Entrenamiento", frame)

    t= cv2.waitKey(1)
    if t==27 or count >=300:
        break
                
cap.release()
cv2.destroyAllWindows()

