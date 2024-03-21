import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('circulo.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# Definição dos valores minimo e max da mascara
# o magenta tem h=300 mais ou menos ou 150 para a OpenCV
image_lower_hsv = np.array([0, 1, 0])  
image_upper_hsv = np.array([170, 255, 255])


mask_hsv = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)
contornos, _ = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask_rgb = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2RGB) 
contornos_img = mask_rgb.copy() # Cópia da máscara para ser desenhada "por cima"





print("Quantidade de contornos encontrado: ", len(contornos))
sorted_contornos = sorted( contornos, key = cv2.contourArea, reverse= True)
coords_list=[]
for i, cont in enumerate(sorted_contornos[:2],1):
    
    #desenha contorno
    cv2.drawContours(contornos_img, cont, -1, [255, 0, 0], 2);

    #acha o meio
    M = cv2.moments(cont)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    coords = (cx,cy)
    coords_list.append(coords)



    #escreve coordenada
    cv2.putText(contornos_img, str (coords), (cont[0,0,0], cont[0,0,1]), cv2.FONT_HERSHEY_SIMPLEX,1,(200,50,0),2,cv2.LINE_AA)
    print("centro de massa na possição: ",coords)
    

    #desenha linha
    size = 20
    color = (128,128,0)
    cv2.line(contornos_img,(cx - size,cy),(cx + size,cy),color,2)
    cv2.line(contornos_img,(cx,cy - size),(cx, cy + size),color,2)


cv2.line(contornos_img, coords_list[0], coords_list[1], (153, 102, 204), 5)

sides = img.shape[1]
for i in coords_list:
    centro_altura = i[1]
    cv2.line(contornos_img, (0, centro_altura), (sides, centro_altura),(153, 102, 204), 5)






   

plt.figure(figsize=(8,6))
plt.imshow(contornos_img);
plt.show()

