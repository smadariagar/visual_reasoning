import cv2
import numpy as np
from ultralytics import SAM

# --- Configuración ---
# Usamos el modelo 'sam_b.pt' (base). Es un buen equilibrio entre velocidad y precisión.
# Se descargará automáticamente la primera vez (aprox 375MB).
model = SAM('sam_b.pt')
image_path = 'test.png'  # Tu imagen con reflejos

# --- 1. Inferencia con IA ---
# El modelo analiza la imagen y devuelve resultados.
# 'conf=0.5' filtra objetos con baja confianza.
results = model(image_path, conf=0.5)

# Cargar la imagen original para dibujar sobre ella
img_original = cv2.imread(image_path)
if img_original is None:
    print("Error al cargar imagen")
    exit()

print(f"\nSe detectaron {len(results[0].masks)} objetos con IA.\n")

# --- 2. Procesar los resultados ---
# Iteramos sobre cada máscara detectada por la IA
for i, mask_data in enumerate(results[0].masks.data):
    # La máscara viene como un tensor en GPU/CPU, la pasamos a imagen numpy (blanco y negro)
    mask_np = mask_data.cpu().numpy().astype('uint8') * 255
    
    # Redimensionar la máscara si es necesario (a veces SAM devuelve un tamaño diferente)
    if mask_np.shape[:2] != img_original.shape[:2]:
        mask_np = cv2.resize(mask_np, (img_original.shape[1], img_original.shape[0]), interpolation=cv2.INTER_NEAREST)

    # --- 3. Encontrar el centroide (Usando método clásico sobre la máscara de IA) ---
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Tomamos el contorno más grande de la máscara (por si acaso hay ruido mínimo)
        largest_contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # --- Dibujar ---
            # Dibujar un círculo en el centro
            cv2.circle(img_original, (cX, cY), 8, (0, 0, 255), -1) # Punto rojo
            # Poner texto
            label = f"Obj {i+1}"
            cv2.putText(img_original, label, (cX - 20, cY - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            print(f"Objeto {i+1}: Centro en X={cX}, Y={cY}")

# --- Mostrar resultado final ---
# Reducimos un poco la imagen para que quepa en pantalla si es muy grande
scale_percent = 60 
width = int(img_original.shape[1] * scale_percent / 100)
height = int(img_original.shape[0] * scale_percent / 100)
dim = (width, height)
resized_show = cv2.resize(img_original, dim, interpolation = cv2.INTER_AREA)

cv2.imshow("Deteccion de Centros con IA (SAM)", resized_show)
cv2.waitKey(0)
cv2.destroyAllWindows()