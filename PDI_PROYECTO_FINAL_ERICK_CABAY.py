'''
//**********************************************************//
//                    PROYECTO FINAL                        //
//      PRE PROCESAMIENTO Y REALZADO DE IMAGENES            // 
//         	         ERICK CABAY YANQUI                     //
//**********************************************************//
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Funcion para evaluar métricas segun SSIM y PSNR
def evaluate_metrics(image1, image2):
    ssim_value = ssim(image1, image2, multichannel=True, win_size=3)  
    mse_value = np.mean((image1 - image2) ** 2)
    psnr_value = 10 * np.log10(255**2 / mse_value)
    return ssim_value, psnr_value

# Funcion Filtro Homomorfico
def homomorphic_filter(image, cutoff, order, high_boost):
    # Convertir la imagen a coma flotante
    image_float = image.astype(np.float32) / 255.0

    # Aplicar la transformada logarítmica a cada canal
    log_image = np.log1p(image_float)

    # Calcular la transformada de Fourier 2D de cada canal
    fft_channels = [np.fft.fft2(log_image[:, :, i]) for i in range(3)]

    # Crear un filtro homomórfico para cada canal
    rows, cols, _ = image.shape
    center_row, center_col = rows // 2, cols // 2
    H_channels = []

    for i in range(3):  # Para cada canal de color (R, G, B)
        H_channel = np.zeros((rows, cols), dtype=np.float32)
        for x in range(rows):
            for y in range(cols):
                d2 = (x - center_row) ** 2 + (y - center_col) ** 2
                H_channel[x, y] = (high_boost - 1) * (1 - np.exp(-d2 / (2 * cutoff ** 2))) + 1
        H_channels.append(H_channel)

    # Aplicar el filtro homomórfico a cada canal multiplicando en el dominio de la frecuencia
    filtered_channels = [fft_channels[i] * H_channels[i] for i in range(3)]

    # Calcular la transformada inversa de Fourier para cada canal
    filtered_images = [np.fft.ifft2(filtered_channels[i]) for i in range(3)]

    # Aplicar la transformación exponencial a cada canal para volver al espacio original
    enhanced_channels = [np.expm1(np.real(filtered_images[i])) for i in range(3)]

    # Combinar los canales en una imagen RGB
    enhanced_image = np.stack(enhanced_channels, axis=-1)

    # Asegurarse de que los valores estén en el rango válido [0, 255]
    enhanced_image = (enhanced_image * 255).clip(0, 255).astype(np.uint8)

    return enhanced_image

# Funcion Ecualizacion de Histograma
def apply_histogram_equalization(image):
    # Convertir la imagen a formato YUV
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Aplicar la ecualización del histograma solo al canal Y (luminancia)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # Convertir la imagen de nuevo a formato BGR
    result_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return result_image

# Cargar la imagen
imagen = cv2.imread('naranja.png')

''' 1. PreProcesamiento (Difuminado) '''
# 1.1 Aumentar el brillo de la imagen
factor_aumento_brillo = 1.5 # Ajusta el valor según se prefiera
imagen_alto_brillo = cv2.convertScaleAbs(imagen, alpha=factor_aumento_brillo, beta=0)

# 1.2 
# Oscurecer imagen
darkened_image = (imagen * 0.5).astype(np.uint8)  # Ajusta el valor 0.5 según se prefiera

''' 2. Realzado '''
# 2.1 Ecualizacion de Histograma
hist_equalization_result = apply_histogram_equalization(imagen_alto_brillo)

# 2.2 Filtro Homomorfico
cutoff_frequency = 32 # Frecuencia de corte
order = 2
high_boost = 2.0
enhanced_image = homomorphic_filter(darkened_image, cutoff_frequency, order, high_boost)



# # Calcular histogramas para cada canal de color (BGR)
# histograma_azul = cv2.calcHist([imagen], [0], None, [256], [0, 256])
# histograma_verde = cv2.calcHist([hist_equalization_result], [1], None, [256], [0, 256])

# # Tramar histogramas
# plt.figure(figsize=(8, 6))
# plt.title('Histograma de la Imagen (Canal BGR)')
# plt.xlabel('Valor de Pixel')
# plt.ylabel('Frecuencia')
# plt.xlim([0, 255])
# plt.plot(histograma_azul, color='blue', label='Azul')
# plt.plot(histograma_verde, color='green', label='Verde')
# plt.legend()
# plt.grid(True)
# plt.show()


'''Mostrar la imagen original, las imagenes con difuminado y las imagenes realzadas
 Borrar esta parte si solo se desea registrar los valores de la comparacion SSIM y PSNR sin mostrar las imagenes. '''
# Crear una figura de matplotlib con subtramas para mostrar todas las imágenes
plt.figure(figsize=(12, 8))

# Primera subtrama: Imagen original
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')

# Segunda subtrama: Imagen con brillo alto
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(imagen_alto_brillo, cv2.COLOR_BGR2RGB))
plt.title('Imagen con Brillo Alto')

# Tercera subtrama: Imagen oscurecida
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(darkened_image, cv2.COLOR_BGR2RGB))
plt.title('Imagen Oscurecida')

# Cuarta subtrama: Imagen ecualizada
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(hist_equalization_result, cv2.COLOR_BGR2RGB))
plt.title('Imagen Ecualizada')

# Quinta subtrama: Imagen con filtro homomórfico
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.title('Imagen con Filtro Homomórfico')

# Mostrar las subtramas
plt.tight_layout()
plt.show()

'''3. Comparacion de Metricas'''
ssim_value_equalization, psnr_value_equalization = evaluate_metrics(imagen, hist_equalization_result)
ssim_value_homomorfic, psnr_value_homomorfic = evaluate_metrics(imagen, enhanced_image)

print(f"SSIM : {ssim_value_equalization}\nPSNR: {psnr_value_equalization}")

print(f"SSIM : {ssim_value_homomorfic}\nPSNR: {psnr_value_homomorfic}")

# Esperar a que se presione una tecla y luego cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
