import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import csv

# Función para calcular SSIM y PSNR
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

# Función para procesar una imagen
def process_image(image_path, contador_imagen):
    print("Procesando imagen ", contador_imagen)
    imagen = cv2.imread(image_path)
    
    ''' 1. PreProcesamiento (Difuminado) '''
    # 1.1 Aumentar el brillo de la imagen
    factor_aumento_brillo = 1.5 # Ajusta el valor según se prefiera
    imagen_alto_brillo = cv2.convertScaleAbs(imagen, alpha=factor_aumento_brillo, beta=0)

    # 1.2 
    # Oscurecer imagen
    darkened_image = (imagen * 0.5).astype(np.uint8)  # Ajusta el valor 0.5 según se prefiera

    ''' 2. Realzado '''
    # 2.1 Ecualización de Histograma
    hist_equalization_result = apply_histogram_equalization(imagen_alto_brillo)

    # 2.2 Filtro Homomórfico
    cutoff_frequency = 32 # Frecuencia de corte
    order = 2
    high_boost = 2.0
    enhanced_image = homomorphic_filter(darkened_image, cutoff_frequency, order, high_boost)
    
    ''' 3. Comparación de Métricas '''
    ssim_value_equalization, psnr_value_equalization = evaluate_metrics(imagen, hist_equalization_result)
    ssim_value_homomorphic, psnr_value_homomorphic = evaluate_metrics(imagen, enhanced_image)
    
    return ssim_value_equalization, psnr_value_equalization, ssim_value_homomorphic, psnr_value_homomorphic

# Carpeta de imágenes
input_folder = 'D:\\NARANJAS\\Img\\test'  # Reemplaza con la ruta de tu carpeta de imágenes
output_file = 'resultados.csv'

# Lista de rutas de imágenes en la carpeta, agregar mas extensiones de archivo de imagen si se necesita
image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(('.jpg', '.png', '.jpeg'))]
cont=1
# Procesar las imágenes en secuencia
results = []
for image_path in image_paths:
    result = process_image(image_path, cont)
    results.append((image_path,) + result)
    cont += 1

# Guardar los resultados en un archivo CSV
with open(output_file, mode='w', newline='') as csvfile:
    fieldnames = ['Imagen', 'SSIM Equalization', 'PSNR Equalization', 'SSIM Homomorphic', 'PSNR Homomorphic']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    promedio_ssim_equalization = 0
    promedio_psnr_equalization = 0
    promedio_ssim_homomorphic = 0
    promedio_psnr_homomorphic = 0
    for result in results:
        promedio_ssim_equalization = promedio_ssim_equalization + result[1]
        promedio_psnr_equalization = promedio_psnr_equalization + result[2]
        promedio_ssim_homomorphic = promedio_ssim_homomorphic + result[3]
        promedio_psnr_homomorphic = promedio_psnr_homomorphic + result[4]
        writer.writerow({
            'Imagen': os.path.basename(result[0]),
            'SSIM Equalization': result[1],
            'PSNR Equalization': result[2],
            'SSIM Homomorphic': result[3],
            'PSNR Homomorphic': result[4]
        })
promedio_ssim_equalization = promedio_ssim_equalization/cont
promedio_psnr_equalization = promedio_psnr_equalization/cont
promedio_ssim_homomorphic = promedio_ssim_homomorphic/cont
promedio_psnr_homomorphic = promedio_psnr_homomorphic/cont
print("Proceso completo. Los resultados se han guardado en", output_file)
print("Promedio SSIM Equalization", promedio_ssim_equalization)
print("Promedio PSNR Equalization", promedio_psnr_equalization)
print("Promedio SSIM Homomorphic", promedio_ssim_homomorphic)
print("Promedio PSNR Homomorphic", promedio_psnr_homomorphic)

