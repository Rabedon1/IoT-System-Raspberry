#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Detección de Aglomeraciones para Raspberry Pi 4

"""

import cv2
from ultralytics import YOLO
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer
import paho.mqtt.client as mqtt
import json
import time
from collections import Counter
import warnings
import os
import gc  # Para liberar memoria
warnings.filterwarnings('ignore')

print("="*80)
print(" Sistema de Detección de Aglomeraciones - Raspberry Pi 4")
print("="*80)

# ========== CONFIGURACIÓN MQTT ==========
BROKER = 'broker.hivemq.com'
PORT = 1883
TOPIC = 'aglomeraciones/raspberry_pi/datos'  # Topic único para Raspberry Pi

print(f"\n Configurando MQTT...")
print(f"   Broker: {BROKER}")
print(f"   Topic: {TOPIC}")

client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("   ✅ Conectado al broker MQTT")
    else:
        print(f"   ❌ Error de conexión MQTT: {rc}")

def on_disconnect(client, userdata, rc):
    print("   ⚠️  Desconectado del broker MQTT")
    if rc != 0:
        print("   🔄 Intentando reconectar...")

client.on_connect = on_connect
client.on_disconnect = on_disconnect

try:
    client.connect(BROKER, PORT, 60)
    client.loop_start()  # Loop en segundo plano
except Exception as e:
    print(f"   ❌ Error al conectar: {e}")

# ========== CARGA DE MODELOS ==========
print(f"\n🤖 Cargando modelos de IA...")

# Ruta del modelo YOLO 
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')

if not os.path.exists(MODEL_PATH):
    print(f"     Modelo no encontrado en: {MODEL_PATH}")
    print(f"   Buscando en directorio actual...")
    MODEL_PATH = 'best.pt'

print(f"    Cargando YOLO desde: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"    YOLO cargado")

# Inicializa HSEmotion 
print(f"    Cargando HSEmotion...")
import timm.models.efficientnet

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
torch.load = _original_torch_load
print(f"  HSEmotion cargado")

# ========== PARÁMETROS PARA RASPBERRY PI ==========
CONFIDENCE_THRESHOLD = 0.4  # Umbral más alto para reducir detecciones
AGGLOMERACION_THRESHOLD = 50
MQTT_INTERVAL = 15  # Envía cada 15 segundos
FRAME_WIDTH = 416   
FRAME_HEIGHT = 416
SKIP_FRAMES = 9    
PROCESSING_DELAY = 0.1  

# CONFIGURACIÓN DE GUARDADO DE IMÁGENES
SAVE_IMAGES = False  
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'capturas')  

# Crea la carpeta de capturas 
if SAVE_IMAGES:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print(f"\n💾 Guardado de imágenes ACTIVADO")
    print(f"   Carpeta: {IMAGES_DIR}")
    print(f"   Se guarda una imagen cada vez que se envían datos MQTT")

print(f"\n⚙️  Configuración:")
print(f"   Resolución: {FRAME_WIDTH}x{FRAME_HEIGHT}")
print(f"   Umbral confianza: {CONFIDENCE_THRESHOLD}")
print(f"   Umbral aglomeración: {AGGLOMERACION_THRESHOLD} personas")
print(f"   Intervalo MQTT: {MQTT_INTERVAL} segundos")
print(f"   Skip frames: Procesa 1 de cada {SKIP_FRAMES + 1}")

# ========== CONFIGURACIÓN DE CÁMARA RASPBERRY PI ==========
print(f"\n📹 Configurando cámara Raspberry Pi...")

# Configuración de la cámara Raspberry Pi 
try:
    from picamera2 import Picamera2
    print("    Usando Picamera2 (cámara Raspberry Pi)")
    
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(4)  # Espera a que la cámara se estabilice
    
    # Captura frames de warming para estabilizar la cámara
    print("    Calentando cámara...")
    for i in range(5):
        try:
            _ = picam2.capture_array()
            time.sleep(0.2)
        except:
            pass
    
    USE_PICAMERA = True
    print("   Cámara Raspberry Pi lista y estabilizada")
    
except ImportError:
    print("    Picamera2 no disponible, usando OpenCV...")
    USE_PICAMERA = False
    
    # Fallback a OpenCV (para cámara USB o si picamera2 no está instalado)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Limita FPS para mejor rendimiento
    
    if not cap.isOpened():
        print("   ❌ Error: No se pudo abrir la cámara")
        exit(1)
    
    print("   ✅ Cámara USB lista")

# ========== FUNCIÓN PARA CAPTURAR FRAME ==========
def capture_frame():
    """Captura un frame de la cámara (Picamera2 o OpenCV)"""
    if USE_PICAMERA:
        frame = picam2.capture_array()
        # Convierte de RGB a BGR para OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cap.read()
        if not ret:
            return None
    return frame

# ========== FUNCIÓN PARA PUBLICAR DATOS ==========
def publish_data(data):
    """Publica datos por MQTT"""
    try:
        payload = json.dumps(data)
        result = client.publish(TOPIC, payload)
        if result.rc == 0:
            print(f"📤 Datos enviados: P={data['conteo_personas']}, C={data['num_caras']}, E={data['emocion_dominante']}")
        else:
            print(f"⚠️  Error al publicar: {result.rc}")
    except Exception as e:
        print(f"❌ Error en publish: {e}")

# ========== FUNCIÓN PARA GUARDAR IMÁGENES ==========
def save_frame(frame, data):
    """Guarda el frame con información de detección"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captura_{timestamp}_P{data['conteo_personas']}_C{data['num_caras']}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)
    
    # Crea una copia del frame para agregar información
    info_frame = frame.copy()
    
    # Agrega texto con información
    cv2.putText(info_frame, f"Personas: {data['conteo_personas']}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(info_frame, f"Caras: {data['num_caras']}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(info_frame, f"Emocion: {data['emocion_dominante']}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(info_frame, f"Estado: {'AGLOMERACION' if data['aglomeracion'] else 'Normal'}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not data['aglomeracion'] else (0, 0, 255), 2)
    cv2.putText(info_frame, timestamp, (10, info_frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Guarda la imagen
    cv2.imwrite(filepath, info_frame)
    print(f"💾 Imagen guardada: {filename}")

# ========== BUCLE PRINCIPAL ==========
print(f"\n{'='*80}")
print(f"🚀 INICIANDO DETECCIÓN EN TIEMPO REAL")
print(f"{'='*80}")
print(f"💡 Presiona Ctrl+C para detener\n")

last_publish = time.time()
frame_count = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    while True:
        # Captura frame
        frame = capture_frame()
        if frame is None:
            print("⚠️  Error al capturar frame")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Procesa solo 1 de cada N frames para mejor rendimiento
        if frame_count % (SKIP_FRAMES + 1) != 0:
            time.sleep(PROCESSING_DELAY)  # Reduce carga de CPU
            continue
        
        # ========== DETECCIÓN DE PERSONAS ==========
        print(f"🔄 Procesando frame {frame_count}...")  # Debug
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=416)  # Fuerza tamaño pequeño
        detections = results[0].boxes
        conteo = len(detections)
        es_aglomeracion = conteo > AGGLOMERACION_THRESHOLD
        
        # Libera memoria 
        gc.collect()
        
        # ========== DETECCIÓN DE EMOCIONES (HSEmotion) ==========
        emociones = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                try:
                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                        continue
                    
                    emotion, scores = emotion_recognizer.predict_emotions(face_img, logits=True)
                    emociones.append(emotion)
                    
                except:
                    continue
        except:
            emociones = []
        
        # ========== ESTADÍSTICAS ==========
        if emociones:
            emotion_counts = Counter(emociones)
            num_caras = len(emociones)
            
            emociones_porcentaje = {}
            for emocion, count in emotion_counts.items():
                porcentaje = (count / num_caras) * 100
                emociones_porcentaje[emocion] = round(porcentaje, 2)
            
            emotion_dominante = emotion_counts.most_common(1)[0][0]
        else:
            emotion_dominante = "N/A"
            emociones_porcentaje = {}
            num_caras = 0
        
        # ========== PUBLICACIÓN MQTT Y GUARDADO DE IMÁGENES ==========
        if time.time() - last_publish > MQTT_INTERVAL:
            # Prepara los datos
            data = {
                "conteo_personas": conteo,
                "num_caras": num_caras,
                "aglomeracion": es_aglomeracion,
                "emociones": emociones,
                "emociones_porcentaje": emociones_porcentaje,
                "emocion_dominante": emotion_dominante,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dispositivo": "Raspberry Pi 4"
            }
            
            # Publica datos por MQTT
            publish_data(data)
            
            # Guarda la imagen con las detecciones
            if SAVE_IMAGES:
                # Crea un frame con las detecciones dibujadas
                output_frame = frame.copy()
                
                # Dibuja personas (azul)
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Dibuja caras (verde)
                for (x, y, w, h) in faces:
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Guarda el frame con información
                save_frame(output_frame, data)
            
            last_publish = time.time()
        
        # Pequeña pausa para no saturar la CPU
        time.sleep(0.01)

except KeyboardInterrupt:
    print(f"\n\n{'='*80}")
    print("🛑 Deteniendo sistema...")
    print(f"{'='*80}")

finally:
    # Limpieza
    if USE_PICAMERA:
        picam2.stop()
        print("✅ Cámara Raspberry Pi detenida")
    else:
        cap.release()
        print("✅ Cámara USB liberada")
    
    client.loop_stop()
    client.disconnect()
    print("✅ Desconectado de MQTT")
    print("\n👋 Sistema IoT detenido correctamente\n")
