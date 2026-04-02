#  Sistema IoT de Detección de Aglomeraciones y Análisis Emocional

Este proyecto es un sistema de **Edge Computing** diseñado para monitorear espacios públicos en tiempo real. Utiliza Inteligencia Artificial local para detectar aglomeraciones y analizar el estado emocional de las personas, transmitiendo los datos mediante el protocolo **MQTT** a un dashboard interactivo.

---

##  ¿Qué hace este proyecto?

El sistema funciona de manera autónoma en una Raspberry Pi 4 realizando las siguientes tareas:

1.  **Detección de Personas**: Utiliza un modelo **YOLO** entrenado para contar personas en tiempo real.
2.  **Análisis de Emociones**: Emplea el modelo **HSEmotion** para clasificar expresiones faciales en 8 categorías (Felicidad, Tristeza, Enojo, etc.).
3.  **Detección de Aglomeraciones**: Activa alertas automáticas cuando el número de personas supera un umbral configurado (ej. 50 personas).
4.  **Comunicación IoT**: Envía los resultados procesados a través del broker **HiveMQ** usando el protocolo MQTT.
5.  **Monitoreo Visual**: Muestra los datos en un **Dashboard Web** responsivo con gráficos dinámicos y alertas.

---

##  ¿Cuál es el rol de Python aquí?

Python actúa como el cerebro del sistema, realizando todo el procesamiento de manera local:
*   **Orquestación**: Coordina la cámara, los modelos de IA y la comunicación.
*   **Visión por Computadora**: Usa **OpenCV** para manejar el flujo de video.
*   **Inferencia local**: Ejecuta redes neuronales profundas (Deep Learning) directamente en el hardware.
*   **Lógica de Negocio**: Calcula estadísticas, promedios y estados de alerta antes de enviar la información.

---
 
## 🛠️Tecnologías Utilizadas

*   **Lenguaje**: Python 3.9+
*   **IA (Visión)**: YOLOv8 (Detección) y HSEmotion (Emociones)
*   **IoT**: Protocolo MQTT (paho-mqtt) y Broker HiveMQ
*   **Frontend**: HTML5, CSS3, JavaScript (MQTT.js, Chart.js)
*   **Hardware**: Raspberry Pi 4 

---

## 📁 Estructura Principal

*   `appDetect_raspberry.py`: Script principal para producción en Raspberry Pi.
*   `models/best.pt`: Modelo YOLO entrenado específicamente para este proyecto.
*   `templates/index.html`: Dashboard web de visualización.

---

## 📡 Arquitectura del Sistema

```mermaid
graph LR
    A[Cámara / Video] --> B[Python - Edge AI]
    B -->|Detección Personas| C[YOLO]
    B -->|Análisis Facial| D[HSEmotion]
    C & D --> E[Lógica de Datos]
    E -->|JSON via MQTT| F[HiveMQ Broker]
    F -->|Real-time Push| G[Dashboard Web]
```

---

## 👩‍💻 Autor
**Alexander Bedón** - 
*Investigación sobre detección de aglomeraciones con análisis de sentimientos mediante tecnologías IoT.*

---

## 📝 Notas Finales

Este proyecto demuestra la integración de múltiples tecnologías modernas:
- ✅ Inteligencia Artificial (YOLO, HSEmotion)
- ✅ Internet of Things (MQTT, Raspberry Pi)
- ✅ Desarrollo Web (Dashboard interactivo)
- ✅ Visión por Computadora (OpenCV)
- ✅ Cloud Computing (HiveMQ)



---


