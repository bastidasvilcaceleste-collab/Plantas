# 🌱 Plant Disease Detection System using Artificial Intelligence

## Descripción

Este proyecto consiste en el desarrollo de un sistema web inteligente para la detección de enfermedades en hojas de plantas mediante técnicas de Inteligencia Artificial y Deep Learning. La aplicación permite a los usuarios cargar una imagen de una hoja, la cual es analizada por un modelo de Redes Neuronales Convolucionales (CNN) entrenado con PyTorch para identificar la enfermedad presente y mostrar el nivel de confianza de la predicción.

El sistema fue desarrollado como parte del curso **Taller de Investigación I** de la carrera de Ingeniería de Sistemas e Informática.

---

## Objetivos

- Detectar enfermedades en hojas de plantas utilizando Inteligencia Artificial.
- Facilitar un diagnóstico rápido mediante visión por computadora.
- Integrar un modelo de Deep Learning dentro de una aplicación web desarrollada con Flask.
- Proporcionar una interfaz intuitiva para agricultores, estudiantes e investigadores.

---

## Tecnologías utilizadas

- Python
- Flask
- PyTorch
- OpenCV
- Pillow (PIL)
- HTML5
- CSS3
- JavaScript
- Bootstrap 5
- SQLite
- SQLAlchemy
- Git & GitHub

---

## Arquitectura del sistema

El proyecto está organizado bajo una arquitectura modular basada en Flask.

```
GIT DE CODIGO
│
├── ai_models
├── datasets
├── models
├── routes
├── services
├── static
├── templates
├── uploads
├── app.py
└── requirements.txt
```

---

## Funcionalidades principales

- Inicio de sesión de usuarios.
- Registro de usuarios.
- Carga de imágenes.
- Predicción automática mediante CNN.
- Visualización del porcentaje de confianza.
- Historial de predicciones.
- Dashboard con estadísticas.
- Gestión de usuarios.
- Interfaz responsive.

---

## Modelo de Inteligencia Artificial

El sistema utiliza un modelo de Redes Neuronales Convolucionales (CNN) desarrollado con PyTorch y entrenado mediante Transfer Learning para la clasificación de enfermedades en hojas de plantas.

El modelo analiza imágenes previamente procesadas y devuelve la clase predicha junto con el porcentaje de confianza correspondiente.

---

## Dataset

El entrenamiento del modelo se realizó utilizando un subconjunto del dataset PlantVillage, adaptado para las clases utilizadas en el proyecto.

---

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/bastidasvilcaceleste-collab/Plantas.git
```

Ingresar al proyecto:

```bash
cd Plantas
```

Crear un entorno virtual:

```bash
python -m venv venv
```

Activar el entorno virtual:

Windows

```bash
venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el proyecto:

```bash
python app.py
```

---

## Estructura del proyecto

```
app.py
templates/
static/
routes/
models/
services/
ai_models/
datasets/
uploads/
instance/
```

---

## Investigación

El desarrollo del proyecto se fundamentó en estándares de calidad y desarrollo de software, incluyendo:

- ISO 9001
- ISO/IEC 25010
- ISO/IEC 29119
- ISO/IEC 27001

Además, se aplicó una metodología basada en investigación científica, estado del arte, validación por expertos y planificación de pruebas.

---

## Autora

**Nikole Celeste Bastidas Vilca**

Ingeniería de Sistemas e Informática

Universidad Continental

Huancayo – Perú

2026

---

## Licencia

Proyecto desarrollado con fines académicos para el curso Taller de Investigación I.
