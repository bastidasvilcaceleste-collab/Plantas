import base64
from PIL import Image, UnidentifiedImageError

FORMATOS_VALIDOS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'}
MAX_SIZE_MB = 10


def validar_archivo_imagen(archivo):
    if archivo is None:
        raise ValueError("No se proporcionó ninguna imagen.")
    if hasattr(archivo, 'filename') and archivo.filename:
        ext = archivo.filename.rsplit('.', 1)[-1].lower()
        if ext not in FORMATOS_VALIDOS:
            raise ValueError(
                f"Formato no soportado: .{ext}. "
                f"Usa: {', '.join(sorted(FORMATOS_VALIDOS))}."
            )
    try:
        img = Image.open(archivo)
        img.verify()
        archivo.seek(0)
    except (UnidentifiedImageError, Exception):
        raise ValueError("El archivo no es una imagen válida o está corrupto.")


def imagen_a_base64(archivo):
    archivo.seek(0)
    img_bytes = archivo.read()
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    content_type = getattr(archivo, 'content_type', 'image/jpeg') or 'image/jpeg'
    return f"data:{content_type};base64,{b64}"
