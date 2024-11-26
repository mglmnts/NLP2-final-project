# Proyecto Final de NLP
**Autores:** Miguel Montes y José Ridruejo

Aquí tienes el markdown con las instrucciones para ejecutar el modelo final:

# Instrucciones para ejecutar el modelo final

## 1. Entrenamiento del modelo

Para entrenar el modelo, debes ejecutar el siguiente comando en la terminal:

```bash
python -m src.train.runs
```

- **Parámetro `id`**: Si deseas especificar un parámetro `id` diferente para la ejecución, puedes hacerlo añadiendo `--id <tu_id>`. Si no se especifica, el valor por defecto es `"A"`.
  
  Ejemplo:
  ```bash
  python -m src.train.runs --id "B"
  ```

- El modelo entrenado se guardará en la siguiente ruta:  
  `data/final-model-train/<id>/runs`  
  Los modelos se guardarán como **checkpoints** para poder continuar con el entrenamiento más tarde o realizar evaluaciones.

## 2. Ejecutar el benchmark

Para ejecutar el benchmark del modelo, debes ejecutar el siguiente comando:

```bash
python -m src.train.benchmarks
```

**Importante**: Asegúrate de descomentar las llamadas a las funciones de **IFEval** dentro del código para que se realice la evaluación de los resultados. Esto te permitirá obtener métricas de precisión y rendimiento del modelo. Tambien puede ser necesario tener localizado el json que contiene los prompts del ifeval

---

# Guía de instalación
## 1. Introducción

Debido a que el software necesario no está preparado en las máquinas virtuales de Aulas Virtuales, los alumnos pueden acudir físicamente al laboratorio y usar el usuario `usuario`, el cual dispone del software necesario. Sin embargo, no es posible acceder a este usuario a través de Aulas Virtuales. 

Para que los estudiantes puedan realizar la instalación del software en los equipos del laboratorio, esta guía detalla cómo descargar e instalar:
- **Visual Studio Code**  
- **WSL con distribución Ubuntu**  
- **Python en Ubuntu**

Los programas instalados no se borrarán al reiniciar el equipo.

---

## 2. Instalación de Visual Studio Code

### 2.1. Descarga
1. Visita la página oficial de Visual Studio Code: [https://code.visualstudio.com/](https://code.visualstudio.com/).
2. Haz clic en el botón **Download** y selecciona la versión para Windows.

### 2.2. Instalación
1. Ejecuta el archivo descargado (`VSCodeSetup.exe`).
2. Sigue las instrucciones del instalador y acepta los términos y condiciones.
3. Selecciona las opciones adicionales deseadas y completa la instalación.

_Nota:_ Reinicia el ordenador y ejecuta Visual Studio Code. Si no aparece, contacta con STIC en el CAU o con Jaime Pizarroso.

### 2.3. Extensiones
Asegúrate de tener las siguientes extensiones instaladas:
1. Python
2. Pylance
3. Python Debugger
4. Remote (incluido WSL)
5. Jupyter
6. Jupyter Cell Tags
7. Jupyter Notebook Renderers
8. Jupyter Slide Show

---

## 3. Instalación de WSL con Ubuntu

### 3.1. Activación de WSL
1. Actualiza el kernel de WSL desde [https://aka.ms/wsl2kernel](https://aka.ms/wsl2kernel) o descarga el paquete de actualización directamente.
2. Ejecuta el archivo `wsl_update_x64.msi` e instala la actualización.
3. Configura WSL 2 como versión predeterminada con el comando:  
   ```bash
   wsl --set-default-version 2
   ```
4. Descarga el archivo de instalación de Ubuntu desde este enlace: [https://aka.ms/wslubuntu2204](https://aka.ms/wslubuntu2204).  
   _Nota:_ No uses Microsoft Store para esta descarga.
5. Renombra el archivo descargado `Ubuntu2204-221101.AppxBundle` a `Ubuntu2204-221101.zip` y extrae su contenido.
6. Navega a la carpeta extraída. Allí deberías ver los archivos como se muestra en la Figura 1.
7. Abre PowerShell, cambia el directorio a la carpeta extraída y ejecuta:  
   ```bash
   Add-AppxPackage .\Ubuntu2204.1.7.0_x64.appx
   ```
8. Añade la nueva distribución al PATH de Windows:  
   ```powershell
   $userenv = [System.Environment]::GetEnvironmentVariable("Path","User")
   [System.Environment]::SetEnvironmentVariable("PATH", $userenv + ";C:\Users\<user>\Ubuntu","User")
   ```
9. Establece la distribución de Ubuntu como predeterminada:  
   ```bash
   wsl --set-default Ubuntu
   ```

Verifica que Ubuntu se haya instalado correctamente ejecutándolo desde el menú de Inicio.

### 3.2. Configuración Inicial
1. Abre Ubuntu desde el menú de inicio.
2. Completa la configuración inicial.
3. Crea un nombre de usuario y contraseña.

---

## 4. Instalación de Python 3.10 en Ubuntu

1. Abre el terminal de Ubuntu.
2. Actualiza la lista de paquetes:  
   ```bash
   sudo apt update
   ```
3. Actualiza los paquetes instalados:  
   ```bash
   sudo apt upgrade
   ```
4. Instala las librerías necesarias:  
   ```bash
   sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev lzma liblzma-dev
   ```
5. Descarga e instala Python 3.10.8:  
   ```bash
   wget https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz
   tar -xzf Python-3.10.8.tgz
   cd Python-3.10.8
   ./configure --enable-optimizations
   make -j 2
   sudo make install
   ```
6. Configura un alias para Python 3.10:  
   ```bash
   alias python='/usr/local/bin/python3.10'
   source ~/.bashrc
   ```
7. Verifica la instalación:  
   ```bash
   python --version
   ```
8. Instala `pip`:  
   ```bash
   sudo apt install python3-pip
   ```
9. Instala las dependencias del proyecto desde el archivo `requirements.txt` de Moodle:  
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Creación de un Token de Hugging Face

1. Inicia sesión o crea una cuenta en [https://huggingface.co/](https://huggingface.co/).
2. Accede al menú **Access Tokens** desde el icono de tu cuenta.
3. Crea un nuevo token, asigna un nombre y otorga permisos para leer y escribir.
4. Copia el token generado y guárdalo en un lugar seguro.
5. Autentícate en WSL con:  
   ```bash
   huggingface-cli login
   ```

Pega el token cuando se solicite.

---

## 6. Comprobación Final

1. Descarga el starter kit de la práctica final desde Moodle.
2. Abre el notebook `model_finetune_and_compression.ipynb` en VS Code usando WSL.
3. Asegúrate de tener instaladas las extensiones necesarias y ejecuta la primera celda del notebook.

---

## 7. Nota Final

Los modelos y datasets de Hugging Face pueden ocupar hasta **15 GB** en disco. Elimina los modelos y datasets que no utilices con el comando:  
```bash
huggingface-cli delete-cache
```



## 8. Notas git

Forzar un pull independientemente de tener cambios sin haber hecho commit o sin haber hecho push

**Opción 1: Descarta todos los cambios locales y realiza un pull**: 

1. Descartar cambios locales
   ```bash
   git reset --hard
   ```
2. Realizar el pull
   ```bash
   git pull
   ```


**Opción 2: Forzar la actualización del repositorio local con el remoto**: 

1. Actualizar la información del repositorio remoto:
   ```bash
   git fetch --all
   ```
2. Resetear la rama local para que coincida con la remota
   ```bash
   git reset --hard origin/main
   ```

### **Resumen de las Diferencias Clave**

| **Aspecto**                         | **Opción 1**                                                | **Opción 2**                                                      |
|-------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------------|
| **Comportamiento Principal**        | Descarta cambios locales no comprometidos y realiza un pull. | Descarta todos los cambios locales y hace que la rama local coincida exactamente con la remota. |
| **Alcance de los Cambios**          | Afecta solo cambios no comprometidos.                      | Afecta cambios no comprometidos y también elimina commits locales no enviados al remoto. |
| **Actualización con el Remoto**     | Fusiona los nuevos commits del remoto con la rama local.    | Restablece la rama local para que coincida exactamente con la rama remota, eliminando cualquier commit local no presente en el remoto. |
| **Riesgo de Pérdida de Datos**      | Descarta cambios no comprometidos, pero mantiene commits locales. | Descarta todos los cambios locales, incluyendo commits no enviados, lo que puede resultar en mayor pérdida de datos. |
| **Uso Recomiendo**                  | Cuando solo deseas descartar cambios no comprometidos y mantener los commits locales. | Cuando deseas una sincronización completa con el remoto, eliminando cualquier divergencia local. |
