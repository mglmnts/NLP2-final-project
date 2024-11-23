# Guía de Instalación para el Proyecto Final de NLP

**Autor:** Nombre del Autor  
**Fecha:** 15 de noviembre de 2024  

---

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
