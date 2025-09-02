#DSD App 


## Instrucciones para Windows

1. **Clonar y crear entorno (Conda recomendado)**
   
   git clone https://github.com/LuciaSandalio/dsd_app.git
   cd dsd_app
   conda env create -f environment.yml
   conda activate dsd_app

   **Con pip y venv**:
   ```python
    py -3.9 -m venv .venv

    .\.venv\Scripts\Activate.ps1
    
    pip install --upgrade pip
    
    pip install -r requirements.txt
   ```


2. **Congiguracion local**

    ```python
    Copy-Item config\config.example.yaml config\config.yaml
    ```
    **Editar fechas y rutas en config\config.yaml**

3. **Ejecutar**

    ```python
    python src\scripts\get_dsdfile.py

    python src\scripts\event.py
    ```
   **o todo junto:**
   ```python
    python src\scripts\master_workflow.py
   ```

**Notas de compatibilidad (Windows)**

Zona horaria: se incluye tzdata, no hace falta instalar nada extra.

Rutas: el código usa pathlib.Path, por lo que podés dejar / en config.yaml y va a funcionar en Windows.

Multiproceso: si activás ProcessPoolExecutor, asegurate de que los lanzadores estén dentro de:
    ```python
    if __name__ == "__main__":
    ...
    ```

    (Esto ya está implementado en los scripts.)

Con estas instrucciones, un usuario externo en Windows puede:

* clonar el repo,

* crear un entorno (Conda o venv),

* instalar las dependencias,

* copiar/editar el config y

* correr la app sin tener que hacer ajustes manuales.


