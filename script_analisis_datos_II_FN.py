import pandas as pd
import hashlib
import os
import warnings
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)

#GESTIÓN DE SEGURIDAD (AES-256 CBC)
class SecurityManager:
    def __init__(self, key):
        self.key = key 

    def encrypt_data(self, plaintext):
        plaintext = str(plaintext).strip()
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        padded_data = plaintext + (" " * (16 - len(plaintext) % 16))
        ciphertext = encryptor.update(padded_data.encode()) + encryptor.finalize()
        
        return (iv + ciphertext).hex()

    def decrypt_data(self, encrypted_hex):
        encrypted_data = bytes.fromhex(encrypted_hex)
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return padded_plaintext.decode('utf-8').strip()

#SIMULACIÓN DE BLOCKCHAIN (SHA-256)
def generate_block_hash(data, previous_hash):
    clean_data = str(data).strip()
    clean_prev_hash = str(previous_hash).strip()
    block_content = f"{clean_data}{clean_prev_hash}"
    return hashlib.sha256(block_content.encode()).hexdigest()

def verify_blockchain(df):
    prev_h = "0" * 64
    for index, row in df.iterrows():
        # El hash protege: Identidad, Horas, Entregas y Nota
        datos_bloque = f"{row['Identidad_Cifrada']}{row['Horas_Conexion']}{row['Porcentaje_Entregas']}{row['Nota_Media']}"
        expected_hash = generate_block_hash(datos_bloque, prev_h)
        if expected_hash != row['Hash_Blockchain']:
            return False 
        prev_h = expected_hash
    return True 

#LEARNING ANALYTICS (Random Forest)
def train_analytics_model():
    X = [
        [40, 90, 8.5], [35, 85, 7.2], [10, 20, 3.1], [5, 10, 2.5], 
        [50, 100, 9.8], [12, 35, 4.5], [25, 60, 6.0], [5, 5, 1.5]
    ]
    y = [0, 0, 1, 1, 0, 1, 0, 1]
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
    clf.fit(X, y)
    return clf

#FLUJO PRINCIPAL
def main():
    master_key = os.urandom(32)
    sec = SecurityManager(master_key)
    
    print("\n[INFO] Cargando datos desde archivo CSV...")
    try:
        df = pd.read_csv('datos_estudiantes.csv')
    except FileNotFoundError:
        print("[ERROR] No se encuentra el archivo 'datos_estudiantes.csv'. Asegúrate de que está en la misma carpeta.")
        return

    df['Identidad_Cifrada'] = df['Nombre_Estudiante'].apply(sec.encrypt_data)

    # Generación de Blockchain incluyendo notas
    hashes = []
    prev_h = "0" * 64 
    for index, row in df.iterrows():
        datos_bloque = f"{row['Identidad_Cifrada']}{row['Horas_Conexion']}{row['Porcentaje_Entregas']}{row['Nota_Media']}"
        current_h = generate_block_hash(datos_bloque, prev_h)
        hashes.append(current_h)
        prev_h = current_h
    df['Hash_Blockchain'] = hashes

    model = train_analytics_model()
    df['Prediccion_Riesgo'] = model.predict(df[['Horas_Conexion', 'Porcentaje_Entregas', 'Nota_Media']])
    df['Estado'] = df['Prediccion_Riesgo'].map({0: 'Seguro', 1: 'Alerta: Riesgo'})

    #SALIDA DE RESULTADOS
    print("\n" + "="*95)
    print("HERRAMIENTA DE LEARNING ANALYTICS: MÓDULO DE SEGURIDAD E INTEGRIDAD")
    print("="*95)
    
    output_display = df[['Identidad_Cifrada', 'Estado', 'Hash_Blockchain']]
    print(output_display.to_string(index=False))
    
    #AUDITORÍA DE SEGURIDAD (Simulación interactiva de ataque)
    print("\n" + "-"*95)
    print("AUDITORÍA DE SEGURIDAD: PRUEBA DE INMUTABILIDAD DE DATOS ACADÉMICOS")
    print("-" * 95)
    
    #HACKEO
    ataque = input("¿Desea simular una inyección maliciosa (alterar una nota en memoria) para probar el Blockchain? (S/N): ").strip().lower()
    
    if ataque == 's':
        nota_original = df.at[0, 'Nota_Media']
        df.at[0, 'Nota_Media'] = 9.9
        print(f"\n[!] SIMULACIÓN: Un atacante ha modificado la Nota Media del Alumno 1 (de {nota_original} a 9.9)...")
    else:
        print("\n[INFO] Procediendo con la verificación de los datos originales...")
    
    #EJECUCIÓN DE PRUEBA DE INTEGRIDAD REAL
    if verify_blockchain(df):
        print("\n[INFO] Resultado de la Auditoría: Cadena de bloques comprobada matemáticamente y VÁLIDA.")
    else:
        print("\n[ALERTA CRÍTICA] Resultado de la Auditoría: LA CADENA DE BLOQUES HA SIDO ALTERADA. HASH INVÁLIDO.")
        
    print("\n[INFO] Protección de datos: AES-256-CBC activo (Encriptación y Desencriptación funcionales).")
    print("[INFO] Análisis: Modelo Random Forest aplicado correctamente.")

    #RECOMENDACIONES
    print("\n[RECOMENDACIONES AUTOMÁTICAS PERSONALIZADAS]")
    for i, row in df.iterrows():
        if row['Estado'] == 'Alerta: Riesgo':
            motivos = []
            if row['Nota_Media'] < 5.0: motivos.append("suspensos")
            if row['Porcentaje_Entregas'] < 50: motivos.append("falta de entregas")
            if row['Horas_Conexion'] < 20: motivos.append("baja participación")
            
            razon = " y ".join(motivos) if motivos else "rendimiento general bajo"
            print(f"- Alum. {i+1} (Protegido): RIESGO. Convocar tutoría por {razon}. (Nota original/registrada: {row['Nota_Media']})")
        else:
            print(f"- Alum. {i+1} (Protegido): SEGURO. Rendimiento adecuado. Felicitar por su constancia. (Nota original/registrada: {row['Nota_Media']})")

    #VISUALIZACIÓN GRÁFICA
    print("\n[INFO] Generando visualización de probabilidades de riesgo en ventana externa...")
    probabilidades = model.predict_proba(df[['Horas_Conexion', 'Porcentaje_Entregas', 'Nota_Media']])[:, 1]
    df['Prob_Abandono_%'] = probabilidades * 100
    
    plt.figure(figsize=(10, 6))
    etiquetas = [f"Alum. {i+1}" for i in range(len(df))]
    colores = ['#F44336' if p >= 50 else '#4CAF50' for p in df['Prob_Abandono_%']]
    barras = plt.bar(etiquetas, df['Prob_Abandono_%'], color=colores)
    
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Umbral de Alerta (50%)')
    plt.title('Probabilidad de Riesgo de Abandono por Estudiante (Análisis Individual)', fontsize=14)
    plt.xlabel('Estudiantes (Identidad Protegida)', fontsize=12)
    plt.ylabel('Probabilidad de Abandono (%)', fontsize=12)
    plt.ylim(0, 110)
    plt.legend()
    
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., altura + 2,
                 f'{int(altura)}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    plt.show(block=False) 
    plt.pause(0.1) 

    #PANEL INTERACTIVO DE DESENCRIPTACIÓN
    print("\n" + "="*95)
    print("PANEL DE ADMINISTRADOR: REVELACIÓN DE IDENTIDADES PARA TUTORÍAS")
    print("="*95)
    print("(*) Consulte la ventana de la gráfica para decidir qué alumnos revisar.")
    
    while True:
        respuesta = input("\n¿Desea desencriptar la identidad de algún alumno en riesgo? \n(Indique el número, ej: 3 o 3, 4, o escriba N para salir): ").strip().lower()
        respuesta = respuesta.replace("'", "").replace('"', "")
        
        if respuesta == 'n' or respuesta == 'no' or respuesta == '':
            print("\n[INFO] Cerrando panel de administrador...")
            break
            
        respuesta_limpia = respuesta.replace(' y ', ',')
        numeros_str = respuesta_limpia.split(',')
        
        try:
            for num_str in numeros_str:
                if num_str.strip() == "": 
                    continue
                num_alumno = int(num_str.strip())
                if 1 <= num_alumno <= len(df):
                    indice = num_alumno - 1
                    id_cifrado = df['Identidad_Cifrada'].iloc[indice]
                    nombre_original = sec.decrypt_data(id_cifrado)
                    estado_alum = df['Estado'].iloc[indice]
                    print(f" -> IDENTIDAD REVELADA [Alum. {num_alumno}]: {nombre_original} | Estado: {estado_alum}")
                else:
                    print(f" [!] El alumno {num_alumno} no existe en la base de datos.")
        except ValueError:
            print(" [!] Entrada no válida. Por favor, use solo números separados por comas (ej: 1, 2) o N para salir.")

    #EVITAR CIERRE
    print("\n[INFO] Ejecución finalizada con éxito.")
    input("Presione ENTER para cerrar la ventana y salir del programa...")

if __name__ == "__main__":
    main()
