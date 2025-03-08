from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Cargando el modelo... (puede tomar unos segundos)")
model_path = "./mistral_user_story_generator"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
print("¡Modelo cargado!")

def generar_respuesta(conversacion):
    prompt = f"[INST] {conversacion} [/INST]"
    
    # Corregimos la tokenización
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True,
        padding=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    return response

def generar_respuesta_normal(conversacion):
    print(f"Procesando conversación normal: {conversacion}")
    inputs = tokenizer(
        conversacion, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True,
        padding=True
    ).to(model.device)
    
    # Verificar cómo está siendo tokenizada la entrada
    print(f"Token IDs: {inputs['input_ids']}")
    
    with torch.no_grad():
        outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=512,
        temperature=0.9,  # Incrementar ligeramente la temperatura
        top_p=0.8,  # Reducir el valor de top_p
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Respuesta generada: {response}")
    
    # Verifica si la respuesta parece ser válida
    if len(response.split()) < 10:  # Si la respuesta es muy corta, posiblemente sea un error
        print("¡Parece que la respuesta no es válida!")
        return "Lo siento, no pude generar una respuesta coherente."
    
    return response



def conversacion_interactiva():
    conversacion = []
    
    # Funcionalidad
    funcionalidad = input("¿Qué funcionalidad necesitas? ")
    conversacion.append(f"User: {funcionalidad}")
    
    # Rol
    print("System: ¿Cuál es tu rol?")
    rol = input("User: ")
    conversacion.extend(["System: ¿Cuál es tu rol?", f"User: {rol}"])
    
    # Características
    print("System: ¿Qué características específicas necesitas?")
    caracteristicas = input("User: ")
    conversacion.extend(["System: ¿Qué características específicas necesitas?", f"User: {caracteristicas}"])
    
    # Requisitos adicionales
    print("System: ¿Algún requisito adicional?")
    requisitos = input("User: ")
    conversacion.extend(["System: ¿Algún requisito adicional?", f"User: {requisitos}"])
    
    # Unir la conversación
    conversacion_completa = "\n".join(conversacion)
    prompt_final = f"{conversacion_completa}\nSystem: Genera una historia de usuario en formato Como/Quiero/Para basada en esta conversación."
    
    return generar_respuesta(prompt_final)

print("\n=== Generador de Historias de Usuario ===")
print("Escribe 'salir' para terminar")
print("----------------------------------------")

while True:
    print("\n1. Iniciar nueva historia de usuario")
    print("2. Salir")

       if opcion == "2" or opcion.lower() == 'salir':
        print("\n¡Hasta luego!")
        break
    
    if opcion == "1":
        try:
            historia = conversacion_interactiva()
            print("\nHistoria de usuario generada:")
            print("------------------------------")
            print(historia)
            print("------------------------------")
        except Exception as e:
            print(f"Ocurrió un error: {str(e)}")
            print("Detalles completos del error:", e.__class__.__name__)
    else:
        print("Opción no válida. Por favor, selecciona 1 o 2.")


print("\n=== Generador de Historias de Usuario ===")
print("Escribe 'salir' para terminar")
print("----------------------------------------")

