import tkinter as tk
from tkinter import scrolledtext, Menu
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cargar el modelo
model_path = "./mistral_user_story_generator"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

def generar_respuesta(conversacion):
    prompt = f"[INST] {conversacion} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True).to(model.device)
    
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
    
    print(f"Token IDs: {inputs['input_ids']}")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=512,
            temperature=0.9,
            top_p=0.8,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Respuesta generada: {response}")
    
    if len(response.split()) < 10:
        print("¡Parece que la respuesta no es válida!")
        return "Lo siento, no pude generar una respuesta coherente."
    
    return response

def conversacion_interactiva():
    conversacion = []
    funcionalidad = input("¿Qué funcionalidad necesitas? ")
    conversacion.append(f"User: {funcionalidad}")
    print("System: ¿Cuál es tu rol?")
    rol = input("User: ")
    conversacion.extend(["System: ¿Cuál es tu rol?", f"User: {rol}"])
    print("System: ¿Qué características específicas necesitas?")
    caracteristicas = input("User: ")
    conversacion.extend(["System: ¿Qué características específicas necesitas?", f"User: {caracteristicas}"])
    print("System: ¿Algún requisito adicional?")
    requisitos = input("User: ")
    conversacion.extend(["System: ¿Algún requisito adicional?", f"User: {requisitos}"])
    
    conversacion_completa = "\n".join(conversacion)
    prompt_final = f"{conversacion_completa}\nSystem: Genera una historia de usuario en formato Como/Quiero/Para basada en esta conversación."
    
    return generar_respuesta(prompt_final)

# Configuración de la ventana en Tkinter
ventana = tk.Tk()
ventana.title("Chatbot - Generador de Historias")
ventana.geometry("500x600")
ventana.configure(bg="#f4f4f4")

# Menú principal
menu_bar = Menu(ventana)
ventana.config(menu=menu_bar)

file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Nueva conversación", command=lambda: chat_log.config(state=tk.NORMAL) or chat_log.delete("1.0", tk.END) or chat_log.config(state=tk.DISABLED))
file_menu.add_separator()
file_menu.add_command(label="Salir", command=ventana.quit)
menu_bar.add_cascade(label="Archivo", menu=file_menu)

# Contenedor del chat
frame_chat = tk.Frame(ventana, bg="white", padx=10, pady=10)
frame_chat.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

chat_log = scrolledtext.ScrolledText(frame_chat, wrap=tk.WORD, width=60, height=25, state=tk.DISABLED, font=("Arial", 12))
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

chat_log.tag_config("user", foreground="#007BFF", font=("Arial", 12, "bold"))
chat_log.tag_config("bot", foreground="#333", font=("Arial", 12))

def enviar_mensaje():
    user_input = entrada_usuario.get()
    if user_input.strip():
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"Usuario: {user_input}\n", "user")
        chat_log.config(state=tk.DISABLED)
        entrada_usuario.delete(0, tk.END)
        
        respuesta = generar_respuesta(user_input)
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"Asistente: {respuesta}\n\n", "bot")
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)

frame_input = tk.Frame(ventana, bg="#f4f4f4")
frame_input.pack(pady=10, padx=10, fill=tk.X)

entrada_usuario = tk.Entry(frame_input, width=40, font=("Arial", 12))
entrada_usuario.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

boton_enviar = tk.Button(frame_input, text="Send", command=enviar_mensaje, bg="#007BFF", fg="white", font=("Arial", 12, "bold"))
boton_enviar.pack(side=tk.RIGHT, padx=5, pady=5)

ventana.mainloop()