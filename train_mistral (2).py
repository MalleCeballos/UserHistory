from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset, load_dataset  # Esta es la línea modificada
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
print("\n=== Verificación del Sistema ===")
print(f"PyTorch versión: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA versión: {torch.version.cuda}")
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Memoria GPU disponible: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB usado")
print(f"BNB_CUDA_VERSION: {os.getenv('BNB_CUDA_VERSION')}")
print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")
print("==============================\n")
# Configuración inicial
HUGGING_FACE_TOKEN = ""  # Reemplaza esto con tu token
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./results_mistral"

# Login en Hugging Face
print("Autenticando en Hugging Face...")
login(HUGGING_FACE_TOKEN)


# Configuración de cuantización de 4-bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Cargar dataset

print("Cargando dataset...")
with open('dataset_historias_usuario.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Crear el dataset de Hugging Face
raw_dataset = Dataset.from_dict({
    'conversation': [item['conversation'] for item in data],
    'user_story': [item['user_story'] for item in data]
})

print(f"\nTamaño del dataset creado: {len(raw_dataset)}")
print("\nEjemplo del primer registro:")
print(f"Conversación: {raw_dataset[0]['conversation']}")
print(f"Historia de usuario: {raw_dataset[0]['user_story']}")


print(f"\nTamaño del dataset creado: {len(raw_dataset)}")


# Cargar tokenizer y modelo
print("Cargando tokenizer y modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Agregar esta línea
tokenizer.padding_side = "right"  # Agregar esta línea

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.pad_token_id  
# Agregar después de cargar el modelo
print(f"Memoria GPU usada después de cargar modelo: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# Configuración LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Preparar modelo para entrenamiento
print("Preparando modelo para entrenamiento...")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

def preprocess_function(examples):
    # Formato de entrada: conversación + instrucción para generar historia
    inputs = [
        f"<s>[INST] Basado en esta conversación, genera una historia de usuario:\n{conv} [/INST]"
        for conv in examples["conversation"]
    ]
    
    # Formato de salida: historia de usuario
    targets = [f"{story}</s>" for story in examples["user_story"]]
    
    # Concatenar entrada y salida
    concatenated = [inp + target for inp, target in zip(inputs, targets)]
    
    model_inputs = tokenizer(
        concatenated,
        max_length=512,  # Aumentado para manejar conversaciones más largas
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    return model_inputs


# Preprocesar dataset
print("Preprocesando dataset...")
tokenized_dataset = raw_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_dataset.column_names
)
tokenized_dataset = tokenized_dataset.shuffle(seed=42)

# División del dataset
train_size = int(0.8 * len(tokenized_dataset))
train_data = tokenized_dataset.select(range(train_size))
eval_data = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

# Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=25,  # Reducido para evaluar más frecuentemente
    learning_rate=1e-4,  # Reducido para un aprendizaje más estable
    per_device_train_batch_size=2,  # Reducido debido al mayor tamaño de las secuencias
    per_device_eval_batch_size=2,
    num_train_epochs=5,  # Aumentado para mejor aprendizaje
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    warmup_steps=100,  # Aumentado para mejor estabilidad
    gradient_accumulation_steps=8,  # Aumentado para compensar el batch size pequeño
    fp16=True,
    gradient_checkpointing=True,
)

# Configurar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

# Entrenamiento
print("Iniciando entrenamiento...")
trainer.train()

# Guardar modelo
print("Guardando modelo...")
model_path = "./mistral_user_story_generator"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Modelo guardado en {model_path}")

# Función de prueba
def generate_user_story(conversation_text, model, tokenizer):
    prompt = f"<s>[INST] Basado en esta conversación, genera una historia de usuario:\n{conversation_text} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Pruebas finales
print("\nRealizando pruebas finales...")
test_inputs = [
    "Sistema de búsqueda avanzada",
    "Panel de administración",
    "Exportación de reportes"
]

for test_input in test_inputs:
    try:
        generated_story = generate_user_story(test_input, model, tokenizer)
        print(f"\nInput: {test_input}")
        print(f"Generated: {generated_story}")
    except Exception as e:
        print(f"Error generando historia para '{test_input}': {str(e)}")

print("\n¡Proceso completado!")