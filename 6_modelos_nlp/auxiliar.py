import torch.nn as nn
import torch
import matplotlib.pyplot as plt

# Funciones para entrenamiento y evaluación
def binary_accuracy_logits(logits, y, threshold = 0.5):
    probs = torch.sigmoid(logits)
    return binary_accuracy(probs, y, threshold)

def binary_accuracy(probs, y, threshold = 0.5):
    preds = (probs >= threshold)
    correct = (preds == y)
    return correct.sum().item() / correct.shape[0]

def evaluate(model, loader, loss_fn, device='cuda'):
    model.eval() 
    loss_acum, accu_acum = 0, 0
    with torch.no_grad(): 
        for x, y in loader:
            # Enviamos los datos la GPU. Y aprovechamos para convertir
            # la etiqueta a float (por defecto, ImageFolder devuelve int64 para las labels)
            x, y = x.to(device), y.float().to(device) # <-- Enviamos datos a la GPU
            preds = model(x).squeeze()
            loss = loss_fn(preds, y)
            loss_acum += loss.item()
            accu_acum += binary_accuracy(preds,y)
    avg_val_loss = loss_acum / len(loader)        
    avg_val_accu = accu_acum / len(loader)   
    return avg_val_loss, avg_val_accu

def train_model(model, epochs, train_loader, val_loader, loss_fn, optimizer, device='cuda'):  
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} 
    # --- Bucle principal ---
    for epoch in range(epochs):
        # --- Fase de Entrenamiento ---
        model.train() 
        loss_acum, accu_acum = 0, 0            
        for x, y in train_loader:
            # Enviamos los datos la GPU.
            x, y = x.to(device), y.to(device) # <-- Enviamos datos a la GPU
            optimizer.zero_grad()
            preds = model(x)
            preds = preds.squeeze()  
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            # calculamos métricas
            loss_acum += loss.item()
            accu_acum += binary_accuracy(preds,y)                          
    
        avg_train_loss = loss_acum / len(train_loader)        
        avg_train_accu = accu_acum / len(train_loader)

        # --- Fase de Validación ---
        avg_val_loss, avg_val_accu = evaluate(model,val_loader,loss_fn)        
        
        print(f"Epoch [{epoch:02d}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Train Accuracy: {avg_train_accu:.4f}, Validation Accuracy: {avg_val_accu:.4f}")

        history['train_loss'].append(avg_train_loss), history['val_loss'].append(avg_val_loss), history['train_acc'].append(avg_train_accu), history['val_acc'].append(avg_val_accu)
    print("\n--- Entrenamiento finalizado ---")
    return history

def plot_history(history):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Pérdida Entrenamiento y Validación')

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.title('Accuracy Entrenamiento y Validación')
    plt.show()
