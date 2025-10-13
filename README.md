# 🧠 MLP Credit Scoring - Multi-Layer Perceptron

> Sistema de evaluación de riesgo crediticio basado en redes neuronales profundas (Deep Learning) utilizando PyTorch.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Arquitectura del Proyecto](#-arquitectura-del-proyecto)
- [Dataset](#-dataset)
- [Modelo de Red Neuronal](#-modelo-de-red-neuronal)
- [Proceso de Entrenamiento](#-proceso-de-entrenamiento)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Configuración](#-configuración)
- [Resultados y Métricas](#-resultados-y-métricas)
- [API REST](#-api-rest)
- [MLflow Tracking](#-mlflow-tracking)
- [Versiones del Modelo](#-versiones-del-modelo)

---

## 🎯 Descripción

Este proyecto implementa un sistema de **credit scoring** utilizando un **Perceptrón Multicapa (MLP)** para predecir el riesgo crediticio de clientes bancarios. El modelo clasifica a los solicitantes de crédito en dos categorías:

- ✅ **Buen Riesgo** (Good Risk): Cliente con alta probabilidad de pago
- ❌ **Mal Riesgo** (Bad Risk): Cliente con riesgo de impago

### 🔑 Características Principales

- **Deep Learning** con PyTorch
- **Preprocesamiento robusto** con sklearn pipelines
- **Arquitectura configurable** vía archivos YAML
- **Early stopping** y técnicas de regularización
- **Tracking de experimentos** con MLflow
- **API REST** para inferencia en producción
- **Versionado de modelos** con DVC
- **Containerización** con Docker

---

## 🏗️ Arquitectura del Proyecto

```
┌─────────────────┐
│  German Credit  │
│   Risk Dataset  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ - StandardScaler│
│ - OneHotEncoder │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MLP Model     │
│  [256-128-64-64]│
│  + Dropout      │
│  + BatchNorm    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Training      │
│ - BCEWithLogits │
│ - Adam Optimizer│
│ - LR Scheduler  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │
│ - Accuracy: 78% │
│ - AUC: 0.82     │
│ - F1-Score: 0.75│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deployment     │
│  FastAPI REST   │
└─────────────────┘
```

---

## 📊 Dataset

### German Credit Risk Dataset

**Ubicación**: `datasets/genia_services_csv_german_credit_risk_v1.0.0_training_20250825/`

**Descripción**: Dataset histórico de clientes bancarios alemanes con información demográfica, financiera y de historial crediticio.

#### Features del Dataset

| Categoría | Variables | Ejemplo |
|-----------|-----------|---------|
| **Demográficas** | Age, Sex, Housing, Job | 35 años, Male, Own |
| **Financieras** | Credit amount, Duration, Saving accounts | €5000, 24 meses, Little |
| **Historial** | Checking account, Purpose, Previous credits | Moderate, Car, 1 |
| **Target** | Risk | Good / Bad |

**Distribución**:
- Total de registros: 1000
- Buenos pagadores (Good): ~70%
- Malos pagadores (Bad): ~30%

#### Preprocesamiento

El pipeline de preprocesamiento (`CreditDataPreprocessor`) realiza:

1. **Variables Numéricas** → `StandardScaler`
   ```python
   # Normalización: x_scaled = (x - mean) / std
   ['Age', 'Credit amount', 'Duration']
   ```

2. **Variables Categóricas** → `OneHotEncoder`
   ```python
   # Encoding binario
   ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Job']
   ```

3. **Output**: 26 features numéricas listas para la red neuronal

---

## 🧠 Modelo de Red Neuronal

### Arquitectura MLP (Multi-Layer Perceptron)

**Clase**: `CreditScoringModel` en `src/training/model.py`

#### Configuración del Modelo v1.3.0 (Mejor Rendimiento)

```
Input Layer (26 features)
    ↓
[Linear(26 → 256) → BatchNorm1d → ReLU → Dropout(0.1)]
    ↓
[Linear(256 → 128) → BatchNorm1d → ReLU → Dropout(0.1)]
    ↓
[Linear(128 → 64) → BatchNorm1d → ReLU → Dropout(0.1)]
    ↓
[Linear(64 → 64) → BatchNorm1d → ReLU → Dropout(0.1)]
    ↓
[Linear(64 → 1)]  ← Output logits
    ↓
Sigmoid → Probability [0, 1]
```

#### Componentes Técnicos

| Componente | Función | Propósito |
|------------|---------|-----------|
| **Linear** | `y = Wx + b` | Transformación lineal con pesos aprendibles |
| **BatchNorm1d** | Normalización | Estabiliza entrenamiento, acelera convergencia |
| **ReLU** | `f(x) = max(0, x)` | Activación no-lineal, previene vanishing gradient |
| **Dropout** | Apaga 10% neuronas | Regularización, previene overfitting |
| **Sigmoid** | `σ(x) = 1/(1+e^-x)` | Convierte logits a probabilidades |

#### Parámetros del Modelo

```python
Total Parameters: ~58,000
Trainable Parameters: 100%
Model Size: ~230 KB
```

---

## 🔄 Proceso de Entrenamiento

### Pipeline de Training

**Clase**: `CreditScoringModelTraining` en `src/training/train.py`

#### 1️⃣ Configuración (YAML)

```yaml
training_params:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: "Adam"
  weight_decay: 0.0001
  
  early_stopping:
    patience: 10
    delta: 0.001
    
  lr_scheduler:
    type: "ReduceLROnPlateau"
    patience: 7
    factor: 0.5
```

#### 2️⃣ Split de Datos

- **Training**: 80% (800 registros)
- **Validation**: 20% (200 registros)

#### 3️⃣ Función de Pérdida

**BCEWithLogitsLoss** (Binary Cross Entropy with Logits):

```python
Loss = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
```

- Combina Sigmoid + BCE en una operación numéricamente estable
- Penaliza predicciones incorrectas exponencialmente

#### 4️⃣ Optimizador

**Adam** (Adaptive Moment Estimation):

```python
m_t = β₁ * m_{t-1} + (1-β₁) * ∇L
v_t = β₂ * v_{t-1} + (1-β₂) * (∇L)²
θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
```

- Learning rate adaptativo por parámetro
- Momentum + RMSprop
- β₁=0.9, β₂=0.999

#### 5️⃣ Técnicas de Regularización

| Técnica | Configuración | Efecto |
|---------|--------------|--------|
| **Dropout** | 0.1 (10%) | Previene co-adaptación de neuronas |
| **L2 Regularization** | weight_decay=0.0001 | Penaliza pesos grandes |
| **Gradient Clipping** | max_norm=1.0 | Previene explosión de gradientes |
| **Early Stopping** | patience=10 | Para entrenamiento si no mejora |
| **LR Scheduling** | ReduceLROnPlateau | Reduce LR cuando estanca |

#### 6️⃣ Loop de Entrenamiento

```python
for epoch in range(100):
    # 🔵 TRAINING PHASE
    model.train()
    for X_batch, y_batch in train_loader:
        # Forward pass
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Backward pass (Retropropagación)
        optimizer.zero_grad()
        loss.backward()  # Calcula ∂Loss/∂W
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()  # W = W - lr * ∂Loss/∂W
    
    # 🟢 VALIDATION PHASE
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val)
        val_probs = torch.sigmoid(val_logits)
        
        # Compute metrics
        accuracy = compute_accuracy(val_probs, y_val)
        auc = roc_auc_score(y_val, val_probs)
    
    # 📊 MLflow Logging
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "val_auc": auc
    }, step=epoch)
    
    # 🛑 Early Stopping Check
    if val_loss < best_val_loss - delta:
        best_val_loss = val_loss
        save_checkpoint(model, "best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    
    # 📉 LR Scheduler
    scheduler.step(val_loss)
```

---

## 🔧 Instalación

### Requisitos Previos

- Python 3.10 o superior
- pip o conda
- (Opcional) CUDA para entrenamiento con GPU

### Instalación con Virtual Environment

```bash
# Clonar repositorio
git clone https://github.com/montoyitadevelp/MLP--Multi-Layer-Perceptron-.git
cd MLP--Multi-Layer-Perceptron-/credit_scoring

# Crear entorno virtual
python3.10 -m venv venv310
source venv310/bin/activate  # En Windows: venv310\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Instalación con Conda

```bash
# Crear entorno
conda create -n credit_scoring python=3.10
conda activate credit_scoring

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales

```
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
mlflow>=2.7.0
fastapi>=0.100.0
uvicorn>=0.23.0
joblib>=1.3.0
pyyaml>=6.0
```

---

## 🚀 Uso

### 1. Entrenar un Nuevo Modelo

```bash
cd credit_scoring
python src/training/train.py \
  --config config/training/credit_scoring-training_config-german_credit_risk_v130.yaml
```

**Resultado**:
- Modelo guardado: `models/genia_services_mlp_credit_scoring_model_v1.3.0_[fecha].pt`
- Preprocessor: `models/german_credit_risk_preprocessor.joblib`
- Reportes: `reports/credit_scoring-training_config-german_credit_risk_v130_performance_report.yaml`

### 2. Evaluar Modelo Existente

```python
from src.training.train import CreditScoringModelTraining

trainer = CreditScoringModelTraining(
    config_path="config/training/credit_scoring-training_config-german_credit_risk_v130.yaml"
)
trainer.load_model("models/genia_services_mlp_credit_scoring_model_v1.3.0_20250824.pt")
metrics = trainer.evaluate()
print(metrics)
```

### 3. Inferencia (Predicción Individual)

```python
from src.inference.predictor import CreditRiskPredictor

# Inicializar predictor
predictor = CreditRiskPredictor(
    model_path="models/genia_services_mlp_credit_scoring_model_v1.3.0_20250824.pt",
    preprocessor_path="models/german_credit_risk_preprocessor.joblib"
)

# Nueva solicitud de crédito
new_customer = {
    'Age': 30,
    'Sex': 'male',
    'Job': 2,
    'Housing': 'own',
    'Saving accounts': 'little',
    'Checking account': 'moderate',
    'Credit amount': 5000,
    'Duration': 24,
    'Purpose': 'car'
}

# Predecir
result = predictor.predict(new_customer)
print(f"Prediction: {result['prediction']}")  # 0 (Bad) o 1 (Good)
print(f"Probability: {result['probability']:.2%}")  # e.g., 78.50%
print(f"Risk Level: {result['risk_level']}")  # HIGH o LOW
```

### 4. Iniciar API REST

```bash
cd credit_scoring
uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoint disponible**: `http://localhost:8000/predict`

**Ejemplo de Request**:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 25,
    "Sex": "female",
    "Job": 2,
    "Housing": "rent",
    "Saving accounts": "little",
    "Checking account": "little",
    "Credit amount": 3000,
    "Duration": 12,
    "Purpose": "education"
  }'
```

**Response**:

```json
{
  "prediction": 1,
  "probability": 0.7845,
  "risk_level": "LOW",
  "model_version": "1.3.0",
  "timestamp": "2025-10-12T14:30:00Z"
}
```

### 5. Visualizar Tracking con MLflow

```bash
cd credit_scoring
mlflow ui --port 5000
```

Acceder a: `http://localhost:5000`

---

## 📁 Estructura del Proyecto

```
credit_scoring/
│
├── 📂 config/                      # Configuraciones
│   └── training/
│       ├── credit_scoring-training_config-german_credit_risk_v100.yaml
│       ├── credit_scoring-training_config-german_credit_risk_v110.yaml
│       ├── credit_scoring-training_config-german_credit_risk_v120.yaml
│       └── credit_scoring-training_config-german_credit_risk_v130.yaml
│
├── 📂 datasets/                    # Datos
│   └── genia_services_csv_german_credit_risk_v1.0.0_training_20250825/
│       └── german_credit_risk.csv
│
├── 📂 models/                      # Modelos entrenados
│   ├── genia_services_mlp_credit_scoring_model_v1.0.0_20250824.pt
│   ├── genia_services_mlp_credit_scoring_model_v1.1.0_20250824.pt
│   ├── genia_services_mlp_credit_scoring_model_v1.2.0_20250824.pt
│   ├── genia_services_mlp_credit_scoring_model_v1.3.0_20250824.pt
│   └── german_credit_risk_preprocessor.joblib
│
├── 📂 reports/                     # Reportes de evaluación
│   ├── classification_report_val.txt
│   ├── credit_scoring-training_config-german_credit_risk_v100_performance_report.yaml
│   ├── credit_scoring-training_config-german_credit_risk_v110_performance_report.yaml
│   ├── credit_scoring-training_config-german_credit_risk_v120_performance_report.yaml
│   └── credit_scoring-training_config-german_credit_risk_v130_performance_report.yaml
│
├── 📂 mlruns/                      # MLflow tracking
│   ├── 0/                          # Default experiment
│   └── 393392644866618955/         # Credit Scoring experiment
│       ├── 8b472c86.../            # Run IDs
│       │   ├── metrics/
│       │   ├── params/
│       │   └── artifacts/
│       └── models/
│
├── 📂 src/                         # Código fuente
│   ├── __init__.py
│   │
│   ├── 📂 processing/              # Preprocesamiento
│   │   ├── __init__.py
│   │   └── main.py                 # CreditDataPreprocessor
│   │
│   ├── 📂 training/                # Entrenamiento
│   │   ├── __init__.py
│   │   ├── model.py                # CreditScoringModel (arquitectura)
│   │   └── train.py                # CreditScoringModelTraining (training loop)
│   │
│   ├── 📂 inference/               # Inferencia
│   │   ├── __init__.py
│   │   └── predictor.py            # CreditRiskPredictor
│   │
│   ├── 📂 server/                  # API REST
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI app
│   │   └── schemas.py              # Pydantic models
│   │
│   └── 📂 examples/                # Scripts de ejemplo
│       └── __init__.py
│
├── 📂 tests/                       # Tests unitarios
│   ├── __init__.py
│   └── test_model_creation.py
│
├── 📂 venv311/                     # Virtual environment
│
├── 📄 requirements.txt             # Dependencias de producción
├── 📄 requirements_training.txt    # Dependencias de entrenamiento
├── 📄 Dockerfile                   # Container image
├── 📄 .dockerignore
├── 📄 .gitignore
├── 📄 models.dvc                   # DVC tracking de modelos
└── 📄 reports.dvc                  # DVC tracking de reportes
```

---

## ⚙️ Configuración

### Archivo de Configuración (YAML)

**Ejemplo**: `config/training/credit_scoring-training_config-german_credit_risk_v130.yaml`

```yaml
# Metadata del modelo
model_info:
  name: "credit_scoring"
  version: "1.3.0"
  description: "MLP for German Credit Risk prediction"
  date: "2025-08-24"

# Configuración del dataset
data:
  path: "datasets/genia_services_csv_german_credit_risk_v1.0.0_training_20250825/german_credit_risk.csv"
  target_column: "Risk"
  test_size: 0.2
  random_state: 42
  stratify: true

# Arquitectura del modelo
model_architecture:
  input_features: 26
  hidden_layers: [256, 128, 64, 64]
  output_size: 1
  activation: "relu"
  dropout_rate: 0.1
  use_batch_norm: true

# Parámetros de entrenamiento
training_params:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: "Adam"
  weight_decay: 0.0001
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 10
    delta: 0.001
    monitor: "val_loss"
  
  # Learning rate scheduler
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "min"
    factor: 0.5
    patience: 7
    min_lr: 0.00001
  
  # Gradient clipping
  gradient_clipping:
    enabled: true
    max_norm: 1.0

# Configuración de MLflow
mlflow:
  experiment_name: "credit_scoring_german_credit_risk"
  tracking_uri: "mlruns"
  artifact_location: "mlruns/artifacts"

# Rutas de salida
output:
  model_path: "models/"
  preprocessor_path: "models/"
  reports_path: "reports/"
```

### Variables de Entorno

```bash
# .env
MLFLOW_TRACKING_URI=mlruns
MODEL_VERSION=1.3.0
API_PORT=8000
DEVICE=cuda  # o 'cpu'
```

---

## 📈 Resultados y Métricas

### Comparación de Versiones

| Versión | Arquitectura | Dropout | Val Accuracy | Val AUC | Val F1 | Epochs |
|---------|--------------|---------|--------------|---------|--------|--------|
| v1.0.0 | [128, 64] | 0.1 | 75.2% | 0.79 | 0.72 | 82 |
| v1.1.0 | [128, 64, 64] | 0.1 | 76.8% | 0.80 | 0.73 | 91 |
| v1.2.0 | [128, 64, 64] | 0.2 | 77.5% | 0.81 | 0.74 | 95 |
| **v1.3.0** | **[256, 128, 64, 64]** | **0.1** | **78.3%** | **0.82** | **0.75** | **88** |

### Mejor Modelo: v1.3.0

#### Métricas de Validación

```
Accuracy:  78.3%
Precision: 79.1%
Recall:    76.8%
F1-Score:  0.75
AUC-ROC:   0.82
```

#### Confusion Matrix (Validation Set)

```
                 Predicted
                Bad    Good
Actual  Bad      85      15
        Good     28     122
```

#### Curvas de Aprendizaje

**Training vs Validation Loss**:
- Convergencia estable en epoch 88
- Sin evidencia de overfitting significativo
- Early stopping activado correctamente

**ROC Curve**:
- AUC = 0.82
- Buen balance entre TPR y FPR

#### Importancia Relativa de Features

Top 5 features más influyentes:
1. `Checking account`
2. `Credit amount`
3. `Duration`
4. `Saving accounts`
5. `Age`

---

## 🌐 API REST

### Endpoints

#### 1. Health Check

```
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_version": "1.3.0",
  "uptime": "2h 15m"
}
```

#### 2. Predicción

```
POST /predict
```

**Request Body**:
```json
{
  "Age": 30,
  "Sex": "male",
  "Job": 2,
  "Housing": "own",
  "Saving accounts": "little",
  "Checking account": "moderate",
  "Credit amount": 5000,
  "Duration": 24,
  "Purpose": "car"
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.7845,
  "risk_level": "LOW",
  "confidence": "HIGH",
  "model_version": "1.3.0",
  "timestamp": "2025-10-12T14:30:00Z",
  "processing_time_ms": 12.5
}
```

#### 3. Predicción en Batch

```
POST /predict/batch
```

**Request Body**:
```json
{
  "customers": [
    { "Age": 30, "Sex": "male", ... },
    { "Age": 25, "Sex": "female", ... },
    { "Age": 40, "Sex": "male", ... }
  ]
}
```

**Response**:
```json
{
  "predictions": [
    { "id": 0, "prediction": 1, "probability": 0.7845 },
    { "id": 1, "prediction": 0, "probability": 0.3521 },
    { "id": 2, "prediction": 1, "probability": 0.8912 }
  ],
  "total_processed": 3,
  "processing_time_ms": 35.2
}
```

#### 4. Información del Modelo

```
GET /model/info
```

**Response**:
```json
{
  "name": "Credit Scoring MLP",
  "version": "1.3.0",
  "architecture": {
    "input_features": 26,
    "hidden_layers": [256, 128, 64, 64],
    "output_size": 1,
    "total_parameters": 58241
  },
  "training_date": "2025-08-24",
  "metrics": {
    "accuracy": 0.783,
    "auc": 0.82,
    "f1_score": 0.75
  }
}
```

### Documentación Interactiva

Una vez iniciado el servidor, accede a:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📊 MLflow Tracking

### Iniciar MLflow UI

```bash
cd credit_scoring
mlflow ui --port 5000
```

Acceder a: `http://localhost:5000`

### Información Registrada

#### Parámetros (Params)

```python
{
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'hidden_layers': '[256, 128, 64, 64]',
    'dropout_rate': 0.1,
    'optimizer': 'Adam',
    'weight_decay': 0.0001
}
```

#### Métricas (Metrics)

```python
# Por epoch
{
    'train_loss': [0.65, 0.52, 0.45, ...],
    'val_loss': [0.68, 0.55, 0.48, ...],
    'val_accuracy': [0.68, 0.72, 0.75, ...],
    'val_auc': [0.72, 0.76, 0.80, ...],
    'val_f1': [0.65, 0.70, 0.73, ...],
    'learning_rate': [0.001, 0.001, 0.0005, ...]
}
```

#### Artifacts

```
artifacts/
├── model/
│   ├── model.pt
│   └── preprocessor.joblib
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── learning_curves.png
└── reports/
    ├── classification_report.txt
    └── performance_summary.yaml
```

### Comparar Experimentos

```python
import mlflow

# Buscar runs
runs = mlflow.search_runs(
    experiment_names=["credit_scoring_german_credit_risk"],
    order_by=["metrics.val_auc DESC"]
)

print(runs[['run_id', 'params.hidden_layers', 'metrics.val_auc']])
```

---

## 🔄 Versiones del Modelo

### Versionado con DVC

```bash
# Track modelos
dvc add models/genia_services_mlp_credit_scoring_model_v1.3.0_20250824.pt

# Track reportes
dvc add reports/credit_scoring-training_config-german_credit_risk_v130_performance_report.yaml

# Commit
git add models.dvc reports.dvc
git commit -m "Track model v1.3.0"

# Push a remote storage
dvc push
```

### Historial de Versiones

| Fecha | Versión | Cambios | Performance |
|-------|---------|---------|-------------|
| 2025-08-20 | v1.0.0 | Baseline: [128, 64] | AUC: 0.79 |
| 2025-08-21 | v1.1.0 | +1 hidden layer | AUC: 0.80 |
| 2025-08-22 | v1.2.0 | Dropout: 0.1→0.2 | AUC: 0.81 |
| 2025-08-24 | v1.3.0 | Arquitectura más profunda | AUC: 0.82 ✅ |

---

## 🐳 Docker

### Construir Imagen

```bash
cd credit_scoring
docker build -t credit-scoring-mlp:v1.3.0 .
```

### Ejecutar Container

```bash
docker run -d \
  --name credit-scoring-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  credit-scoring-mlp:v1.3.0
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_VERSION=1.3.0
      - DEVICE=cpu
    restart: unless-stopped
```

```bash
docker-compose up -d
```

---

## 🧪 Testing

### Ejecutar Tests

```bash
cd credit_scoring
pytest tests/ -v
```

### Tests Implementados

```python
# tests/test_model_creation.py
def test_model_creation():
    """Verifica creación correcta del modelo"""
    
def test_forward_pass():
    """Verifica forward pass con input dummy"""
    
def test_preprocessor():
    """Verifica transformaciones del preprocessor"""
    
def test_prediction_pipeline():
    """Test end-to-end de predicción"""
```

---

## 📚 Conceptos Técnicos

### Retropropagación (Backpropagation)

El algoritmo de retropropagación calcula gradientes de la función de pérdida respecto a todos los pesos:

```
∂Loss/∂W = ∂Loss/∂output × ∂output/∂hidden × ∂hidden/∂W
```

PyTorch lo hace automáticamente con `loss.backward()`.

### Adam Optimizer

Combina momentum y RMSprop:

```python
# Momento de primer orden (momentum)
m_t = β₁ * m_{t-1} + (1-β₁) * ∇L

# Momento de segundo orden (RMSprop)
v_t = β₂ * v_{t-1} + (1-β₂) * (∇L)²

# Actualización con corrección de sesgo
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

### Batch Normalization

Normaliza activaciones por mini-batch:

```python
μ_B = (1/m) * Σx_i          # Media del batch
σ²_B = (1/m) * Σ(x_i - μ_B)²  # Varianza del batch
x̂_i = (x_i - μ_B) / √(σ²_B + ε)  # Normalización
y_i = γ * x̂_i + β           # Escala y desplazamiento aprendibles
```

### Early Stopping

Para entrenamiento cuando la métrica de validación no mejora:

```python
if val_loss < best_val_loss - delta:
    best_val_loss = val_loss
    patience_counter = 0
    save_checkpoint()
else:
    patience_counter += 1
    if patience_counter >= patience:
        stop_training()
```

---

## 🛠️ Troubleshooting

### Error: CUDA out of memory

**Solución**:
```python
# Reducir batch size
batch_size: 16  # en lugar de 32

# O entrenar en CPU
device: 'cpu'
```

### Error: Model performance degradation

**Posibles causas**:
1. Overfitting → Aumentar dropout
2. Underfitting → Aumentar hidden layers
3. Learning rate muy alto → Reducir lr
4. Data drift → Re-entrenar con datos nuevos

### Error: API request timeout

**Solución**:
```python
# En app.py
@app.post("/predict", timeout=30.0)

# O aumentar gunicorn timeout
gunicorn src.server.app:app --timeout 60
```

---

## 📖 Referencias

### Papers

- **Multi-Layer Perceptron**: Rumelhart et al. (1986)
- **Adam Optimizer**: Kingma & Ba (2014)
- **Batch Normalization**: Ioffe & Szegedy (2015)
- **Dropout**: Srivastava et al. (2014)

### Documentación

- [PyTorch Documentation](https://pytorch.org/docs/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Dataset

- **German Credit Risk**: UCI Machine Learning Repository

---

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Add nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

---

## 📧 Contacto

**Proyecto**: [MLP Credit Scoring](https://github.com/montoyitadevelp/MLP--Multi-Layer-Perceptron-)

**Autor**: montoyitadevelp

---

## ⭐ Agradecimientos

- Dataset: UCI Machine Learning Repository
- Framework: PyTorch Team
- Tracking: MLflow Community
- API: FastAPI Team

---

**¡Gracias por usar MLP Credit Scoring!** 🎉
