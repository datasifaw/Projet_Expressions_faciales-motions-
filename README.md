# 😊 Reconnaissance des émotions faciales — FER2013

Projet de **Deep Learning et Computer Vision** consacré à la classification automatique des émotions faciales à partir d'images.

Le projet utilise **PyTorch** et compare deux approches :

1. un **CNN construit from scratch** ;
2. un modèle **ResNet18 pré-entraîné** utilisant le Transfer Learning.

Une phase supplémentaire d'optimisation du ResNet18 avec un scheduler de Learning Rate permet d'atteindre une **accuracy finale de 62,69 %** sur le jeu de test FER2013.

---

## 🎯 Objectif du projet

L'objectif est de développer un modèle capable d'identifier automatiquement l'émotion exprimée sur un visage parmi **7 catégories** : Angry 😠   Disgust 🤢  Fear 😨  Happy 😊    Neutral 😐  Sad 😢    Surprise 😲 

Le projet permet également de comparer l'efficacité d'un réseau convolutif simple avec celle d'un modèle pré-entraîné plus profond.

---

## 📊 Dataset

Le projet utilise le dataset **FER2013** pour la reconnaissance des émotions faciales.

Les images sont chargées avec :

```python
from torchvision.datasets import ImageFolder
```

Le jeu de test utilisé dans le notebook contient **7 178 images** réparties entre les sept émotions.

La distribution du jeu de test observée lors de l'évaluation est :

| Émotion  | Nombre d'images |
| -------- | --------------: |
| Angry    |             958 |
| Disgust  |             111 |
| Fear     |           1 024 |
| Happy    |           1 774 |
| Neutral  |           1 233 |
| Sad      |           1 247 |
| Surprise |             831 |

Cette distribution montre notamment que certaines classes sont beaucoup moins représentées que d'autres.

---

# 🧹 Prétraitement des données

Les images sont converties en **niveaux de gris** puis redimensionnées en :

```text
48 × 48 pixels
```

Une normalisation est ensuite appliquée :

```python
transforms.Normalize((0.5,), (0.5,))
```

---

## 🔄 Data Augmentation

Afin d'améliorer la capacité de généralisation du modèle, plusieurs transformations sont appliquées aux données d'entraînement :

```python
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

Les transformations comprennent :

* conversion en niveaux de gris ;
* redimensionnement en 48 × 48 ;
* retournement horizontal aléatoire ;
* rotation jusqu'à 15° ;
* translation aléatoire ;
* modification de la luminosité ;
* modification du contraste ;
* normalisation.

Pour le jeu de test, seules les transformations nécessaires à l'inférence sont utilisées.

---

# 🧠 Partie I — CNN from Scratch

La première expérience consiste à construire un réseau de neurones convolutif directement avec PyTorch.

## Architecture

```text
Image 48 × 48 × 1
        │
        ▼
Conv2D
1 → 32 filtres
3 × 3
        │
        ▼
ReLU
        │
        ▼
MaxPooling 2 × 2
        │
        ▼
Conv2D
32 → 64 filtres
3 × 3
        │
        ▼
ReLU
        │
        ▼
MaxPooling 2 × 2
        │
        ▼
Flatten
        │
        ▼
Dense 128
        │
        ▼
ReLU
        │
        ▼
Dropout 0.5
        │
        ▼
Dense 7
        │
        ▼
7 émotions
```

Le réseau est défini avec :

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )
```

---

## ⚙️ Configuration

La fonction de perte utilisée est :

```python
criterion = nn.CrossEntropyLoss()
```

L'optimiseur est **Adam** :

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Le modèle est entraîné pendant :

```text
35 epochs
```

avec des batches de :

```text
128 images
```

---

## 📈 Résultat du CNN

Après entraînement :

```text
Accuracy test : 54,64 %
```

Cette première expérience sert de baseline pour comparer les performances avec une architecture plus avancée.

---

# 🚀 Partie II — Transfer Learning avec ResNet18

La deuxième approche utilise **ResNet18**, un réseau convolutif pré-entraîné.

```python
resnet18 = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)
```

L'objectif est de profiter des caractéristiques visuelles déjà apprises par le réseau puis de l'adapter au problème de reconnaissance des émotions.

---

## 🔧 Adaptation de ResNet18

Le modèle ResNet18 original utilise des images RGB à trois canaux.

Comme FER2013 est utilisé ici en niveaux de gris, la première couche est remplacée par :

```python
resnet18.conv1 = nn.Conv2d(
    1,
    64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)
```

La dernière couche est également adaptée pour produire **7 classes** :

```python
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 7)
```

---

## ⚙️ Fine-tuning

La fonction de perte reste :

```python
criterion = nn.CrossEntropyLoss()
```

L'optimiseur utilisé est Adam avec un Learning Rate plus faible :

```python
optimizer = optim.Adam(
    resnet18.parameters(),
    lr=0.0001
)
```

Le premier fine-tuning est réalisé pendant :

```text
30 epochs
```

---

## 📈 Résultat avec ResNet18

Après cette phase :

```text
Accuracy test : 60,53 %
```

Le Transfer Learning améliore donc les performances par rapport au CNN construit from scratch.

| Modèle           | Accuracy |
| ---------------- | -------: |
| CNN from Scratch |  54,64 % |
| ResNet18         |  60,53 % |

---

# ⚡ Partie III — Optimisation des hyperparamètres

Une troisième phase poursuit l'entraînement du ResNet18 avec une stratégie de réduction progressive du Learning Rate.

Le scheduler utilisé est :

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(
    optimizer,
    step_size=20,
    gamma=0.5
)
```

Cela signifie que le Learning Rate est divisé par deux toutes les **20 epochs**.

```python
scheduler.step()
```

Cette phase supplémentaire est exécutée pendant :

```text
100 epochs
```

La Loss d'entraînement diminue fortement au cours de cette phase, jusqu'à environ :

```text
0.033
```

---

# 🏆 Résultat final

Après optimisation :

```text
Accuracy finale : 62,69 %
```

### Comparaison des expériences

| Expérience | Modèle                     | Accuracy test |
| ---------- | -------------------------- | ------------: |
| 1          | CNN from Scratch           |       54,64 % |
| 2          | ResNet18 Transfer Learning |       60,53 % |
| 3          | ResNet18 + StepLR          |   **62,69 %** |

Le passage du CNN simple au ResNet18 optimisé apporte donc un gain d'environ :

```text
+8 points d'accuracy
```

---

# 📊 Évaluation détaillée

L'évaluation finale utilise plusieurs métriques :

* Accuracy
* Precision
* Recall
* F1-score
* Support
* Matrice de confusion

Les résultats obtenus sont :

| Émotion  | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| Angry    |      0.55 |   0.53 |     0.54 |
| Disgust  |      0.80 |   0.65 |     0.72 |
| Fear     |      0.52 |   0.47 |     0.49 |
| Happy    |      0.78 |   0.82 | **0.80** |
| Neutral  |      0.54 |   0.60 |     0.57 |
| Sad      |      0.51 |   0.48 |     0.50 |
| Surprise |      0.78 |   0.77 | **0.77** |

### Résultat global

```text
Accuracy     : 62,69 %
Macro F1     : 0.63
Weighted F1  : 0.63
```

Les catégories **Happy** et **Surprise** font partie des émotions les mieux reconnues par le modèle.

Les catégories **Fear**, **Sad** et **Angry** restent plus difficiles à distinguer.

---

# 🔥 Matrice de confusion

Une matrice de confusion est générée avec **Scikit-learn** et **Seaborn** afin d'analyser les erreurs du modèle.

```python
cm = confusion_matrix(all_labels, all_preds)
```

Puis :

```python
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    xticklabels=classes,
    yticklabels=classes
)
```

Cette visualisation permet d'identifier les émotions que le modèle confond le plus souvent.

---

# 💾 Sauvegarde du modèle

Le modèle ResNet18 entraîné est sauvegardé avec :

```python
torch.save(
    resnet18.state_dict(),
    "resnet18_fer2013.pth"
)
```

Le projet sauvegarde également les résultats nécessaires pour une future application :

```python
np.save("confusion_matrix.npy", cm)
np.save("model_accuracy.npy", np.array([accuracy]))
```

ainsi que le rapport de classification :

```python
with open("classification_report.txt", "w") as f:
    f.write(report)
```

---

# 🛠️ Technologies utilisées

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab
* CNN
* ResNet18
* Transfer Learning

---

# 📂 Structure suggérée du projet

```text
FER2013-Emotion-Recognition/
│
├── data_mininig_said.ipynb
│
├── resnet18_fer2013.pth
│
├── confusion_matrix.npy
│
├── model_accuracy.npy
│
├── classification_report.txt
│
└── README.md
```

---

# ▶️ Installation

Cloner le dépôt :

```bash
git clone https://github.com/datasifaw/Projet_Expressions_faciales-motions-
```

Puis installer les dépendances :

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn jupyter
```

---

# 📁 Préparation des données

Le notebook utilise une archive :

```text
archive.zip
```

Dans la version actuelle du notebook, son chemin est :

```python
zip_path = "/content/archive.zip"
```

L'archive est extraite dans :

```text
/content/fer2013/
```

avec une structure de type :

```text
fer2013/
│
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

---

# ▶️ Exécution

Lancer le notebook puis exécuter les cellules dans l'ordre afin de :

1. Importer les bibliothèques ;
2. Décompresser le dataset ;
3. Préparer les images ;
4. Appliquer la Data Augmentation ;
5. Entraîner le CNN ;
6. Tester le CNN ;
7. Charger ResNet18 ;
8. Effectuer le fine-tuning ;
9. Optimiser le Learning Rate ;
10. Evaluer le modèle final ;
11. Générer la matrice de confusion ;
12. Sauvegarder le modèle.

---

# 🔮 Améliorations possibles

Plusieurs améliorations peuvent être explorées :

* Early Stopping ;
* Séparation train / validation / test ;
* Gestion du déséquilibre entre les classes ;
* Pondération de `CrossEntropyLoss` ;
* Sampler équilibré ;
* gel puis dégel progressif des couches de ResNet18 ;
* Comparaison avec ResNet34, EfficientNet ou MobileNet ;
* Optimisation systématique des hyperparamètres ;
* Ajout de Batch Normalization ;
* Réduction du surapprentissage ;
* Détection automatique d'un visage avant classification ;
* Création d'une application Streamlit ;
* Prédiction en temps réel avec une webcam ;
* Déploiement du modèle via une API.

---

# 💡 Conclusion

Ce projet illustre plusieurs étapes importantes d'un pipeline de Deep Learning en Computer Vision :

* Préparation et augmentation des données ;
* Construction d'un CNN from scratch ;
* Transfer Learning ;
* Fine-tuning d'un réseau pré-entraîné ;
* Optimisation du Learning Rate ;
* Comparaison expérimentale des modèles ;
* Evaluation avec plusieurs métriques ;
* Sauvegarde du modèle pour son utilisation future.

Le meilleur modèle obtenu est le **ResNet18 optimisé**, avec une accuracy finale de :

# **62,69 % 🎯**
