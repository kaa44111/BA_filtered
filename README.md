# Schritte zur besseren einsicht
(Siehe: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W%26B_Artifacts.ipynb#scrollTo=0Fr8vsH0C-S2)

## Dataset 
Teile die Data nach train.data & test.data
* Trainings Data :
    - Training Dataset
    - Validierungs Dataset 
* Test Data:
    - Test dataset 
* Methode:
    - Eine Methode (ex. load()), um die Datasets zu definieren.

## Project Tracking
Am Anfang oder am Ende. Speichert die Info über die Datasets. Es erzeugt keine Charts nur die Artefakten von den bestimmten Datasets.

* Initialisieren:
    1. create a Run with wandb.init
    2. create an Artifact for the dataset 
    3. save and log the associated files
* Jede Dataset bekommt einen eignen Projekt:
    - Projektname : Datasetname
        - Aktion in einem Projekt haben unterschiedlichern Job : z.B Datasetinfo, Info über preprozess, Modell usw...
            - Diese Jobs haben unterschiedlichere Artefakte 

## Preprocess
Logge die Preprozessing der Bilddaten in den Artifacts, damit man sehen kann welche preprozessing durchgeführt wurde. Damit nicht verloren geht was man verändert hat.

## Modell
1. Initialiesiern des Modells
2. Modell mit wandb.config (object to store all of the hyperparameters)

## Die schon gespeicherten Artefakte zum trainieren benutzen
- Use a Logged Model Artifact (https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)


