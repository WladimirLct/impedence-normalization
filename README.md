# Projet YESPRO-35

## Mission
Développer une application de visualisation et d'analyse de courbes de spectroscopie par impédance.

## Instructions d'installation
1. **Télécharger** l'archive au format .zip en cliquant ici :
   1. https://github.com/WladimirLct/impedence-normalization/archive/refs/heads/main.zip
2. **Décompresser** l'archive
3. **Double-cliquer** sur le fichier `Installer` et attendre que la fenêtre se ferme automatiquement
4. **Double-cliquer** sur le fichier `Lancer`
-> **Installation terminée**

## Conseils d'utilisation
1. **Dossiers temporaires** : Deux dossiers se créent lors du premier lancement et contiennent des fichiers éphémères supprimables si nécessaire :
   - `tmp/`
   - `normalizations/`
2. **Fermeture de l'application** : Pour fermer complètement l'application, fermez la fenêtre **cmd** ou **invite de commandes** qui s'ouvre après le démarrage.
3. **Environnement Python** : L'environnement Python de l'application sera installé dans le dossier `.venv/`. En cas de besoin d'espace disque, vous pouvez le supprimer sans problème, il sera réinstallé seul (~1Go).

## Architecture des fichiers
```
(.venv/)             # Contient l'environnement python et les librairies
(tmp/)               # Contient les fichiers temporaires avec les informations sur les graphiques
(normalizations/)    # Contient les normalisations de max. 20 expériences déjà lancées

scripts/             # Contient les fichiers scripts d'installation et de lancement

app/
├── assets/          # Contient toutes les icônes
├── callbacks/       # Contient tous les appels de callbacks
├── config/          # Contient les fichiers de configuration (fréquence par défaut, etc.)
├── pages/           # Contient le layout de chaque page
├── utils/           # Contient toutes les fonctions helpers utilisées dans les callbacks
└── main.py          # Fichier d'initialisation de l'application
```

**⚠️ Important** : *Le reste des fichiers non mentionné sert à configurer & installer l'environnement Python, ils sont tous indispensables !*

## Équipe de développement
- **Wladimir LUCET**
- **Antoine MIGNIEN**
- **Victor DUSSAUSSOIS**
- **Bilal BOUSSARI**
- **William SAVRE**

> [Présentation de projet](https://www.canva.com/design/DAGt63vEi4Y/Olky_ElUDvMrCBJ_iYXuxA/edit?utm_content=DAGt63vEi4Y&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)