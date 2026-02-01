# Utiliser une image Python légère
FROM python:3.11-slim

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système (nécessaire pour certains packages)
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Copie des requirements et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code de l'application
COPY . .

# On expose le port 8000
EXPOSE 8000

# Commande de lancement
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]