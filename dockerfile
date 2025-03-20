# Utilise une image de base Python
FROM python:3.12-slim

# Crée un répertoire /app dans le conteneur
WORKDIR /app

# Copie le fichier requirements.txt dans le répertoire /app du conteneur
COPY requirements.txt /app/

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste des fichiers du répertoire local dans /app du conteneur
COPY . /app/

# Expose le port 8000
EXPOSE 8000

# Démarre l'application FastAPI avec uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]