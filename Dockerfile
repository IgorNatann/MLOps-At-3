FROM python:3.10-slim 

# Instala libgomp e limpa o cache do apt para manter a imagem leve
RUN apt-get update && apt-get install -y libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copia os arquivos do projeto para dentro do container
COPY . .

# Comando para rodar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]