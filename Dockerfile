FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip==26.0 \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]