FROM python:3.10-slim

WORKDIR /app

# (Optional) Only install awscli if you really need it
# You are not using AWS, so this can be removed
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends awscli && \
#     rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]