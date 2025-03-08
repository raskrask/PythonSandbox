FROM python:3.11

RUN apt-get update && apt-get install -y \
    vim curl wget git zip unzip bash iputils-ping net-tools \
    libatlas-base-dev awscli\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

CMD ["tail", "-f", "/dev/null"]
