# Cocoruta
Cocoruta is a LLM developed in a domain-driven way in order to be specialized in Blue Amazon legislation.

# Running the model within web interface

## Requirements
- Docker installed
- Windows or Linux OS

## Clone repo
```bash
git clone https://github.com/felipeoes/cocoruta.git
```

## Change to repo directory
```bash
cd cocoruta
```

## Build docker image for Cocoruta UI
```bash
cd chatbot-ui
docker build -t cocoruta_ui:latest .
```

## Run application
```bash
cd ..
docker compose up
```