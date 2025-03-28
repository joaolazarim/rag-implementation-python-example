# Exemplo de recuperação de documentos com SentenceTransformers e FAISS

Este exemplo demonstra como realizar **busca semântica** em uma base de conhecimento utilizando o modelo `all-MiniLM-L6-v2` da `sentence-transformers` e o índice vetorial `FAISS`.

## Instalação das dependências

Rode o seguinte comando para instalar as bibliotecas necessárias:

```
pip install sentence-transformers faiss-cpu
```

## Como rodar

Execute o script principal:

```
python main.py
```

## Exemplo de uso

O código irá ler o arquivo `base_conhecimento.txt`, gerar embeddings com o modelo `all-MiniLM-L6-v2`, criar um índice vetorial com FAISS, e responder à seguinte pergunta:

```
Qual é a capital do Brasil?
```

A saída será uma lista com os 3 documentos mais relevantes para essa consulta.

---

> Este projeto é um exemplo básico de RAG (Retrieval-Augmented Generation), focando apenas na parte de recuperação de contexto.