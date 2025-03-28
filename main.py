from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')

def carregar_documentos(caminho_arquivo):
    with open(caminho_arquivo, 'r') as arquivo:
        documentos = arquivo.read().split('\n\n')
    return documentos

def gerar_embeddings(documentos):
    embeddings = model.encode(documentos)
    return embeddings

def criar_indice_faiss(embeddings):
    dimensao = embeddings.shape[1]
    indice = faiss.IndexFlatL2(dimensao)
    indice.add(embeddings)
    return indice

def recuperar_documentos(consulta, indice, documentos, top_k=3):
    embedding_consulta = model.encode([consulta])
    distancias, indices = indice.search(embedding_consulta, top_k)

    documentos_recuperados = [documentos[i] for i in indices[0]]
    return documentos_recuperados

documentos = carregar_documentos('base_conhecimento.txt')
embeddings = gerar_embeddings(documentos)
indice = criar_indice_faiss(embeddings)

consulta = "Qual é a capital do Brasil?"
documentos_relevantes = recuperar_documentos(consulta, indice, documentos)

print("Documentos relevantes:")
for doc in documentos_relevantes:
    print(doc)