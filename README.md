# ReCiclaí

**Uma assistente de inteligência artificial local para transformar dúvidas sobre resíduos em decisões mais conscientes.**

Separar um material para reciclagem parece simples, mas as orientações variam conforme o tipo de resíduo e a coleta disponível em cada cidade. O ReCiclaí reúne conversa e identificação visual em uma interface acessível para ajudar as pessoas a descobrir como descartar, reaproveitar e entender melhor o impacto dos materiais.

## O que o projeto oferece

- Chat em português para dúvidas sobre reciclagem, compostagem, reuso e sustentabilidade, com histórico de conversas.
- Identificação experimental de imagens de resíduos usando um modelo de visão computacional.
- Cadastro local com senhas armazenadas em formato de hash.
- Interface adaptável a computadores e celulares, com sugestões para iniciar uma conversa.
- Processamento da conversa por um modelo de linguagem executado localmente com Ollama, sem chave de API paga.

## Como funciona

O backend em **Python e FastAPI** integra as funções do aplicativo. O **Hermes 3 8B** responde às mensagens através do **Ollama**. Para imagens, o aplicativo carrega sob demanda o **CLIP** (Hugging Face Transformers), um componente separado do modelo de conversa. As contas e mensagens são guardadas em **SQLite**. O frontend usa HTML, CSS e JavaScript, sem uma etapa de compilação.

Na primeira execução, é preciso baixar o modelo Hermes (cerca de 4,7 GB). O primeiro envio de imagem também baixa os arquivos do CLIP e pode exigir internet. Depois desses downloads, a inferência dos modelos ocorre no próprio computador. A velocidade depende da memória e do processador disponíveis.

## Executar no Windows

Os passos completos, em formato de texto simples, estão em [`EXECUTAR.txt`](EXECUTAR.txt). Em resumo: instale Python 3.12 e Ollama, instale as dependências Python, baixe `hermes3:8b` e execute `tools/Start-Local.ps1`. O aplicativo abre em `http://127.0.0.1:8000`.

O modelo de conversa e a identificação visual são recursos diferentes: o Hermes recebe texto; o CLIP compara a imagem com categorias predefinidas. A identificação visual é exploratória e não reconhece todos os produtos nem garante a destinação correta.

## Limitações e responsabilidade

As regras de coleta variam por município; confirme as orientações com o serviço local antes do descarte. As respostas e classificações são educativas e podem estar incorretas. Este projeto é um protótipo para execução local, não um serviço pronto para exposição na internet: a autenticação e a autorização das rotas ainda precisam ser reforçadas antes de uso multiusuário ou público.

## Tecnologias

Python · FastAPI · SQLite · Ollama · Hermes 3 8B · PyTorch · Hugging Face Transformers/CLIP · HTML · CSS · JavaScript

## Créditos dos modelos

- [Hermes 3 8B no Ollama](https://ollama.com/library/hermes3:8b)
- [Ollama para Windows](https://docs.ollama.com/windows)
- [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32)
