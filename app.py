import os
import traceback
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import bcrypt
from pydantic import BaseModel
from sqlalchemy import Column, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
import torch
from PIL import Image

try:
  from dotenv import load_dotenv

  load_dotenv()
except ImportError:
  pass

# Todos os caminhos são resolvidos a partir da pasta onde este arquivo está,
# e não da pasta em que o comando foi executado. Isso evita bugs difíceis de
# diagnosticar (banco "sumindo", template "não encontrado" etc.) quando o
# servidor é iniciado de um diretório diferente.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# A IA de visão (CLIP) é carregada somente quando uma imagem for enviada,
# para não bloquear o servidor e as telas de login/histórico enquanto o
# modelo é baixado.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"
model = None
processor = None
AI_OK = False


def load_ai_model():
  global model, processor, AI_OK
  if AI_OK:
    return True
  try:
    print("Carregando modelo de IA de visão (CLIP)... Aguarde.")
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    AI_OK = True
    print("Modelo de IA de visão carregado com sucesso!")
  except Exception as e:
    print(f"Aviso: IA de visão desativada por erro: {e}")
  return AI_OK


# --- IA de conversa (chat de texto especializado em meio ambiente) ---
# Usa a API da Anthropic. Requer a variável de ambiente ANTHROPIC_API_KEY.
# Sem ela, o chat de texto continua funcionando com uma mensagem avisando
# que a chave não foi configurada, em vez de travar o servidor.
ai_client = None
try:
  import anthropic

  _api_key = os.environ.get("ANTHROPIC_API_KEY")
  if _api_key:
    ai_client = anthropic.Anthropic(api_key=_api_key)
  else:
    print(
        "Aviso: ANTHROPIC_API_KEY não definida - o chat por IA vai responder"
        " com um aviso até a chave ser configurada."
    )
except ImportError:
  print("Aviso: pacote 'anthropic' não instalado - chat por IA desativado.")

CHAT_MODEL = "claude-sonnet-5"  # troque para "claude-haiku-4-5-20251001" se quiser respostas mais rápidas/baratas

SYSTEM_PROMPT_AMBIENTAL = """Você é o assistente de conversa do ReCiclaí, um aplicativo de reciclagem e sustentabilidade.

Seu papel é conversar de forma natural e prestativa, especializado em:
- reciclagem, coleta seletiva e separação correta de resíduos;
- reuso e reaproveitamento criativo de materiais;
- compostagem e resíduos orgânicos;
- sustentabilidade, consumo consciente e economia circular;
- meio ambiente, mudanças climáticas, poluição, fauna, flora e ecologia em geral.

Regras de resposta:
- Responda sempre em português do Brasil.
- Seja claro, amigável e direto; evite respostas longas demais.
- Se a pergunta do usuário não tiver relação com meio ambiente, responda
  normalmente mesmo assim (você é um assistente de conversa completo), mas,
  quando fizer sentido, conecte a resposta a alguma dica prática de
  sustentabilidade.
- Você não analisa fotos diretamente pelo chat de texto - a identificação de
  resíduos por imagem é feita por um modelo de visão computacional separado
  no mesmo app. Se o usuário perguntar sobre uma foto, oriente-o a enviá-la
  pelo botão de anexo (📎) do chat.
- Pode usar **negrito**, *itálico*, uma linha "### Título" e listas com "- "
  quando ajudar a organizar a resposta - a interface do chat renderiza esses
  elementos. Evite tabelas, blocos de código e markdown mais complexo, pois
  não são exibidos corretamente."""


def obter_historico_para_ia(db: Session, user_id: int, limite: int = 12):
  """Busca as últimas mensagens do usuário para dar contexto à IA de chat."""
  msgs = (
      db.query(Message)
      .filter(Message.user_id == user_id)
      .order_by(Message.id.desc())
      .limit(limite)
      .all()
  )
  msgs.reverse()
  return [{"role": m.role, "content": m.content} for m in msgs]


def gerar_resposta_ia(mensagem_usuario: str, historico: list) -> str:
  if ai_client is None:
    return (
        "O chat por IA ainda não está configurado neste servidor. Defina a"
        " variável de ambiente ANTHROPIC_API_KEY com uma chave da API da"
        " Anthropic (veja o README) e reinicie o servidor."
    )
  try:
    mensagens = historico + [{"role": "user", "content": mensagem_usuario}]
    resposta = ai_client.messages.create(
        model=CHAT_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT_AMBIENTAL,
        messages=mensagens,
    )
    texto = "".join(
        bloco.text for bloco in resposta.content if bloco.type == "text"
    ).strip()
    return texto or "Não consegui gerar uma resposta agora. Tente novamente."
  except Exception as e:
    traceback.print_exc()
    return f"Erro ao consultar a IA de chat: {e}"


app = FastAPI()

# CORS liberado: útil caso o front-end seja acessado de uma porta/origem
# diferente da do backend (ex.: testando com Live Server). Não usamos
# cookies/sessão baseada em credenciais do navegador, então isso é seguro
# aqui.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração de Pastas e Templates (caminhos absolutos - ver BASE_DIR)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Configuração do Banco de Dados (SQLite)
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'reciclai.db')}"
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Modelos do Banco de Dados
class User(Base):
  __tablename__ = "users"
  id = Column(Integer, primary_key=True, index=True)
  username = Column(String, unique=True, index=True)
  hashed_password = Column(String)
  messages = relationship("Message", back_populates="owner")


class Message(Base):
  __tablename__ = "messages"
  id = Column(Integer, primary_key=True, index=True)
  user_id = Column(Integer, ForeignKey("users.id"))
  role = Column(String)  # 'user' ou 'assistant'
  content = Column(Text)
  owner = relationship("User", back_populates="messages")


Base.metadata.create_all(bind=engine)
print(f"Banco de dados em uso: {os.path.join(BASE_DIR, 'reciclai.db')}")


# Segurança (Hash de Senhas)
def hash_password(password: str) -> str:
  return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode(
      "utf-8"
  )


def verify_password(password: str, hashed_password: str) -> bool:
  return bcrypt.checkpw(
      password.encode("utf-8"), hashed_password.encode("utf-8")
  )


def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()


# Base de Conhecimento de Resíduos
BASE = {
    "a plastic PET bottle": {
        "nome": "Garrafa PET",
        "lixeira": "Vermelha · plástico",
        "decomposicao": "Aproximadamente 400 anos",
        "reuso": [
            "Vaso autoirrigável com barbante.",
            "Horta vertical suspensa.",
            "Comedouro para pássaros.",
        ],
        "alerta": "Evite reutilizar para armazenar água potável por longos períodos.",
        "impacto": "1 kg de PET reciclado economiza cerca de 5,3 kWh de energia.",
    },
    "an aluminum can": {
        "nome": "Lata de alumínio",
        "lixeira": "Amarela · metal",
        "decomposicao": "200 a 500 anos",
        "reuso": [
            "Porta-lápis decorado.",
            "Luminária artesanal.",
            "Mini-horta de temperos.",
        ],
        "alerta": "Bordas cortadas são afiadas. Lixe ou cubra com fita isolante.",
        "impacto": "Reciclar alumínio gasta 95% menos energia.",
    },
    "a glass bottle or glass jar": {
        "nome": "Vidro (garrafa ou pote)",
        "lixeira": "Verde · vidro",
        "decomposicao": "Mais de 4.000 anos",
        "reuso": [
            "Potes herméticos para temperos.",
            "Vaso para plantas aquáticas.",
        ],
        "alerta": "Vidro quebrado deve ser embalado em jornal e sinalizado.",
        "impacto": "O vidro é 100% reciclável infinitas vezes.",
    },
    "cardboard box": {
        "nome": "Papelão",
        "lixeira": "Azul · papel",
        "decomposicao": "Aproximadamente 3 meses",
        "reuso": ["Organizadores de gaveta.", "Composteira seca."],
        "alerta": "Papelão engordurado não é reciclável.",
        "impacto": "1 tonelada de papel poupa cerca de 20 árvores.",
    },
    "organic food waste, fruit or vegetable scraps": {
        "nome": "Resíduo orgânico",
        "lixeira": "Marrom · orgânico",
        "decomposicao": "Algumas semanas",
        "reuso": ["Composteira doméstica.", "Adubo rico em cálcio."],
        "alerta": "Não composte carnes ou laticínios em sistemas caseiros.",
        "impacto": "Metade do lixo doméstico é orgânico.",
    },
}
PROMPTS = list(BASE.keys())
PROMPTS_EXTRA = [
    "a person",
    "a landscape",
    "an animal",
    "a building",
    "food on a plate",
    "a generic background",
]
TODOS_PROMPTS = PROMPTS + PROMPTS_EXTRA


# Esquemas Pydantic
class UserAuth(BaseModel):
  username: str
  password: str


# --- ROTAS DA API ---


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
  return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
def health():
  """Endpoint simples para checar rapidamente o estado do servidor."""
  return {
      "status": "ok",
      "banco_de_dados": os.path.join(BASE_DIR, "reciclai.db"),
      "ia_visao_carregada": AI_OK,
      "ia_chat_configurada": ai_client is not None,
  }


@app.post("/api/register")
def register(user: UserAuth, db: Session = Depends(get_db)):
  username = user.username.strip()
  password = user.password

  if not username or not password:
    raise HTTPException(
        status_code=400, detail="Usuário e senha são obrigatórios."
    )

  existing = db.query(User).filter(User.username == username).first()
  if existing:
    raise HTTPException(status_code=400, detail="Usuário já existe.")

  try:
    hashed = hash_password(password)
    new_user = User(username=username, hashed_password=hashed)
    db.add(new_user)
    db.commit()
  except Exception as e:
    db.rollback()
    traceback.print_exc()
    raise HTTPException(
        status_code=500, detail=f"Erro interno ao criar usuário: {e}"
    )

  return {"success": True, "message": "Conta criada com sucesso!"}


@app.post("/api/login")
def login(user: UserAuth, db: Session = Depends(get_db)):
  try:
    db_user = db.query(User).filter(User.username == user.username.strip()).first()
  except Exception as e:
    traceback.print_exc()
    raise HTTPException(
        status_code=500, detail=f"Erro interno ao consultar usuário: {e}"
    )

  if not db_user or not verify_password(user.password, db_user.hashed_password):
    raise HTTPException(status_code=400, detail="Usuário ou senha incorretos.")

  return {"success": True, "user_id": db_user.id, "username": db_user.username}


@app.get("/api/history/{user_id}")
def get_history(user_id: int, db: Session = Depends(get_db)):
  messages = (
      db.query(Message)
      .filter(Message.user_id == user_id)
      .order_by(Message.id.asc())
      .all()
  )
  return [{"role": m.role, "content": m.content} for m in messages]


@app.post("/api/chat")
async def chat(
    user_id: int = Form(...),
    message: str = Form(""),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
  resposta_texto = ""

  # Histórico recente ANTES de salvar a mensagem atual, usado como contexto
  # para a IA de chat.
  historico_ia = obter_historico_para_ia(db, user_id)

  # Salva mensagem do usuário no banco
  user_msg_content = message
  if file and file.filename:
    user_msg_content += f" [Arquivo enviado: {file.filename}]"

  db.add(Message(user_id=user_id, role="user", content=user_msg_content))
  db.commit()

  # Processamento de Imagem com IA de visão (CLIP)
  if file and file.filename and load_ai_model():
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
      buffer.write(await file.read())
    try:
      img = Image.open(filepath).convert("RGB")
      entradas = processor(
          text=TODOS_PROMPTS, images=img, return_tensors="pt", padding=True
      ).to(DEVICE)
      with torch.no_grad():
        saida = model(**entradas)
      probs = saida.logits_per_image.softmax(dim=1)[0].tolist()
      ranking = sorted(zip(TODOS_PROMPTS, probs), key=lambda x: -x[1])
      top, conf = ranking[0]

      if top in PROMPTS_EXTRA or conf < 0.35:
        resposta_texto = (
            "Não consegui identificar o resíduo com confiança na imagem."
        )
      else:
        d = BASE[top]
        reusos = "\n".join(f"- {r}" for r in d["reuso"])
        resposta_texto = f"""### {d['nome']} ({conf*100:.0f}%)
**Lixeira:** {d['lixeira']}
**Decomposição:** {d['decomposicao']}

**Sugestões de Reutilização:**
{reusos}

**Atenção:** {d['alerta']}
*{d['impacto']}*"""
    except Exception as e:
      resposta_texto = f"Erro ao processar imagem: {str(e)}"
    finally:
      if os.path.exists(filepath):
        os.remove(filepath)
  elif message:
    # Chat de conversa livre, especializado em meio ambiente.
    resposta_texto = gerar_resposta_ia(message, historico_ia)
  else:
    resposta_texto = "Por favor, envie uma mensagem ou anexe uma foto."

  # Salva resposta do assistente no banco
  db.add(
      Message(user_id=user_id, role="assistant", content=resposta_texto)
  )
  db.commit()

  return {"response": resposta_texto}


if __name__ == "__main__":
  import uvicorn

  # Permite rodar tanto com "python app.py" quanto com
  # "uvicorn app:app --reload" (recomendado para desenvolvimento).
  uvicorn.run(app, host="0.0.0.0", port=8000)