import os
import traceback
from io import BytesIO
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
import bcrypt
from pydantic import BaseModel
from sqlalchemy import Column, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from PIL import Image

try:
  from dotenv import load_dotenv

  load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
  pass

from local_ai import generate_reply, model_status

# Todos os caminhos são resolvidos a partir da pasta onde este arquivo está,
# e não da pasta em que o comando foi executado. Isso evita bugs difíceis de
# diagnosticar (banco "sumindo", template "não encontrado" etc.) quando o
# servidor é iniciado de um diretório diferente.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(BASE_DIR, ".runtime", "huggingface"))

# A IA de visão (CLIP) é carregada somente quando uma imagem for enviada,
# para não bloquear o servidor e as telas de login/histórico enquanto o
# modelo é baixado.
DEVICE = "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"
model = None
processor = None
AI_OK = False


def load_ai_model():
  global model, processor, AI_OK, DEVICE
  if AI_OK:
    return True
  try:
    print("Carregando modelo de IA de visão (CLIP)... Aguarde.")
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    AI_OK = True
    print("Modelo de IA de visão carregado com sucesso!")
  except Exception as e:
    print(f"Aviso: IA de visão desativada por erro: {e}")
  return AI_OK


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
- Use estas orientações básicas como referência: papel e papelão devem estar
  limpos e secos; esvazie e achate caixas de papelão para reduzir o volume.
  Separe restos de comida dos materiais recicláveis. A aceitação de embalagens
  e materiais especiais depende da coleta local. Não invente etapas de preparo.
- Não invente pontos de coleta ou regras municipais. Quando depender da cidade,
  peça a localização e recomende confirmar com o serviço local de coleta.
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


async def gerar_resposta_ia(mensagem_usuario: str, historico: list) -> str:
  return await generate_reply(SYSTEM_PROMPT_AMBIENTAL, mensagem_usuario, historico)


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
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'reciclai.db')}")
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
print("Banco de dados inicializado.")


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
async def health():
  """Endpoint simples para checar rapidamente o estado do servidor."""
  local = await model_status()
  return {
      "status": "ok",
      "ia_visao_carregada": AI_OK,
      "ia_chat_configurada": local["available"],
      "ia_chat": local,
  }


@app.post("/api/register")
def register(user: UserAuth, db: Session = Depends(get_db)):
  username = user.username.strip()
  password = user.password

  if not username or not password:
    raise HTTPException(
        status_code=400, detail="Usuário e senha são obrigatórios."
    )

  if len(password.encode("utf-8")) > 72:
    raise HTTPException(400, "A senha deve ter no máximo 72 bytes em UTF-8.")

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

  if (not db_user or len(user.password.encode("utf-8")) > 72
      or not verify_password(user.password, db_user.hashed_password)):
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
  message = message.strip()
  if not db.get(User, user_id):
    raise HTTPException(404, "Usuário não encontrado.")
  if not message and not (file and file.filename):
    raise HTTPException(400, "Envie uma mensagem ou uma imagem.")
  if len(message) > 6000:
    raise HTTPException(400, "A mensagem deve ter no máximo 6000 caracteres.")

  # Histórico recente ANTES de salvar a mensagem atual, usado como contexto
  # para a IA de chat.
  historico_ia = obter_historico_para_ia(db, user_id)

  # Salva mensagem do usuário no banco
  user_msg_content = message
  if file and file.filename:
    user_msg_content += f" [Arquivo enviado: {file.filename}]"

  # Processamento de Imagem com IA de visão (CLIP)
  if file and file.filename:
    data = await file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
      raise HTTPException(413, "A imagem deve ter no máximo 10 MB.")
    try:
      img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
      raise HTTPException(400, "Envie um arquivo de imagem válido.") from exc
    if not await run_in_threadpool(load_ai_model):
      raise HTTPException(503, "O modelo local de imagens está indisponível.")
    try:
      import torch
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
  elif message:
    # Chat de conversa livre, especializado em meio ambiente.
    resposta_texto = await gerar_resposta_ia(message, historico_ia)
  else:
    resposta_texto = "Por favor, envie uma mensagem ou anexe uma foto."

  # Salva resposta do assistente no banco
  db.add(Message(user_id=user_id, role="user", content=user_msg_content))
  db.add(
      Message(user_id=user_id, role="assistant", content=resposta_texto)
  )
  db.commit()

  return {"response": resposta_texto}


if __name__ == "__main__":
  import uvicorn

  # Permite rodar tanto com "python app.py" quanto com
  # "uvicorn app:app --reload" (recomendado para desenvolvimento).
  uvicorn.run(app, host="127.0.0.1", port=8000)
