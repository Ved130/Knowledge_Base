from sqlalchemy import create_engine,Text,Column,Integer,DateTime,String
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

Db_url = os.getenv("DATABASE_URL")

engine = create_engine(Db_url)
sessionlocal = sessionmaker(bind = engine)
Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer,primary_key=True,index=True)
    question = Column(Text,nullable=False)
    answer = Column(Text,nullable=False)
    task_type = Column(String(50))
    created_at = Column(DateTime,default=datetime.now)

def init_db():
    Base.metadata.create_all(bind = engine)
    print("Database tables created")

def save_convo(question:str,answer:str,task_type:str = "qa"):
    db = sessionlocal()

    try:
        convo = Conversation(
            question = question,
            answer = answer,
            task_type = task_type
        )

        db.add(convo)
        db.commit()

    finally:
        db.close()

def get_recent(limit:int = 5):
    db = sessionlocal()

    try:
        convo = db.query(Conversation).order_by(Conversation.created_at.desc()).limit(limit).all()

        output = []
        for c in convo:
            user = {"question":c.question,"answer":c.answer}
            output.append(user)

        return output
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
            
    