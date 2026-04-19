import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

ENV = os.getenv('RUN_ENV', 'local')

if ENV == 'docker':
    DATABASE_URL = "sqlite:////app/face.db"
else:
    DATABASE_URL = "sqlite:///../data/face.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()