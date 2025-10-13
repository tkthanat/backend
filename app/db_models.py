from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, LargeBinary, JSON, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from .config import settings
import os

Base = declarative_base()
engine = create_engine(settings.db_url, echo=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Subject(Base):
    __tablename__ = "subjects"
    subject_id = Column(Integer, primary_key=True, autoincrement=True)
    subject_name = Column(String(255), nullable=False)
    section = Column(String(100))
    schedule = Column(String(255))
    users = relationship("User", back_populates="subject")
    logs = relationship("AttendanceLog", back_populates="subject")

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    role = Column(Enum("admin","operator","viewer"), nullable=False)
    password_hash = Column(String(255))
    embeddings_enc = Column(LargeBinary)
    subject_id = Column(Integer, ForeignKey("subjects.subject_id", ondelete="SET NULL"))
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    subject = relationship("Subject", back_populates="users")

class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"))
    subject_id = Column(Integer, ForeignKey("subjects.subject_id", ondelete="SET NULL"))
    action = Column(Enum("enter","exit"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id = Column(String(50), nullable=False)
    flags = Column(JSON)
    confidence = Column(Float)
    user = relationship("User")
    subject = relationship("Subject", back_populates="logs")
    alert = relationship("Alert", back_populates="log", uselist=False)

class Alert(Base):
    __tablename__ = "alerts"
    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(Integer, ForeignKey("attendance_logs.log_id", ondelete="CASCADE"))
    type = Column(Enum("short_stay","multiple_faces","low_confidence","spoofing"), nullable=False)
    confidence = Column(Float)
    status = Column(Enum("open","approved","rejected"), default="open", index=True)
    log = relationship("AttendanceLog", back_populates="alert")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    audit_id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)
    record_id = Column(Integer)
    user_actor = Column(String(255))
    changes = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()