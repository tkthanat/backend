from sqlalchemy import text
from app.db_models import engine

try:
    with engine.connect() as conn:
        res = conn.execute(text("SELECT 1"))
        print("✅ Database connected:", res.scalar())
except Exception as e:
    print("❌ Database connection failed:", e)
