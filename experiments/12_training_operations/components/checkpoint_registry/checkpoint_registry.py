
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class CheckpointRecord(Base):
    """
    SQLAlchemy Model for the Checkpoints Table.
    Stores metadata about where the checkpoint lives (S3) and its status.
    """
    __tablename__ = 'checkpoints'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, index=True)
    step = Column(Integer)
    s3_key = Column(String, unique=True)
    
    # Metadata
    loss = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Governance Tags
    tag = Column(String)  # 'growth', 'lora', 'temporary'
    is_protected = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Checkpoint(step={self.step}, tag='{self.tag}', protected={self.is_protected})>"

class CheckpointRegistry:
    """
    The Governance Layer for Checkpoints.
    Enforces 'No Delete' rules for critical checkpoints.
    Connects to Postgres (AWS RDS) or SQLite (Local).
    """
    def __init__(self, db_url: str = "sqlite:////tmp/checkpoints.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        print(f"✓ CheckpointRegistry connected to {db_url}")

    def register_checkpoint(self, run_id: str, step: int, s3_key: str, loss: float, tag: str = "temporary"):
        """
        Register a new checkpoint after it has been uploaded to S3.
        Auto-protects 'growth' and 'lora' tags.
        """
        session = self.Session()
        try:
            # Policy Logic: Auto-protect certain tags
            is_protected = tag in ['growth', 'lora', 'release_candidate']
            
            record = CheckpointRecord(
                run_id=run_id,
                step=step,
                s3_key=s3_key,
                loss=loss,
                tag=tag,
                is_protected=is_protected
            )
            session.add(record)
            session.commit()
            print(f"✓ Registered checkpoint: {s3_key} (Tag: {tag}, Protected: {is_protected})")
            return record.id
        except Exception as e:
            session.rollback()
            print(f"✗ Failed to register checkpoint: {e}")
            raise
        finally:
            session.close()

    def can_delete(self, s3_key: str) -> bool:
        """
        Policy Check: Is it safe to delete this checkpoint?
        Returns False if the checkpoint is protected.
        """
        session = self.Session()
        record = session.query(CheckpointRecord).filter_by(s3_key=s3_key).first()
        session.close()
        
        if not record:
            # If we don't know about it, assume it's unsafe to delete automatically
            # (Or safe, depending on your risk tolerance. Here we say unsafe).
            print(f"⚠️  Unknown checkpoint {s3_key}. Preventing deletion.")
            return False
            
        if record.is_protected:
            print(f"⛔ Blocked deletion of protected checkpoint {s3_key} (Tag: {record.tag})")
            return False
            
        return True

    def mark_for_deletion(self, s3_key: str):
        """
        Remove fro registry. ONLY if not protected.
        """
        if not self.can_delete(s3_key):
            raise ValueError(f"Cannot delete protected checkpoint {s3_key}")
            
        session = self.Session()
        session.query(CheckpointRecord).filter_by(s3_key=s3_key).delete()
        session.commit()
        session.close()
        print(f"✓ Removed checkpoint record: {s3_key}")
