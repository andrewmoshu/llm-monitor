class LatencyRecord(Base):
    __tablename__ = "latency_records"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    timestamp = Column(DateTime)
    latency_ms = Column(Float)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    cost = Column(Float)
    arena_score = Column(Float, nullable=True)
    context_window = Column(Integer) 