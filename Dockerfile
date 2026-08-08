FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application runtime modules (must match app.py imports)
COPY app.py .
COPY cy_search.py .
COPY cy_search_real.py .
COPY datasets_registry.py .
COPY hall_of_fame.py .
COPY geometry_store.py .
COPY physics_dossier.py .
COPY physics_extensions.py .
COPY job_store.py .
COPY ml_roadmap.py .
COPY templates/ templates/
COPY static/ static/
COPY data/ data/
COPY scripts/ scripts/

# Create data directory for KS database cache (also holds baked seed JSON)
RUN mkdir -p data/ks_cache

# Expose port 5102
EXPOSE 5102

# Run with gunicorn (Coolify sets PORT dynamically)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5102} --workers 4 --timeout 120 app:app"]

# Persistent Coolify volume mounts at /app/static/data
# (hall_of_fame.sqlite, geometry.sqlite). geometry.sqlite is created on boot
# if missing; baked seeds are upserted from data/*.json without wiping offline rows.
