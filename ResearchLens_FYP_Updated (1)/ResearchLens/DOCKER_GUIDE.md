# Optional Docker setup

Windows users can use START_WINDOWS.bat without installing Docker.

```bash
docker build -t researchlens .
docker volume create researchlens-data
docker run -d --name researchlens --restart unless-stopped -p 127.0.0.1:8501:8501 --env-file .env -v researchlens-data:/app/data researchlens
```

Open http://localhost:8501. Keep the named volume to retain accounts, indexed
papers and research notes when replacing the container. Use one application process per local SQLite data directory. Use HTTPS via a reverse proxy
for remote access; port 8501 is bound to loopback here.

Never bake real API keys into the Docker image. Back up the complete data volume.
The Dockerfile runs as an unprivileged user. Docker building/running was not
available for this release's validation; Python/UI tests were executed locally.
