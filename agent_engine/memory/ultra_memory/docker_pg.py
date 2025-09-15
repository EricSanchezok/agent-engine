from __future__ import annotations
import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))



import socket
import subprocess
import time
from dataclasses import dataclass
import os
from shutil import which
from typing import Optional

from agent_engine.agent_logger.agent_logger import AgentLogger


logger = AgentLogger("UltraMemoryDockerPG")


@dataclass
class DockerPGConfig:
    container_name: str = "ultramemory-pg"
    image: str = "pgvector/pgvector:pg16"
    image_candidates: list[str] = None  # type: ignore[assignment]
    host: str = "127.0.0.1"
    port: int = 55432
    user: str = "ultra"
    password: str = "ultra"
    database: str = "ultra"
    volume_name: str = "ultramemory_pg_data"
    wait_timeout_sec: int = 60


@dataclass
class DockerTSConfig:
    container_name: str = "ultramemory-ts"
    image: str = "timescale/timescaledb:latest-pg16"
    host: str = "127.0.0.1"
    port: int = 56432
    user: str = "ultra"
    password: str = "ultra"
    database: str = "ultra"
    volume_name: str = "ultramemory_ts_data"
    wait_timeout_sec: int = 60


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except Exception:
        return False


def ensure_docker_postgres(cfg: Optional[DockerPGConfig] = None) -> Optional[str]:
    """Ensure a PostgreSQL+pgvector container is running and return DSN.

    Returns DSN like: postgresql://user:password@host:port/database
    """
    cfg = cfg or DockerPGConfig()
    if cfg.image_candidates is None:
        cfg.image_candidates = [
            cfg.image,
            "ghcr.io/pgvector/pgvector:pg16",
            "m.daocloud.io/pgvector/pgvector:pg16",
        ]

    if which("docker") is None:
        logger.error("Docker executable not found in PATH. Please install Docker Desktop and retry.")
        return None

    # Ensure Docker daemon is up (Windows: attempt to start Docker Desktop)
    if not _ensure_docker_running():
        logger.error("Docker daemon is not running. Please start Docker Desktop and retry.")
        return None

    # Check if container exists
    ps = _run(["docker", "ps", "-a", "--filter", f"name=^{cfg.container_name}$", "--format", "{{.Status}}"])
    if ps.returncode != 0:
        logger.error(f"Failed to query docker: {ps.stderr.strip()}")
        return None
    status = (ps.stdout or "").strip()
    if status:
        # Container exists
        if "Up" in status:
            logger.info(f"Container {cfg.container_name} already running")
        else:
            logger.info(f"Starting existing container {cfg.container_name}")
            st = _run(["docker", "start", cfg.container_name])
            if st.returncode != 0:
                logger.error(f"Failed to start container: {st.stderr.strip()}")
                return None
    else:
        # Create new container
        last_err = None
        for image in cfg.image_candidates:
            logger.info(f"Creating container {cfg.container_name} from image {image}")
            # Try to pull first (helps clearer logs)
            pull = _run(["docker", "pull", image])
            if pull.returncode != 0:
                last_err = pull.stderr.strip() or pull.stdout.strip()
                logger.warning(f"Failed to pull {image}: {last_err}")
                continue
            run_cmd = [
                "docker", "run", "-d",
                "--name", cfg.container_name,
                "-e", f"POSTGRES_USER={cfg.user}",
                "-e", f"POSTGRES_PASSWORD={cfg.password}",
                "-e", f"POSTGRES_DB={cfg.database}",
                "-p", f"{cfg.port}:5432",
                "-v", f"{cfg.volume_name}:/var/lib/postgresql/data",
                image,
            ]
            rc = _run(run_cmd)
            if rc.returncode == 0:
                last_err = None
                break
            last_err = rc.stderr.strip() or rc.stdout.strip()
            logger.warning(f"Failed to run with {image}: {last_err}")
        if last_err:
            logger.error(f"Failed to run container with all candidate images. Last error: {last_err}")
            logger.error("If you're in a restricted network, configure Docker registry mirrors or pre-pull the image manually.")
            return None

    # Wait for port
    start_ts = time.time()
    while time.time() - start_ts < cfg.wait_timeout_sec:
        if _is_port_open(cfg.host, cfg.port):
            break
        time.sleep(1.0)
    else:
        logger.error(f"PostgreSQL port {cfg.host}:{cfg.port} did not open within timeout")
        return None

    dsn = f"postgresql://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}"
    def _mask(s: str) -> str:
        try:
            at = s.find('@')
            scheme = s.split('://', 1)[0]
            if '://' in s and ':' in s.split('://',1)[1].split('@',1)[0]:
                user = s.split('://',1)[1].split('@',1)[0].split(':',1)[0]
                hostpart = s.split('@',1)[1]
                return f"{scheme}://{user}:***@{hostpart}"
        except Exception:
            pass
        return s
    logger.info(f"PostgreSQL is ready at {_mask(dsn)}")

    # Try to create extension if psycopg is available
    try:
        import psycopg  # type: ignore

        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.commit()
                except Exception:
                    pass
    except Exception:
        # psycopg not installed or connection failed; non-fatal at this stage
        logger.warning("Could not verify/create extension 'vector' (psycopg missing or connection failed).")

    return dsn


def ensure_docker_timescale(cfg: Optional[DockerTSConfig] = None) -> Optional[str]:
    cfg = cfg or DockerTSConfig()
    if which("docker") is None:
        logger.error("Docker executable not found in PATH. Please install Docker Desktop and retry.")
        return None
    if not _ensure_docker_running():
        logger.error("Docker daemon is not running. Please start Docker Desktop and retry.")
        return None
    ps = _run(["docker", "ps", "-a", "--filter", f"name=^{cfg.container_name}$", "--format", "{{.Status}}"])
    if ps.returncode != 0:
        logger.error(f"Failed to query docker: {ps.stderr.strip()}")
        return None
    status = (ps.stdout or "").strip()
    if status:
        if "Up" in status:
            logger.info(f"Container {cfg.container_name} already running")
        else:
            st = _run(["docker", "start", cfg.container_name])
            if st.returncode != 0:
                logger.error(f"Failed to start container: {st.stderr.strip()}")
                return None
    else:
        run_cmd = [
            "docker", "run", "-d",
            "--name", cfg.container_name,
            "-e", f"POSTGRES_USER={cfg.user}",
            "-e", f"POSTGRES_PASSWORD={cfg.password}",
            "-e", f"POSTGRES_DB={cfg.database}",
            "-p", f"{cfg.port}:5432",
            "-v", f"{cfg.volume_name}:/var/lib/postgresql/data",
            cfg.image,
        ]
        rc = _run(run_cmd)
        if rc.returncode != 0:
            logger.error(f"Failed to run container: {rc.stderr.strip()} | {rc.stdout.strip()}")
            return None
    start_ts = time.time()
    while time.time() - start_ts < cfg.wait_timeout_sec:
        if _is_port_open(cfg.host, cfg.port):
            break
        time.sleep(1.0)
    else:
        logger.error(f"Timescale port {cfg.host}:{cfg.port} did not open within timeout")
        return None
    dsn = f"postgresql://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}"
    def _mask(s: str) -> str:
        try:
            scheme = s.split('://', 1)[0]
            user = s.split('://',1)[1].split('@',1)[0].split(':',1)[0]
            hostpart = s.split('@',1)[1]
            return f"{scheme}://{user}:***@{hostpart}"
        except Exception:
            return s
    logger.info(f"Timescale is ready at {_mask(dsn)}")
    return dsn


def _ensure_docker_running() -> bool:
    """Return True if docker daemon is running; try to start Docker Desktop on Windows if needed."""
    if _docker_ok():
        return True
    if os.name == "nt":
        _try_start_docker_desktop()
        # Wait for docker to come up
        deadline = time.time() + 120
        while time.time() < deadline:
            if _docker_ok():
                return True
            time.sleep(2)
        return False
    return False


def _docker_ok() -> bool:
    res = _run(["docker", "info"])
    return res.returncode == 0


def _try_start_docker_desktop() -> None:
    # Attempt to start Docker Desktop on Windows
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", r"C:\\Program Files"), "Docker", "Docker", "Docker Desktop.exe"),
        os.path.join(os.environ.get("ProgramW6432", r"C:\\Program Files"), "Docker", "Docker", "Docker Desktop.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)"), "Docker", "Docker", "Docker Desktop.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                subprocess.Popen([path], close_fds=True, creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
                logger.info(f"Attempted to start Docker Desktop: {path}")
                return
            except Exception as e:
                logger.warning(f"Failed to start Docker Desktop at {path}: {e}")
    logger.warning("Docker Desktop executable not found in default locations. Please start it manually.")


