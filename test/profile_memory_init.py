import os
import time
from pathlib import Path

from agent_engine.memory.memory import Memory
from agent_engine.memory.embedder import Embedder
from agent_engine.utils.project_root import get_project_root


def time_block(label, func):
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    duration = end - start
    print(f"{label}: {duration:.3f}s")
    return result, duration


def main():
    print("=== Memory Initialization Performance Profiling ===")

    # Locate project root and default memory path
    project_root, t_root = time_block("Find project root", get_project_root)
    print(f"Project root: {project_root}")

    test_db_name = "memory_perf_test"

    # Clean previous db for cold start comparison
    memory_dir = Path(project_root) / ".memory"
    db_file = memory_dir / f"{test_db_name}.db"
    if db_file.exists():
        try:
            db_file.unlink()
            print(f"Removed existing DB for cold start: {db_file}")
        except Exception as e:
            print(f"Warning: failed to remove existing DB: {e}")

    # Measure Embedder construction and fit (likely the heaviest part)
    def build_embedder():
        emb = Embedder(model_name="all-MiniLM-L6-v2")
        emb.fit()
        return emb

    embedder, t_embedder_fit = time_block("Embedder fit (model load)", build_embedder)
    print(f"Embedder info: method={embedder.method}, fitted={embedder._fitted}, dim={embedder.get_vector_dimension()}")

    # Cold start Memory (includes embedder.fit internally and DB init)
    def build_memory_cold():
        return Memory(name=test_db_name, model_name="all-MiniLM-L6-v2", preload_embedder=True)

    mem_cold, t_mem_cold = time_block("Memory init (cold)", build_memory_cold)
    print(f"DB path: {mem_cold.db_path}")

    # Warm start Memory (should skip DB init due to in-process cache; still includes embedder.fit)
    def build_memory_warm():
        return Memory(name=test_db_name, model_name="all-MiniLM-L6-v2", preload_embedder=True)

    mem_warm, t_mem_warm = time_block("Memory init (warm, same process)", build_memory_warm)

    # Measure DB-only ensure step by triggering another instance after deleting embedder effect
    # Note: Embedder fit runs every init; this quantifies its cost vs DB ensure.
    print("Note: If 'Embedder fit' dominates, model loading is the bottleneck (first-run may download).")

    # Measure a simple add (embeds once) to estimate per-embed cost
    def add_sample():
        mem_warm.add("The quick brown fox jumps over the lazy dog.")
        return True

    _, t_add = time_block("Memory add (single embed)", add_sample)

    print("--- Summary ---")
    print(f"Find project root: {t_root:.3f}s")
    print(f"Embedder fit (model load): {t_embedder_fit:.3f}s")
    print(f"Memory init (cold): {t_mem_cold:.3f}s")
    print(f"Memory init (warm, same process): {t_mem_warm:.3f}s")
    print(f"Add single record (embed): {t_add:.3f}s")
    print("Tip: If 'Embedder fit' ~ 'Memory init' times, consider reusing a shared Embedder or lazy-init.")


if __name__ == "__main__":
    main()


