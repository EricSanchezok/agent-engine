"""
Quick smoke test for core.umls.UMLSClient.

Usage:
  run.bat test\test_umls_client.py

Requirements:
  - Environment variable UMLS_API_KEY present (can be provided via .env)
"""

import os
import sys
from pathlib import Path


def main() -> int:
    try:
        # Load .env from project root robustly (works even without python-dotenv)
        def _load_env_from_project_root() -> None:
            # Prefer the PROJECT_ROOT provided by run.bat; fallback to locating pyproject.toml
            root = os.getenv("PROJECT_ROOT", "").strip()
            if not root:
                here = Path(__file__).resolve()
                for parent in here.parents:
                    if (parent / "pyproject.toml").exists():
                        root = str(parent)
                        break
            if not root:
                return
            env_path = Path(root) / ".env"
            if not env_path.exists():
                # Try loading via python-dotenv if available (covers custom locations)
                try:
                    from dotenv import load_dotenv  # type: ignore
                    load_dotenv()
                except Exception:
                    return
                return
            try:
                with env_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass

        _load_env_from_project_root()

        from core.umls import UMLSClient, UMLSClientError

        api_key = os.getenv("UMLS_API_KEY", "").strip()
        if not api_key:
            print("ERROR: Missing UMLS_API_KEY in environment (you can set it in .env).")
            return 2

        client = UMLSClient(api_key=api_key)

        term = "sepsis"
        print(f"Searching term: {term}")

        # 1) Raw search (first page only), low-level JSON
        raw = client.search(term, root_source="SNOMEDCT_US", search_type="exact")
        items = raw.get("result", {}).get("results", [])
        print(f"search(): first-page count={len(items)}; sample=")
        if items:
            sample = items[0]
            print(f"  - ui={sample.get('ui')} name={sample.get('name')} rootSource={sample.get('rootSource')}")

        # 2) CUIs list convenience (aggregates all pages)
        cuis = client.search_cuis(term, root_source="SNOMEDCT_US", search_type="exact")
        print(f"search_cuis(): total CUIs={len(cuis)}")
        if not cuis:
            print("WARNING: No CUIs returned; exiting early.")
            return 0

        first_cui = cuis[0]
        print(f"Using first CUI: {first_cui}")

        # 3) CUI details (semantic types)
        detail = client.get_cui_details(first_cui)
        stype = detail.semantic_types[0]["name"] if detail.semantic_types else "N/A"
        print(f"get_cui_details(): name={detail.name}; first semantic type={stype}")

        # 4) Map CUI to source vocab codes (SNOMED + RxNorm)
        mapping = client.map_cui_to_source_codes(first_cui, sabs=["SNOMEDCT_US", "RXNORM"])
        print(f"map_cui_to_source_codes(): total codes={len(mapping.results)}; sample=")
        if mapping.results:
            r0 = mapping.results[0]
            print(f"  - code(ui)={r0.ui} name={r0.name} source={r0.root_source}")

        print("SUCCESS: UMLS client basic calls executed successfully.")
        return 0

    except UMLSClientError as e:  # type: ignore[name-defined]
        print(f"UMLSClientError: {e}")
        return 3
    except Exception as e:
        print(f"Unhandled error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


