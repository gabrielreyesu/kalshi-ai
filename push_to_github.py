"""Sube este proyecto a TU repositorio de GitHub desde Colab (sin usar la terminal).

Antes:
  1) En github.com crea un repositorio VACÍO (ej. kalshi-ai), privado está bien.
  2) Crea un token: Settings > Developer settings > Fine-grained tokens, con acceso SOLO a ese
     repositorio y permisos: Contents = Read and write, Workflows = Read and write.
Después ejecuta en Colab:   %run push_to_github.py
El token se pide con getpass (no queda escrito en ningún archivo ni en el notebook).
"""
import getpass, subprocess, shutil, tempfile, os

usuario = input("Tu usuario de GitHub: ").strip()
repo = input("Nombre del repositorio (ej. kalshi-ai): ").strip()
token = getpass.getpass("Token de GitHub (no se muestra): ").strip()

aqui = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
dest = os.path.join(tmp, "repo")
shutil.copytree(aqui, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "data", "reports"))

def run(*cmd):
    r = subprocess.run(cmd, cwd=dest, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(token, "***")
    if r.returncode != 0:
        raise SystemExit("Falló: " + " ".join(c if token not in c else "***" for c in cmd) + "\n" + out)
    return out

run("git", "init", "-b", "main")
run("git", "config", "user.name", usuario)
run("git", "config", "user.email", f"{usuario}@users.noreply.github.com")
run("git", "add", "-A")
run("git", "commit", "-m", "Kalshi Soccer: paquete inicial")
run("git", "remote", "add", "origin", f"https://x-access-token:{token}@github.com/{usuario}/{repo}.git")
run("git", "push", "-u", "origin", "main")
print(f"Listo: https://github.com/{usuario}/{repo}")
print("Siguiente: en el repo -> Settings -> Secrets and variables -> Actions -> agrega FOOTBALL_DATA_API_KEY,")
print("luego pestaña Actions -> 'soccer-backtest' -> Run workflow.")
