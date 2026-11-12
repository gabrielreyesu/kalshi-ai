"""Sube o ACTUALIZA este proyecto en tu repositorio de GitHub desde Colab.
Si el repositorio ya tiene contenido, conserva su historial y su carpeta predictions/ (el registro diario).
Ejecuta en Colab:   %run kalshi_ai/push_to_github.py      (pide usuario, repositorio y token; el token no se guarda)
"""
import getpass, os, shutil, subprocess, tempfile

usuario = input("Tu usuario de GitHub: ").strip()
repo = input("Nombre del repositorio (ej. kalshi-ai): ").strip()
token = getpass.getpass("Token de GitHub (no se muestra): ").strip()
url = f"https://x-access-token:{token}@github.com/{usuario}/{repo}.git"
aqui = os.path.dirname(os.path.abspath(__file__))
dest = os.path.join(tempfile.mkdtemp(), "repo")


def run(*cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd or dest, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(token, "***")
    if check and r.returncode != 0:
        raise SystemExit("Falló: " + " ".join("***" if token in c else c for c in cmd) + "\n" + out)
    return r.returncode, out


code, _ = run("git", "clone", "-q", url, dest, cwd=aqui, check=False)
existente = code == 0 and os.path.isdir(os.path.join(dest, ".git")) and run("git", "rev-parse", "HEAD", check=False)[0] == 0
if not existente:
    shutil.rmtree(dest, ignore_errors=True); os.makedirs(dest)
    run("git", "init", "-b", "main"); run("git", "remote", "add", "origin", url)
omitir = {"__pycache__", ".git", "data", "reports"} | ({"predictions"} if existente else set())
for nombre in os.listdir(aqui):
    if nombre in omitir:
        continue
    o, d = os.path.join(aqui, nombre), os.path.join(dest, nombre)
    if os.path.isdir(o):
        shutil.copytree(o, d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(o, d)
run("git", "config", "user.name", usuario); run("git", "config", "user.email", f"{usuario}@users.noreply.github.com")
run("git", "add", "-A")
if run("git", "diff", "--cached", "--quiet", check=False)[0] == 0:
    print("No hay cambios que subir.")
else:
    run("git", "commit", "-m", "Actualiza Kalshi AI" if existente else "Kalshi AI: paquete inicial")
    run("git", "push", "-u", "origin", "main")
    print(f"Listo ({'actualizado' if existente else 'creado'}): https://github.com/{usuario}/{repo}")
