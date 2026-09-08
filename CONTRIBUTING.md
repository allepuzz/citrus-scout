# Guía de desarrollo

## Flujo de trabajo

`main` está protegida: no se commitea directamente. Todo cambio entra por Pull Request.

```bash
# 1. Partir siempre de main actualizada
git checkout main && git pull

# 2. Rama nueva
git checkout -b feat/descripcion-corta

# 3. Trabajar y commitear
git add -p                  # revisa lo que subes, trozo a trozo
git commit -m "feat: añade cargador del dataset de hoja"

# 4. Subir y abrir PR
git push -u origin feat/descripcion-corta
gh pr create --fill
```

### Nombres de rama

| Prefijo | Uso |
|---|---|
| `feat/` | Funcionalidad nueva |
| `fix/` | Corrección de bug |
| `refactor/` | Cambio interno sin alterar comportamiento |
| `exp/` | Experimento de modelo o datos |
| `docs/` | Solo documentación |
| `chore/` | Tooling, dependencias, CI |

## Commits

Se usa [Conventional Commits](https://www.conventionalcommits.org/):

```
<tipo>: <descripción en imperativo y minúscula>

[cuerpo opcional explicando el porqué]
```

Tipos: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `exp`.

```bash
# Bien
feat: añade métrica de VPP a prevalencia configurable
fix: corrige fuga de datos entre train y val al estratificar

# Mal
cambios
update
arreglado el bug
```

Escribe el **porqué** en el cuerpo, no el qué — el qué ya está en el diff.

## Antes de abrir el PR

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest
```

Los pre-commit hooks lo hacen automáticamente. Instálalos una vez:

```bash
uv run pre-commit install
```

## Reglas que no se saltan

**Nunca commitear:**
- Credenciales (`kaggle.json`, `.env`, tokens) — el `.gitignore` y los hooks lo bloquean,
  pero revisa el diff igualmente
- Datos o imágenes — van por DVC
- Checkpoints de modelos — van a W&B o almacenamiento aparte
- Salidas de notebooks — `nbstripout` las limpia sola

Si un secreto llega a `main`, **no basta con borrarlo en un commit posterior**: queda en el
historial. Hay que rotar la credencial inmediatamente y reescribir el historial.

## Experimentos

Los experimentos van en ramas `exp/`. Un experimento se considera reproducible si:

1. La configuración está en `configs/` y versionada
2. La semilla está fijada y registrada
3. La versión de los datos está anclada con DVC
4. Las métricas están en W&B

Un PR que cambia el modelo debe reportar **PR-AUC, F1 y VPP a prevalencia real**.
La *accuracy* sola no se acepta como evidencia: con prevalencia del 2 %, un modelo que
predice siempre "sano" tiene 98 % de accuracy y es inútil.

## Estructura del código

- `src/citrus_scout/` es un paquete instalable. Los imports son absolutos:
  `from citrus_scout.data import LeafDataset`
- La lógica va en el paquete, **no en los notebooks**. Los notebooks exploran y visualizan,
  pero el código que se reutiliza se mueve a `src/`.
- Todo entrenamiento se lanza por CLI con un archivo de configuración, nunca con constantes
  a mano en un script. Esto es lo que permite ejecutar lo mismo en local y en Colab.
