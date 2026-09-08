## Qué cambia

<!-- Una o dos frases. Qué hace este PR, no cómo. -->

## Por qué

<!-- El problema que resuelve. Enlaza el issue si existe: Closes #N -->

## Cómo probarlo

<!-- Pasos concretos para verificar el cambio. Comando exacto si aplica. -->

```bash
uv run pytest
```

## Impacto en resultados

<!-- Solo si toca modelos, datos o métricas. Si no, borra esta sección.
     Incluye las métricas antes/después: PR-AUC, F1, VPP. Nunca accuracy sola. -->

| Métrica | Antes | Después |
|---|---|---|
| PR-AUC | | |
| F1 | | |
| VPP @ prev. 2 % | | |

## Checklist

- [ ] `uv run ruff check .` y `uv run ruff format --check .` pasan
- [ ] `uv run pytest` pasa
- [ ] No hay credenciales, datos ni checkpoints en el diff
- [ ] Los cambios de configuración están documentados
- [ ] El README está actualizado si cambia el uso
