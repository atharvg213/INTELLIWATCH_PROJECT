# Deployment a Cloudflare Pages (desde GitHub)

Este proyecto está configurado para desplegarse automáticamente a Cloudflare Pages cada vez que se hace push a la rama `master`.

## Requisitos

- Tener una cuenta en Cloudflare
- El repositorio conectado a Cloudflare Pages (o usar GitHub Actions con Wrangler)

## Opción Recomendada: Conectar el repositorio directamente en Cloudflare Pages

Esta es la forma más simple y recomendada.

### Pasos:

1. Ve a [Cloudflare Dashboard → Pages](https://dash.cloudflare.com/?to=/:account/pages)

2. Haz clic en **Create a project** → **Connect to Git**

3. Selecciona el repositorio `aether-2032`

4. Configura los siguientes valores:

   | Campo                    | Valor                          |
   |--------------------------|--------------------------------|
   | Production branch        | `master`                       |
   | Framework preset         | `Next.js`                      |
   | Build command            | `npm run build`                |
   | Build output directory   | `out`                          |

5. Haz clic en **Save and Deploy**

Una vez creado el proyecto, Cloudflare detectará automáticamente los cambios en GitHub y desplegará en cada push.

---

## Opción Avanzada: GitHub Actions + Wrangler

Si prefieres usar GitHub Actions (más control), sigue estos pasos:

### 1. Crear API Token en Cloudflare

1. Ve a: https://dash.cloudflare.com/profile/api-tokens
2. Clic en **Create Token** → **Custom token**
3. Nombre: `AETHER Deploy`
4. Permissions:
   - Pages → Edit
5. (Opcional) Scope the token a tu cuenta
6. Crea el token y cópialo

### 2. Agregar Secrets en GitHub

Ve a: https://github.com/sekirosevillans-sys/aether-2032/settings/secrets/actions

Agrega los siguientes secretos:

- `CLOUDFLARE_API_TOKEN` → (el token que creaste)
- `CLOUDFLARE_ACCOUNT_ID` → (lo encuentras en el dashboard de Cloudflare, abajo a la derecha)

### 3. Crear el proyecto en Cloudflare Pages (recomendado)

Aunque uses GitHub Actions, es buena idea crear primero el proyecto:

- Nombre del proyecto: `aether-2032`
- Conectar a GitHub (opcional)

### 4. Deploy manual (una sola vez)

```bash
cd web
npm run build
npx wrangler pages deploy out --project-name=aether-2032
```

Después de esto, los deploys automáticos se encargarán vía GitHub Actions.

---

## Notas

- El sitio está configurado como **Static Export** (`output: 'export'`).
- La carpeta de salida es `out/`.
- Cada push a `master` dispara el workflow y despliega automáticamente.

---

**URL del sitio (después del primer deploy):**

https://aether-2032.pages.dev

---

Cualquier problema, revisa los logs del workflow en GitHub Actions.