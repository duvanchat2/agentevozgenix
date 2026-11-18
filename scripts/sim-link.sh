#!/usr/bin/env bash
# Imprime el enlace de la página de confirmación del cliente.
#
#   ./scripts/sim-link.sh
#
# La página vive en la raíz del subdominio y ya no pide token, así que este enlace
# se puede compartir con el cliente.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
. ./.env
set +a

[ -n "${DISPATCHER_DOMAIN:-}" ] || { echo "Falta DISPATCHER_DOMAIN en .env (la página necesita HTTPS)."; exit 1; }
echo "https://${DISPATCHER_DOMAIN}/"
