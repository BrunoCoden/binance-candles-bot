# Configuración dYdX para Usuario Diego

## Resumen

El sistema tiene **dos implementaciones diferentes** de dYdX que requieren credenciales distintas:

1. **`watcher_alertas.py`**: Usa `dydx_v4_client.client.Client` (cliente legacy)
2. **`trading/exchanges/dydx.py`**: Usa `dydx_v4_client.node.client.NodeClient` (cliente v4 nativo)

## Configuración Actual en `oci_accounts.yaml`

```yaml
- id: diego
  exchanges:
    dydx:
      exchange: dydx
      api_key_env: DIEGO_DYDX_API_KEY_LIVE
      api_secret_env: DIEGO_DYDX_API_SECRET_LIVE
      environment: live
      extra:
        symbol: ETH-USD
        subaccount: 0
      notional_usdc: 50.0
      max_position_usdc: 400.0
      margin_mode: isolated
```

## Variables de Entorno Requeridas

### Para `trading/exchanges/dydx.py` (OrderExecutor - usado por watcher para nuevas órdenes):

**Variables obligatorias:**
- `DIEGO_DYDX_API_KEY_LIVE`: **Wallet address de dYdX** (formato: `dydx1...`)
- `DIEGO_DYDX_API_SECRET_LIVE`: **Private key de la wallet** (formato: `0x...` o hex sin prefijo)

**Variables opcionales:**
- `DYDX_GRPC_HOST`: Endpoint gRPC (default: `dydx-dao-grpc-1.polkachu.com:443`)
- `DYDX_INDEXER_URL`: URL del indexer (default: `https://indexer.dydx.trade/v4/perpetualMarkets`)

### Para `watcher_alertas.py` (funciones de posición/cierre):

**Variables obligatorias:**
- `DIEGO_DYDX_API_KEY_LIVE`: **API Key de dYdX** (formato tradicional de API key)
- `DIEGO_DYDX_API_SECRET_LIVE`: **API Secret de dYdX** (formato tradicional de API secret)

**Variables opcionales (si están en oci_accounts.yaml):**
- `DIEGO_DYDX_PASSPHRASE_LIVE`: Passphrase de dYdX (si aplica)
- `DIEGO_DYDX_STARK_KEY_LIVE`: Stark private key (si aplica)

## ⚠️ PROBLEMA DETECTADO

Hay una **inconsistencia** entre las dos implementaciones:

- `trading/exchanges/dydx.py` espera **wallet address + private key**
- `watcher_alertas.py` espera **API key + API secret tradicionales**

Esto significa que actualmente **NO pueden funcionar ambas al mismo tiempo** con las mismas credenciales.

## Solución Recomendada

### Opción 1: Usar solo OrderExecutor (recomendado)

Modificar `watcher_alertas.py` para que use `OrderExecutor` también para consultas de posición y cierres, eliminando el uso directo de `dydx_v4_client.client.Client`.

**Ventajas:**
- Una sola fuente de credenciales (wallet address + private key)
- Implementación más moderna (v4 nativo)
- Consistencia en todo el sistema

### Opción 2: Mantener ambas implementaciones

Configurar ambas variables con diferentes valores:
- `DIEGO_DYDX_API_KEY_LIVE`: Wallet address (para OrderExecutor)
- `DIEGO_DYDX_API_SECRET_LIVE`: Private key (para OrderExecutor)
- Crear nuevas variables para el watcher legacy:
  - `DIEGO_DYDX_API_KEY_LEGACY_LIVE`: API key tradicional
  - `DIEGO_DYDX_API_SECRET_LEGACY_LIVE`: API secret tradicional

Y modificar `watcher_alertas.py` para usar estas variables alternativas.

## Cómo Obtener las Credenciales

### Para Wallet Address + Private Key (OrderExecutor):

1. **Crear wallet en dYdX v4:**
   - Acceder a dYdX y crear una nueva wallet
   - Se generará una dirección tipo `dydx1...` (esta es la wallet address)
   - Se generará una frase secreta de 24 palabras o una private key en formato hex

2. **Obtener private key:**
   - Si tienes frase secreta: derivar la private key usando herramientas de dYdX
   - Si tienes la private key directamente: usar formato `0x...` o hex sin prefijo

3. **Configurar en variables de entorno:**
   ```bash
   export DIEGO_DYDX_API_KEY_LIVE="dydx1wuwftf648zcky2jjzwzyj98x7s52kxycj0uq5z"
   export DIEGO_DYDX_API_SECRET_LIVE="0xa27af528e1153c14fd051f5900f96304407a1618814672de8c1a93a856a5f6d9"
   ```

### Para API Key + API Secret (watcher legacy):

1. **Crear API keys en dYdX:**
   - Acceder a dYdX → Settings → API
   - Crear nueva API key
   - Guardar API key y API secret

2. **Configurar permisos:**
   - Habilitar permisos de trading
   - Configurar restricciones de IP si es necesario

3. **Configurar en variables de entorno:**
   ```bash
   export DIEGO_DYDX_API_KEY_LIVE="tu_api_key_aqui"
   export DIEGO_DYDX_API_SECRET_LIVE="tu_api_secret_aqui"
   ```

## Verificación

### Validar credenciales básicas:
```bash
python scripts/validate_accounts.py --accounts trading/accounts/oci_accounts.yaml --verbose
```

### Probar conexión con OrderExecutor:
```python
from trading.accounts.manager import AccountManager
from trading.orders.executor import OrderExecutor
from trading.orders.models import OrderRequest, OrderSide, OrderType

manager = AccountManager.from_file("trading/accounts/oci_accounts.yaml")
executor = OrderExecutor(manager)

# Test dry-run
order = OrderRequest(
    symbol="ETH-USD",
    side=OrderSide.BUY,
    type=OrderType.MARKET,
    quantity=0.01,
)

response = executor.execute("diego", "dydx", order, dry_run=True)
print(response)
```

## Estado Actual

Según el `.env` (que no está en git), ya tienes configurado:
- `DIEGO_DYDX_API_KEY_LIVE=dydx1wuwftf648zcky2jjzwzyj98x7s52kxycj0uq5z` ✅
- `DIEGO_DYDX_API_SECRET_LIVE=0xa27af528e1153c14fd051f5900f96304407a1618814672de8c1a93a856a5f6d9` ✅
- `DYDX_GRPC_HOST=dydx-dao-grpc-1.polkachu.com:443` ✅

**Estas credenciales son del formato wallet address + private key**, lo que significa:
- ✅ Funcionará con `OrderExecutor` (trading/exchanges/dydx.py)
- ❌ NO funcionará con las funciones legacy en `watcher_alertas.py`

## Próximos Pasos

1. **Decidir qué implementación usar:**
   - Recomendado: Migrar `watcher_alertas.py` para usar solo `OrderExecutor`
   - Alternativa: Mantener ambas y configurar credenciales separadas

2. **Verificar que las credenciales estén en OCI:**
   ```bash
   ssh oci-bot "cd ~/bot && python scripts/validate_accounts.py --accounts trading/accounts/oci_accounts.yaml --verbose"
   ```

3. **Probar una orden dry-run desde OCI:**
   ```bash
   ssh oci-bot "cd ~/bot && source .venv/bin/activate && python -c '...'"
   ```



