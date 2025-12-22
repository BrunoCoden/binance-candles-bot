# Checklist: Configuración dYdX para Diego

## ✅ Estado Actual

### Configuración en `oci_accounts.yaml`:
- ✅ Usuario Diego configurado
- ✅ Exchange dYdX configurado
- ✅ Symbol: ETH-USD
- ✅ Subaccount: 0
- ✅ Notional USDC: 50.0
- ✅ Max Position USDC: 400.0
- ✅ Margin Mode: isolated

### Variables de Entorno en OCI:
- ✅ `DIEGO_DYDX_API_KEY_LIVE=dydx1wuwftf648zcky2jjzwzyj98x7s52kxycj0uq5z`
- ✅ `DIEGO_DYDX_API_SECRET_LIVE=0xa27af528e1153c14fd051f5900f96304407a1618814672de8c1a93a856a5f6d9`
- ✅ `DYDX_GRPC_HOST=dydx-dao-grpc-1.polkachu.com:443`

## ⚠️ Problema Detectado

Hay **dos implementaciones diferentes** de dYdX en el código:

1. **`trading/exchanges/dydx.py`** (OrderExecutor - usado para nuevas órdenes):
   - Espera: **wallet address** (dydx1...) + **private key** (0x...)
   - ✅ **Tus credenciales actuales son de este formato**

2. **`watcher_alertas.py`** (funciones legacy de posición/cierre):
   - Espera: **API key tradicional** + **API secret tradicional**
   - ❌ **Tus credenciales NO son de este formato**

### Impacto:
- ✅ Las **nuevas órdenes** funcionarán correctamente (OrderExecutor)
- ❌ Las **consultas de posición y cierres** en `watcher_alertas.py` pueden fallar

## 🔧 Soluciones

### Opción 1: Migrar watcher_alertas.py (Recomendado)
Modificar `watcher_alertas.py` para usar `OrderExecutor` también para consultas de posición y cierres.

**Ventajas:**
- Una sola fuente de credenciales
- Implementación más moderna
- Consistencia en todo el sistema

**Archivos a modificar:**
- `watcher_alertas.py` líneas 253-273 (`_current_position`)
- `watcher_alertas.py` líneas 307-329 (`_close_position`)
- `watcher_alertas.py` líneas 496-517 (`_close_opposite_position`)

### Opción 2: Mantener ambas implementaciones
Crear variables de entorno adicionales para el watcher legacy:
- `DIEGO_DYDX_API_KEY_LEGACY_LIVE`: API key tradicional
- `DIEGO_DYDX_API_SECRET_LEGACY_LIVE`: API secret tradicional

Y modificar `watcher_alertas.py` para usar estas variables.

## 📋 Checklist de Verificación

### 1. Credenciales Configuradas
- [x] Wallet address configurada (`DIEGO_DYDX_API_KEY_LIVE`)
- [x] Private key configurada (`DIEGO_DYDX_API_SECRET_LIVE`)
- [x] DYDX_GRPC_HOST configurado

### 2. Validar Configuración
```bash
ssh oci-bot "cd ~/bot && source .venv/bin/activate && python scripts/validate_accounts.py --accounts trading/accounts/oci_accounts.yaml --verbose"
```

### 3. Probar OrderExecutor (Dry-Run)
```python
from trading.accounts.manager import AccountManager
from trading.orders.executor import OrderExecutor
from trading.orders.models import OrderRequest, OrderSide, OrderType

manager = AccountManager.from_file("trading/accounts/oci_accounts.yaml")
executor = OrderExecutor(manager)

order = OrderRequest(
    symbol="ETH-USD",
    side=OrderSide.BUY,
    type=OrderType.MARKET,
    quantity=0.01,
)

response = executor.execute("diego", "dydx", order, dry_run=True)
print(response.success, response.status)
```

### 4. Verificar Wallet en dYdX
- [ ] Wallet tiene fondos USDC
- [ ] Wallet está conectada a mainnet
- [ ] Subaccount 0 está activo
- [ ] Permisos de trading habilitados

### 5. Configurar Trading en Watcher
Para habilitar trading real:
```bash
# En OCI, editar /etc/systemd/system/bot.env o .env
WATCHER_ENABLE_TRADING=true
WATCHER_TRADING_USER=diego
WATCHER_TRADING_EXCHANGE=dydx
WATCHER_TRADING_DRY_RUN=false  # Cambiar a false para trading real
WATCHER_TRADING_DEFAULT_QTY=0.01  # O usar notional
```

### 6. Probar Funciones Legacy (si no se migra)
Si decides mantener las funciones legacy en `watcher_alertas.py`, necesitarás:
- [ ] Crear API keys tradicionales en dYdX
- [ ] Configurar `DIEGO_DYDX_API_KEY_LEGACY_LIVE`
- [ ] Configurar `DIEGO_DYDX_API_SECRET_LEGACY_LIVE`
- [ ] Modificar `watcher_alertas.py` para usar estas variables

## 📝 Notas Importantes

1. **Formato de Credenciales:**
   - Wallet address: `dydx1...` (tu valor actual es correcto)
   - Private key: `0x...` o hex sin prefijo (tu valor actual es correcto)

2. **Símbolo:**
   - Configurado como `ETH-USD` en `oci_accounts.yaml`
   - Asegúrate que este mercado exista en dYdX mainnet

3. **Subaccount:**
   - Configurado como `0` (subaccount principal)
   - Si usas otro subaccount, actualizar en `oci_accounts.yaml`

4. **Notional y Posición:**
   - Notional por trade: 50 USDC
   - Máximo posición: 400 USDC
   - Ajustar según tu estrategia de riesgo

5. **Margin Mode:**
   - Configurado como `isolated`
   - Asegúrate que dYdX soporte isolated margin para ETH-USD

## 🚀 Próximos Pasos

1. **Decidir estrategia:**
   - [ ] Opción 1: Migrar `watcher_alertas.py` (recomendado)
   - [ ] Opción 2: Mantener ambas implementaciones

2. **Si eliges Opción 1:**
   - Modificar `watcher_alertas.py` para usar `OrderExecutor`
   - Probar en dry-run
   - Verificar que consultas de posición funcionen

3. **Si eliges Opción 2:**
   - Crear API keys tradicionales en dYdX
   - Configurar variables legacy
   - Modificar `watcher_alertas.py` para usar variables alternativas

4. **Habilitar trading:**
   - Verificar fondos en wallet
   - Configurar variables de entorno de trading
   - Probar con dry-run primero
   - Monitorear logs en OCI

## 📚 Referencias

- Documentación completa: `docs/DYDX_SETUP_DIEGO.md`
- Script de prueba: `scripts/test_dydx_diego.py`
- Configuración de cuentas: `trading/accounts/oci_accounts.yaml`



