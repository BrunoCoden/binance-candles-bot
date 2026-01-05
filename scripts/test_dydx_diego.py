#!/usr/bin/env python3
"""
Script de prueba para verificar la configuración de dYdX para el usuario Diego.
"""
import os
import sys
from pathlib import Path

# Asegura imports relativos al repo
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading.accounts.manager import AccountManager
from trading.orders.executor import OrderExecutor
from trading.orders.models import OrderRequest, OrderSide, OrderType, TimeInForce

def test_dydx_config():
    """Prueba la configuración de dYdX para Diego."""
    print("=" * 60)
    print("Test de Configuración dYdX - Usuario Diego")
    print("=" * 60)
    
    # 1. Cargar configuración
    accounts_file = Path("trading/accounts/oci_accounts.yaml")
    if not accounts_file.exists():
        print(f"[ERROR] No se encontró {accounts_file}")
        return 1
    
    try:
        manager = AccountManager.from_file(accounts_file)
        print(f"[OK] Configuración cargada desde {accounts_file}")
    except Exception as exc:
        print(f"[ERROR] No se pudo cargar configuración: {exc}")
        return 1
    
    # 2. Verificar cuenta Diego
    try:
        account = manager.get_account("diego")
        print(f"[OK] Cuenta Diego encontrada: {account.label}")
    except KeyError:
        print("[ERROR] No se encontró la cuenta 'diego'")
        return 1
    
    # 3. Verificar exchange dYdX
    try:
        cred = account.get_exchange("dydx")
        print(f"[OK] Exchange dYdX configurado para Diego")
        print(f"    - Environment: {cred.environment.value}")
        print(f"    - Symbol: {cred.extra.get('symbol', 'N/A')}")
        print(f"    - Subaccount: {cred.extra.get('subaccount', 'N/A')}")
        print(f"    - Notional USDC: {cred.notional_usdc}")
        print(f"    - Max Position USDC: {cred.max_position_usdc}")
        print(f"    - Margin Mode: {cred.margin_mode}")
    except KeyError:
        print("[ERROR] Diego no tiene configuración para dYdX")
        return 1
    
    # 4. Verificar variables de entorno
    print("\n[INFO] Verificando variables de entorno...")
    api_key_env = cred.api_key_env
    api_secret_env = cred.api_secret_env
    
    api_key = os.getenv(api_key_env)
    api_secret = os.getenv(api_secret_env)
    
    if not api_key:
        print(f"[ERROR] Variable {api_key_env} no está definida")
        return 1
    else:
        print(f"[OK] {api_key_env} = {api_key[:10]}...{api_key[-10:]}")
    
    if not api_secret:
        print(f"[ERROR] Variable {api_secret_env} no está definida")
        return 1
    else:
        masked_secret = api_secret[:6] + "..." + api_secret[-6:] if len(api_secret) > 12 else "***"
        print(f"[OK] {api_secret_env} = {masked_secret}")
    
    # 5. Verificar formato de credenciales
    print("\n[INFO] Verificando formato de credenciales...")
    if api_key.startswith("dydx1"):
        print(f"[OK] API Key tiene formato wallet address (dydx1...)")
    else:
        print(f"[WARN] API Key no tiene formato wallet address esperado (dydx1...)")
        print(f"       Esto puede indicar que es una API key tradicional (legacy)")
    
    if api_secret.startswith("0x"):
        print(f"[OK] API Secret tiene formato private key (0x...)")
    elif len(api_secret) == 64:
        print(f"[OK] API Secret tiene formato hex (64 caracteres)")
    else:
        print(f"[WARN] API Secret no tiene formato private key esperado")
        print(f"       Esto puede indicar que es un API secret tradicional (legacy)")
    
    # 6. Probar OrderExecutor (dry-run)
    print("\n[INFO] Probando OrderExecutor (dry-run)...")
    try:
        executor = OrderExecutor(manager)
        order = OrderRequest(
            symbol=cred.extra.get("symbol", "ETH-USD"),
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.01,
            time_in_force=TimeInForce.GTC,
        )
        
        response = executor.execute("diego", "dydx", order, dry_run=True)
        if response.success:
            print(f"[OK] OrderExecutor funciona correctamente (dry-run)")
            print(f"    - Status: {response.status}")
            print(f"    - Symbol: {order.symbol}")
            print(f"    - Side: {order.side.value}")
            print(f"    - Quantity: {order.quantity}")
        else:
            print(f"[ERROR] OrderExecutor falló: {response.error}")
            return 1
    except Exception as exc:
        print(f"[ERROR] Error probando OrderExecutor: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 7. Verificar variables opcionales
    print("\n[INFO] Verificando variables opcionales...")
    grpc_host = os.getenv("DYDX_GRPC_HOST")
    if grpc_host:
        print(f"[OK] DYDX_GRPC_HOST = {grpc_host}")
    else:
        print(f"[INFO] DYDX_GRPC_HOST no definida (usará default: dydx-dao-grpc-1.polkachu.com:443)")
    
    indexer_url = os.getenv("DYDX_INDEXER_URL")
    if indexer_url:
        print(f"[OK] DYDX_INDEXER_URL = {indexer_url}")
    else:
        print(f"[INFO] DYDX_INDEXER_URL no definida (usará default)")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Configuración de dYdX para Diego está completa")
    print("=" * 60)
    print("\nNotas:")
    print("- Las credenciales están configuradas correctamente")
    print("- OrderExecutor funciona en modo dry-run")
    print("- Para trading real, asegúrate de:")
    print("  1. Tener fondos en la wallet de dYdX")
    print("  2. Configurar WATCHER_ENABLE_TRADING=true")
    print("  3. Configurar WATCHER_TRADING_DRY_RUN=false")
    print("  4. Verificar que el símbolo ETH-USD esté disponible en dYdX")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_dydx_config())




