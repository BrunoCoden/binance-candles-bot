#!/usr/bin/env python3
"""
Script de prueba para enviar una orden a dYdX con el notional mínimo configurado.
"""
import os
import sys
from pathlib import Path

# Asegura imports relativos al repo
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from trading.accounts.manager import AccountManager
from trading.orders.executor import OrderExecutor
from trading.orders.models import OrderRequest, OrderSide, OrderType, TimeInForce

# Cargar variables de entorno
load_dotenv()

def test_dydx_order(dry_run=True, notional_override=None):
    """Prueba enviar una orden a dYdX."""
    print("=" * 60)
    print("Test de Orden dYdX - Usuario Diego")
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
    
    # 2. Obtener cuenta y credenciales
    try:
        account = manager.get_account("diego")
        cred = account.get_exchange("dydx")
        print(f"[OK] Cuenta Diego - Exchange dYdX")
        print(f"    - Symbol: {cred.extra.get('symbol', 'ETH-USD')}")
        print(f"    - Subaccount: {cred.extra.get('subaccount', 0)}")
        print(f"    - Notional USDC configurado: {cred.notional_usdc}")
        print(f"    - Max Position USDC: {cred.max_position_usdc}")
    except Exception as exc:
        print(f"[ERROR] No se pudo obtener cuenta/credenciales: {exc}")
        return 1
    
    # 3. Verificar credenciales
    api_key = os.getenv("DIEGO_DYDX_API_KEY_LIVE")
    api_secret = os.getenv("DIEGO_DYDX_API_SECRET_LIVE")
    
    if not api_key or not api_secret:
        print("[ERROR] Variables DIEGO_DYDX_API_KEY_LIVE o DIEGO_DYDX_API_SECRET_LIVE no están definidas")
        return 1
    
    print(f"[OK] Credenciales encontradas")
    print(f"    - API Key: {api_key[:10]}...{api_key[-10:]}")
    
    # 4. Obtener precio actual del mercado (usando indexer)
    try:
        import requests
        market_symbol = cred.extra.get("symbol", "ETH-USD")
        market_symbol_upper = market_symbol.upper()
        
        # Intentar obtener precio desde el indexer
        indexer_base = os.getenv("DYDX_INDEXER_BASE", "https://indexer.dydx.trade/v4")
        markets_url = f"{indexer_base}/perpetualMarkets"
        
        r = requests.get(markets_url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # El indexer puede devolver en diferentes formatos
        markets = data.get("markets") or data.get("perpetualMarkets") or {}
        market_info = markets.get(market_symbol_upper)
        
        if not market_info:
            print(f"[WARN] Mercado {market_symbol_upper} no encontrado en indexer, intentando obtener precio de otra forma...")
            # Intentar obtener desde el endpoint de ticker
            ticker_url = f"{indexer_base}/tickers/{market_symbol_upper}"
            try:
                r2 = requests.get(ticker_url, timeout=10)
                r2.raise_for_status()
                ticker_data = r2.json()
                market_info = ticker_data.get("ticker") or ticker_data
            except:
                pass
        
        if not market_info:
            print(f"[ERROR] No se pudo obtener información del mercado {market_symbol_upper}")
            print(f"[INFO] Usando precio estimado de $3000 para ETH-USD (ajustar manualmente si es necesario)")
            current_price = 3000.0
        else:
            # Obtener precio actual (probar diferentes campos)
            current_price = 0.0
            for price_field in ["indexPrice", "markPrice", "oraclePrice", "price", "lastPrice"]:
                price_val = market_info.get(price_field)
                if price_val:
                    try:
                        current_price = float(price_val)
                        break
                    except:
                        continue
            
            if current_price == 0:
                print(f"[WARN] No se pudo extraer precio del mercado, usando estimado $3000")
                current_price = 3000.0
        
        print(f"[OK] Precio actual de {market_symbol}: ${current_price:.2f}")
    except Exception as exc:
        print(f"[WARN] Error obteniendo precio del indexer: {exc}")
        print(f"[INFO] Usando precio estimado de $3000 para ETH-USD")
        current_price = 3000.0
    
    # 5. Calcular cantidad basada en notional
    notional = notional_override if notional_override else cred.notional_usdc
    if not notional:
        print("[ERROR] No hay notional configurado y no se proporcionó uno")
        return 1
    
    # Usar un notional mínimo de prueba (más pequeño que el configurado)
    test_notional = min(notional, 10.0)  # Máximo 10 USDC para prueba
    quantity = test_notional / current_price
    
    print(f"[INFO] Notional de prueba: ${test_notional:.2f} USDC")
    print(f"[INFO] Cantidad calculada: {quantity:.6f} {market_symbol.split('-')[0]}")
    
    # 6. Crear OrderExecutor
    try:
        executor = OrderExecutor(manager)
        print("[OK] OrderExecutor creado")
    except Exception as exc:
        print(f"[ERROR] Error creando OrderExecutor: {exc}")
        return 1
    
    # 7. Crear orden de prueba (BUY)
    order = OrderRequest(
        symbol=market_symbol,
        side=OrderSide.BUY,
        type=OrderType.MARKET,  # MARKET para ejecución inmediata
        quantity=quantity,
        price=None,  # MARKET no requiere precio
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
    )
    
    print(f"\n[INFO] Preparando orden:")
    print(f"    - Tipo: {'DRY-RUN' if dry_run else 'LIVE'}")
    print(f"    - Symbol: {order.symbol}")
    print(f"    - Side: {order.side.value}")
    print(f"    - Type: {order.type.value}")
    print(f"    - Quantity: {order.quantity:.6f}")
    
    # 8. Enviar orden
    try:
        print(f"\n[INFO] Enviando orden...")
        response = executor.execute("diego", "dydx", order, dry_run=dry_run)
        
        print(f"\n[RESULTADO]")
        print(f"    - Success: {response.success}")
        print(f"    - Status: {response.status}")
        if response.error:
            print(f"    - Error: {response.error}")
        if response.raw:
            print(f"    - Raw response: {response.raw}")
        
        if response.success:
            print(f"\n[SUCCESS] Orden {'simulada' if dry_run else 'enviada'} correctamente")
            if not dry_run:
                print(f"    - Exchange Order ID: {response.exchange_order_id}")
                print(f"    - Filled Quantity: {response.filled_quantity}")
                print(f"    - Avg Price: {response.avg_price}")
            return 0
        else:
            print(f"\n[FAILED] La orden falló")
            return 1
            
    except Exception as exc:
        print(f"\n[ERROR] Excepción al enviar orden: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prueba envío de orden a dYdX")
    parser.add_argument("--live", action="store_true", help="Enviar orden real (no dry-run)")
    parser.add_argument("--notional", type=float, help="Notional personalizado en USDC")
    
    args = parser.parse_args()
    
    dry_run = not args.live
    if not dry_run:
        print("\n⚠️  ADVERTENCIA: Se enviará una orden REAL a dYdX")
        confirm = input("¿Continuar? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelado")
            sys.exit(0)
    
    sys.exit(test_dydx_order(dry_run=dry_run, notional_override=args.notional))

