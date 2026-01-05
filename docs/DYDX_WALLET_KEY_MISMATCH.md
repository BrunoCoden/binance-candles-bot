# Error: pubKey does not match signer address

## Error Actual

Al intentar enviar una orden a dYdX, se recibió:

```
pubKey does not match signer address dydx1whkj5d959clyjvr0ctmge6wzche82895302k3n with signer index: 0: invalid pubkey
```

## Causa

La **private key** configurada en `DIEGO_DYDX_API_SECRET_LIVE` **no corresponde** a la **wallet address** configurada en `DIEGO_DYDX_API_KEY_LIVE`.

## Wallet Configurada

- **Wallet Address**: `dydx1whkj5d959clyjvr0ctmge6wzche82895302k3n` ✅ (existe en dYdX)
- **Private Key**: `0xa27af528e1153c14fd051f5900f96304407a1618814672de8c1a93a856a5f6d9` ❌ (no corresponde)

## Solución

Necesitas obtener la **private key correcta** que corresponde a la wallet `dydx1whkj5d959clyjvr0ctmge6wzche82895302k3n`.

### Pasos:

1. **Conectar la wallet en dYdX** usando la misma private key que quieres usar
2. **Verificar que la dirección coincida** con `dydx1whkj5d959clyjvr0ctmge6wzche82895302k3n`
3. **Actualizar `DIEGO_DYDX_API_SECRET_LIVE`** en `.env` con la private key correcta

### Formato de Private Key:

- Puede empezar con `0x` o no
- Debe ser el hex de la private key que deriva a la wallet address `dydx1whkj5d959clyjvr0ctmge6wzche82895302k3n`

## Progreso

✅ **Wallet existe**: La wallet `dydx1whkj5d959clyjvr0ctmge6wzche82895302k3n` está registrada en dYdX  
✅ **Código funciona**: El código puede obtener precio y preparar la orden  
✅ **Orden se envía**: La transacción se crea correctamente  
❌ **Firma inválida**: La private key no corresponde a la wallet address

Una vez que actualices la private key correcta, la orden debería ejecutarse exitosamente.



