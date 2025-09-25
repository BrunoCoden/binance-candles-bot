<# tg_check.ps1
    Uso:
      # Opción 1: lee TOKEN/CHAT_ID de variables de entorno
      .\tg_check.ps1

      # Opción 2: pasar parámetros explícitos
      .\tg_check.ps1 -Token "123456789:AAabc..." -ChatId "123456789" -Text "Ping ✅"

    Tip:
      En Telegram abrí tu bot y mandale /start antes de correr esto.
#>

param(
  [string]$Token  = $env:TELEGRAM_BOT_TOKEN,
  [string]$ChatId = $env:TELEGRAM_CHAT_ID,
  [string]$Text   = "Ping ✅ desde PowerShell"
)

function Fail($msg){ Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }
function Info($msg){ Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Warn($msg){ Write-Host "[WARN]  $msg" -ForegroundColor Yellow }

# 1) Validar token
if (-not $Token) { Fail "Token vacío. Seteá TELEGRAM_BOT_TOKEN o pasá -Token." }
$Token = $Token.Trim(@(' ', '"'))
if ($Token -notmatch '^\d+:[A-Za-z0-9_\-]{20,}$') {
  Warn "El token no matchea el formato típico. Igual probamos..."
}

# 2) Llamar getUpdates
$updatesUrl = "https://api.telegram.org/bot$Token/getUpdates"
Info "Consultando: $updatesUrl"
try {
  $resp = Invoke-RestMethod -Method Get -Uri $updatesUrl -ErrorAction Stop
} catch {
  # Mensajes útiles para 401/404 y otras perlas
  if ($_.Exception.Response -and $_.Exception.Response.StatusCode.Value__ -eq 401) {
    Fail "401 Unauthorized. Token inválido o revocado."
  } elseif ($_.Exception.Response -and $_.Exception.Response.StatusCode.Value__ -eq 404) {
    Fail "404 Not Found. URL mal formada o token vacío. Verificá que estés usando: https://api.telegram.org/botTOKEN/getUpdates"
  } else {
    Fail ("Fallo consultando getUpdates: " + $_.Exception.Message)
  }
}

if (-not $resp.ok) { Fail ("Telegram respondió ok=false: " + ($resp | ConvertTo-Json -Depth 6)) }

# 3) Extraer chats
$rows = @()

foreach ($it in $resp.result) {
  $c = $null
  if ($it.message)         { $c = $it.message.chat }
  elseif ($it.edited_message){ $c = $it.edited_message.chat }
  elseif ($it.channel_post){ $c = $it.channel_post.chat }
  elseif ($it.edited_channel_post){ $c = $it.edited_channel_post.chat }
  elseif ($it.my_chat_member){ $c = $it.my_chat_member.chat }
  elseif ($it.chat_member){ $c = $it.chat_member.chat }

  if ($c) {
    $rows += [pscustomobject]@{
      id         = $c.id
      type       = $c.type
      title      = $c.title
      username   = $c.username
      first_name = $c.first_name
      last_name  = $c.last_name
    }
  }
}

# Dejar únicos por id
$unique = $rows | Group-Object id | ForEach-Object { $_.Group | Select-Object -First 1 }

if ($unique.Count -eq 0) {
  Warn "No hay chats en getUpdates. Mandale /start al bot y escribí algo, luego corré el script de nuevo."
} else {
  Info "Chats detectados:"
  $unique | Sort-Object type,id | Format-Table -Auto
}

# 4) Si pasaste ChatId o hay uno solo, mandar test
if (-not $ChatId -and $unique.Count -eq 1) {
  $ChatId = $unique[0].id
  Info "Usando chat_id único detectado: $ChatId"
}

if ($ChatId) {
  $ChatId = "$ChatId".Trim()
  $sendUrl = "https://api.telegram.org/bot$Token/sendMessage"
  $body = @{ chat_id = $ChatId; text = $Text; parse_mode = "HTML"; disable_web_page_preview = $true } | ConvertTo-Json
  Info "Enviando mensaje de prueba al chat_id $ChatId ..."
  try {
    $sendResp = Invoke-RestMethod -Method Post -Uri $sendUrl -ContentType "application/json" -Body $body -ErrorAction Stop
    if ($sendResp.ok) {
      Write-Host "[OK] Mensaje enviado." -ForegroundColor Green
    } else {
      Warn ("Telegram respondió ok=false: " + ($sendResp | ConvertTo-Json -Depth 6))
    }
  } catch {
    if ($_.Exception.Response -and $_.Exception.Response.StatusCode.Value__ -eq 400) {
      Warn "400 Bad Request. Suele ser chat_id incorrecto o el bot no tiene permiso (en grupo/canal)."
    } elseif ($_.Exception.Response -and $_.Exception.Response.StatusCode.Value__ -eq 401) {
      Fail "401 Unauthorized. Token inválido."
    } else {
      Fail ("Fallo sendMessage: " + $_.Exception.Message)
    }
  }
} else {
  Warn "No se envió test porque no se definió ChatId. Elegí uno de la tabla y corré:  .\tg_check.ps1 -ChatId <ID>"
}

Write-Host "`nTips:" -ForegroundColor DarkCyan
Write-Host " - Chat privado: id positivo (ej: 123456789). Escribile /start al bot primero."
Write-Host " - Grupo: id negativo (ej: -1001234567890). Agregá el bot y quizás desactivá privacy en @BotFather (/setprivacy -> Disable)."
Write-Host " - Canal: el bot debe ser admin. Usá el @username del canal o el id negativo grande."
