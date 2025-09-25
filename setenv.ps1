# Activa el entorno virtual
.\.venv\Scripts\activate

# Configuración del bot
$env:SYMBOL="ETHUSDT.P"
$env:INTERVAL="1m"
$env:LIMIT="1200"
$env:RB_MULTI="4.0"
$env:RB_LB="10"
$env:RB_FILTER_TREND="false"
$env:TZ="America/Argentina/Buenos_Aires"

# Telegram
$env:TG_ENABLE="1"
$env:TELEGRAM_BOT_TOKEN="8254936223:AAEaZontJ5Kkqfa0czHm_CqVQGBDtw8t3Mk"
$env:TELEGRAM_CHAT_ID="436048117"

# Comportamiento
$env:SHOW="1"              # dibuja UNA VEZ al iniciar
$env:SLEEP_SEC="10"        # iteración cada 10s
$env:WINDOW_BARS="400"
$env:WARMUP_BARS="500"
$env:STREAM_CSV="stream_table.csv"
$env:REFRESH_FLAG="refresh.flag"   # archivo para redibujar a pedido
# $env:SAVEFIG="chart.png"         # si querés PNG del plot inicial

Write-Host "Entorno cargado. Ahora corré:"
Write-Host "python .\velas_TV_sin_sdk.py"
