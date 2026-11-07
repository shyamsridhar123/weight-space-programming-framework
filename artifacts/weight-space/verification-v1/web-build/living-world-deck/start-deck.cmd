@echo off
setlocal
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is required to serve this local presentation.
  echo Install Node.js 22 or newer, then open start-deck.cmd again.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:4180/ in your browser after the server starts.
echo Keep this window open while viewing the deck.
node serve.mjs 4180
if errorlevel 1 pause
