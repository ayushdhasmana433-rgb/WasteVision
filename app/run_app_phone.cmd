@echo off
rem Same as run_app.cmd, but phones on the same Wi-Fi can open the app too.
rem On the phone, tap the upload area and choose "Take photo" to use the phone camera.
cd /d "%~dp0"
call run_app.cmd --lan
