@echo off
call conda activate torch_ev
if not exist offline_packages mkdir offline_packages
pip download -r requirements_extra.txt -d offline_packages
echo.
echo Download complete. Copy offline_packages folder to target PC.
pause
