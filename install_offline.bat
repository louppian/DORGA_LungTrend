@echo off
call conda activate torch_ev
pip install --no-index --find-links offline_packages -r requirements_extra.txt
echo.
echo Install complete. Run run.bat to start.
pause
