@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Python 3.12 is required. Run: py install 3.12
        pause
        exit /b 1
    )
)
echo Preparing ResearchLens. First setup can take a few minutes...
.venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo Package installation failed. Check the error above and your connection.
    pause
    exit /b 1
)
if not exist .env copy .env.example .env >nul
echo Starting ResearchLens. Open AI connection inside the app to connect Groq.
.venv\Scripts\python.exe -m streamlit run app.py
pause
