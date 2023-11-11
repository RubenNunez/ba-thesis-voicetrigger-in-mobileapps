@echo off
setlocal enabledelayedexpansion

:: Define directories
set "BASE_DIR=D:\Projects\ba-thesis-voicetrigger-in-mobileapps\data-wakeup-ConvLSTM-v2\"
set "DIRS=FOOBY Hello_FOOBY Hey_FOOBY Hi_FOOBY OK_FOOBY other other"

:: Define the full path to ffmpeg
set "FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe"

:: Function to convert ogg to wav and then remove the ogg file
:ConvertOggToWav
set "dir_path=%~1"
for %%f in ("%dir_path%\*.ogg") do (
    set "file=%%~f"
    set "out_file=!file:.ogg=.wav!"
    "!FFMPEG_PATH!" -i "!file!" -y "!out_file!"
    echo Converted: !file! -^> !out_file!
    del "!file!"
    echo Removed: !file!
)
exit /b

:: Main script logic
for %%d in (%DIRS%) do (
    call :ConvertOggToWav "%BASE_DIR%\%%d"
)

echo Conversion and removal completed.