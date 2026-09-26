@echo off
chcp 65001 >nul
rem 在桌面创建「Sampan」快捷方式：图标用 assets\sampan.ico，双击它等于双击「启动.bat」。
rem 只需要运行一次。换了电脑、或者项目文件夹挪了位置，再运行一次。
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=(Get-Location).Path; $bat=Join-Path $root ([string][char]0x542F+[char]0x52A8+'.bat'); $ico=Join-Path $root 'assets\sampan.ico'; $lnk=Join-Path ([Environment]::GetFolderPath('Desktop')) 'Sampan.lnk'; $s=(New-Object -ComObject WScript.Shell).CreateShortcut($lnk); $s.TargetPath=$bat; $s.WorkingDirectory=$root; $s.IconLocation=$ico+',0'; $s.Description='Sampan'; $s.Save(); Write-Host ('OK: '+$lnk)"
if errorlevel 1 (
  echo [错误] 创建失败，请截图上面的信息。
) else (
  echo 已在桌面创建「Sampan」快捷方式。原来那个旧的快捷方式可以直接删掉。
)
pause
